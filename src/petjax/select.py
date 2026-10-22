"""Adaptive-cutoff machine for UPETCalculator — the in-JIT pipeline half.

Given a structure dict (from ``structure.to_structure``), produce a
``truncated`` dict keyed to ``UPET.__call__``'s parameters so the model
forward is ``model.apply(params, **truncated)``, plus a ``Truncation`` with
the selection's accounting (the host-side ``Selection`` plus ``overflow``).

``truncate`` splits at the displacement boundary: it derives ``R_ij`` via
``edge_displacements`` (single structure, one cell) and delegates to
``truncate_edges``, the layout-agnostic selection+pack core. Consumers with
precomputed per-edge displacements — e.g. training pipelines batching several
structures, where no single cell exists — call ``truncate_edges`` directly.
``pack_edges`` is the selection-free sibling: fixed-width packing of every
unmasked pair, for NLs already trimmed upstream (e.g. a kNN pass).

Two distinct JIT contexts touch this module:
  - ``_select_edges_kernel`` (``@jax.jit``, the host-side path) — CPU-pinned
    by ``select_edges``, which returns the selection as host arrays. The
    calculator sizes ``k_sel`` from it once per NL-rebuild.
  - ``truncate`` is **undecorated** and traced by ``predict.predict_fn`` —
    runs on the default device, every step.

Rule: do not ``@jax.jit`` the traced helpers below — that nests jit inside
``predict_fn``. Decorate only entry points.
"""

import numpy as np
import jax
import jax.numpy as jnp

from functools import partial

from typing import NamedTuple

from .express import Express
from .utils import cutoff_bump, edge_displacements, safe_norm

# Per-atom model inputs a structure dict may carry besides the geometry.
CONDITIONING_KEYS = ("charge", "spin_multiplicity")

# -- truncate: structure -> (truncated dict, Truncation) — the per-step entry --


class Truncation(NamedTuple):
    """What ``truncate_edges`` / ``pack_edges`` selected and whether it fit:
    ``Selection``'s fields as traced arrays, plus ``overflow``. The raw-pair
    arrays index the input flat NL, not the packed layout. ``pack_edges`` has
    no cutoffs (``None``) and selects every unmasked pair."""

    atomic_cutoffs: jax.Array | None  # [N] adaptive cutoff per atom
    pair_cutoffs: jax.Array | None  # [n_pair_padded] mean of the two endpoint cutoffs
    selected: jax.Array  # [n_pair_padded] bool, r_ij <= pair_cutoff; False on padding
    counts: jax.Array  # [N] selected pairs per center, unclipped; k_sel must cover its max
    overflow: jax.Array  # scalar bool, True iff counts.max() > k_sel


def truncate(
    structure,
    num_neighbors_adaptive,
    cutoff,
    cutoff_width_adaptive,
    method="grid",
    no_shadow=False,
    express=None,
):
    """Adaptive selection + pack for a single-structure dict: derive ``R_ij``
    from ``(positions, cell_shifts, cell)``, read ``k_sel`` off the
    ``k_sel_sizer`` carrier, delegate to ``truncate_edges``. Per-atom model
    inputs beyond the geometry (``charge``, ``spin_multiplicity``) pass
    through untouched when present. ``express`` selects the performance
    switches (``Express``); with ``neighbor_list="device"`` the raw list is
    built here from ``positions`` and ``cell`` instead of read off the dict."""
    express = Express.from_any(express)
    if express.neighbor_list == "device":
        raise NotImplementedError("neighbor_list='device' is not implemented yet")
    R_ij = edge_displacements(
        structure["positions"],
        structure["centers"],
        structure["others"],
        structure["cell_shifts"],
        structure["cell"],
    )
    truncated, truncation = truncate_edges(
        R_ij,
        structure["centers"],
        structure["others"],
        structure["reverse"],
        structure["pair_mask"],
        structure["atomic_numbers"],
        structure["atom_mask"],
        structure["k_sel_sizer"].shape[-1],
        num_neighbors_adaptive,
        cutoff,
        cutoff_width_adaptive,
        method=method,
        no_shadow=no_shadow,
        express=express,
    )
    for key in CONDITIONING_KEYS:
        if key in structure:
            truncated[key] = structure[key]
    return truncated, truncation


def truncate_edges(
    R_ij,
    centers,
    others,
    reverse,
    pair_mask,
    atomic_numbers,
    atom_mask,
    k_sel,
    num_neighbors_adaptive,
    cutoff,
    cutoff_width_adaptive,
    method="grid",
    no_shadow=False,
    express=None,
):
    """Adaptive cutoff selection on a flat NL with precomputed displacements:
    pack survivors into the rectangular ``[N * k_sel]`` layout, return the
    truncated dict keyed to ``UPET.__call__``'s parameter names (so the forward
    is ``model.apply(params, **truncated)``) and the ``Truncation``. Its
    ``counts`` is what a driver sizes ``k_sel`` from without leaving jit.

    ``cutoff`` is the trained maximum cutoff and ``cutoff_width_adaptive`` the
    selection taper width only — the final ``cutoff_bump`` taper inside the
    model runs on ``cutoff_width``. Both are Python floats (trace-time
    constants), as is the ``method`` choice ("grid" or "solver").

    Layout-agnostic: works on a single structure or on several concatenated
    ones (atoms sample-contiguous, padding at the tail), since the selection
    couples pairs only through their center/other atoms. Requires ``centers``
    non-decreasing; ``k_sel`` must be a static int under jit.

    ``express`` picks the solver and the token layout (``Express``); the
    ``sorted_window`` solver and the ``flat`` layout pack from the sorted
    candidate prefix instead of ``_pack_selected_to_flat``.
    """
    express = Express.from_any(express)
    N = atomic_numbers.shape[0]

    atomic_cutoffs, pair_cutoffs, selected = _select_edges(
        R_ij,
        centers,
        others,
        pair_mask,
        N,
        num_neighbors_adaptive,
        cutoff,
        cutoff_width_adaptive,
        method=method,
        no_shadow=no_shadow,
        express=express,
    )
    if express.tokens != "padded":
        raise NotImplementedError(f"tokens={express.tokens!r} is not implemented yet")
    slot, sel_to_pair, pair_mask_sel, counts, overflow = _pack_selected_to_flat(
        selected, centers, N, k_sel
    )

    # Reverse map in P_sel space. Under overflow a pair can fit while its
    # reverse does not; then slot[reverse] = P_sel - 1 (the masked sentinel).
    truncated = {
        "R_ij": R_ij[sel_to_pair],
        "centers": centers[sel_to_pair],
        "neighbors": others[sel_to_pair],
        "atomic_numbers": atomic_numbers,
        "reverse": slot[reverse[sel_to_pair]],
        "pair_mask": pair_mask_sel,
        "atom_mask": atom_mask,
        "pair_cutoffs": pair_cutoffs[sel_to_pair],
    }
    return truncated, Truncation(atomic_cutoffs, pair_cutoffs, selected, counts, overflow)


def pack_edges(R_ij, centers, others, reverse, pair_mask, atomic_numbers, atom_mask, k):
    """Fixed-width pack on a flat NL with precomputed displacements — no
    selection: every unmasked pair goes into the rectangular ``[N * k]``
    layout. Returns the truncated dict keyed like ``truncate_edges``'s, with
    ``pair_cutoffs=None`` (the model applies its plain scalar-cutoff bump), and
    a ``Truncation`` without cutoffs whose ``selected`` is ``pair_mask``.

    ``overflow`` is True iff a center holds more than ``k`` unmasked pairs —
    that pair is silently dropped, so trim upstream to ``<= k`` per center.
    Same layout requirements as ``truncate_edges``: ``centers``
    non-decreasing, ``k`` a static int under jit.

    Pairs whose ``reverse`` is masked (an asymmetric NL, e.g. from a per-center
    rank trim upstream) are legal: the model gathers a zero partner message
    for them. That differs from what a checkpoint trained on a symmetric NL
    has seen, and ``overflow`` does not flag it.
    """
    N = atomic_numbers.shape[0]
    slot, sel_to_pair, pair_mask_sel, counts, overflow = _pack_selected_to_flat(
        pair_mask, centers, N, k
    )
    truncated = {
        "R_ij": R_ij[sel_to_pair],
        "centers": centers[sel_to_pair],
        "neighbors": others[sel_to_pair],
        "atomic_numbers": atomic_numbers,
        "reverse": slot[reverse[sel_to_pair]],
        "pair_mask": pair_mask_sel,
        "atom_mask": atom_mask,
        "pair_cutoffs": None,
    }
    return truncated, Truncation(None, None, pair_mask, counts, overflow)


# -- select_edges: the adaptive selection host-side, via a CPU-pinned jit kernel --


class Selection(NamedTuple):
    atomic_cutoffs: np.ndarray  # [N_padded] adaptive cutoff per atom
    pair_cutoffs: np.ndarray  # [n_pair_padded] mean of the two endpoint cutoffs
    selected: np.ndarray  # [n_pair_padded] bool, r_ij <= pair_cutoff; False on padding
    counts: np.ndarray  # [N_padded] selected pairs per center; k_sel is its max


def select_edges(
    structure,
    num_neighbors_adaptive,
    cutoff,
    cutoff_width_adaptive,
    method="grid",
    express=None,
):
    """Run the adaptive selection on a structure dict on CPU and return what it
    produced, as host arrays. The calculator sizes ``k_sel`` from it once per
    NL rebuild; ``selected`` is the model's actual adjacency for anything
    downstream (e.g. a Hessian sparsity pattern). ``express`` picks the solver
    variant, as in the forward, so that sizing and forward agree."""
    # Only what the kernel reads: extra dict keys would change the jit cache key.
    keys = ("positions", "cell", "centers", "others", "cell_shifts", "pair_mask")
    # Resolved per call, not at import: importing petjax must not init a JAX backend.
    cpu = jax.devices("cpu")[0]
    cpu_structure = jax.device_put({k: structure[k] for k in keys}, cpu)
    out = _select_edges_kernel(
        cpu_structure,
        num_neighbors_adaptive,
        cutoff,
        cutoff_width_adaptive,
        method=method,
        express=Express.from_any(express),
    )
    return Selection(*(np.asarray(x) for x in out))


# jit here is fine in steady state: everything but the structure is a
# per-model constant (hence static args — the probe grid is built from
# cutoff / width at trace time), so the kernel compiles once per Calculator
# and is reused. Across-shape calls (e.g. a Calculator reused on
# different-sized structures) re-compile.
@partial(
    jax.jit,
    static_argnames=(
        "num_neighbors_adaptive",
        "cutoff",
        "cutoff_width_adaptive",
        "method",
        "express",
    ),
)
def _select_edges_kernel(
    structure,
    num_neighbors_adaptive,
    cutoff,
    cutoff_width_adaptive,
    method="grid",
    express=None,
):
    centers = structure["centers"]
    N = structure["positions"].shape[0]
    R_ij = edge_displacements(
        structure["positions"],
        centers,
        structure["others"],
        structure["cell_shifts"],
        structure["cell"],
    )
    atomic_cutoffs, pair_cutoffs, selected = _select_edges(
        R_ij,
        centers,
        structure["others"],
        structure["pair_mask"],
        N,
        num_neighbors_adaptive,
        cutoff,
        cutoff_width_adaptive,
        method=method,
        express=express,
    )
    counts = jax.ops.segment_sum(
        selected.astype(int), centers, num_segments=N, indices_are_sorted=True
    )
    return atomic_cutoffs, pair_cutoffs, selected, counts


# -- adaptive per-atom cutoffs --


def get_adaptive_cutoffs(
    centers, r_ij, pair_mask, num_neighbors, num_atoms, cutoff, cutoff_width
):
    """Compute per-atom adaptive cutoffs via probe-based Gaussian selection.
    ``cutoff_width`` here is the checkpoint's ``cutoff_width_adaptive``, which
    also sets the probe spacing — not the final-taper ``cutoff_width``. The
    probe grid is built here from the (static) ``cutoff`` / ``cutoff_width``,
    as upstream does; 0.5 is upstream's minimum probe cutoff."""
    probes = jnp.arange(0.5, cutoff, cutoff_width / 4)
    num_probes = probes.shape[0]

    weights = cutoff_bump(r_ij[None, :], probes[:, None], cutoff_width) * pair_mask[None, :]
    eff = jax.ops.segment_sum(weights.T, centers, num_atoms).T
    eff = eff.T  # [N, num_probes]

    diff = eff - num_neighbors
    x = jnp.linspace(0, 1, num_probes)
    diff = diff + num_neighbors * x**3

    # Centered gradient (matches torch.gradient)
    grad_interior = (diff[:, 2:] - diff[:, :-2]) / 2
    grad_left = diff[:, 1:2] - diff[:, 0:1]
    grad_right = diff[:, -1:] - diff[:, -2:-1]
    width_t = jnp.concatenate([grad_left, grad_interior, grad_right], axis=1)
    width_t = jnp.abs(width_t)
    width_t = jnp.clip(width_t, 1e-12, None)

    logw = -0.5 * (diff / width_t) ** 2
    w = jnp.exp(logw - logw.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)

    return w @ probes


def get_adaptive_cutoffs_solver(
    centers,
    r_ij,
    pair_mask,
    num_neighbors,
    num_atoms,
    cutoff,
    cutoff_width,
    segment_sum=None,
):
    """Per-atom adaptive cutoffs via a Newton-bisection root find on the
    smoothed neighbor count (metatrain's "solver" method, the default for
    recent checkpoints). Solves ``n_total(r) = num_neighbors`` per atom, where
    ``n_total(r) = sum_j bump(r_j, r, w) + num_neighbors * (r / cutoff)**3``;
    the cubic baseline makes ``n_total`` monotonic so the root is unique and
    bracketed by ``[0, cutoff]``.

    The iteration runs on gradient-detached distances; the trailing
    implicit-function-theorem step re-attaches gradients through the residual,
    so the backward never differentiates through the solver loop. As in the
    grid method, ``cutoff_width`` is the checkpoint's ``cutoff_width_adaptive``.

    ``segment_sum`` maps a per-pair array to per-center sums (``x -> [N]``);
    ``None`` is ``jax.ops.segment_sum`` over ``centers`` (``_per_center_sum``
    builds the variants).
    """
    if segment_sum is None:
        segment_sum = _per_center_sum("segment_sum", centers, num_atoms)
    inv_cutoff = 1.0 / cutoff
    r_ij_d = jax.lax.stop_gradient(r_ij)

    # Bracket [r_lo, r_hi] with f(r_lo) <= 0 <= f(r_hi): n_total(0) = 0 and the
    # baseline alone reaches num_neighbors at r = cutoff.
    #
    # Under shard_map, arrays derived from the per-device batch are typed as
    # "varying" over the mesh axis, and fori_loop insists that the carry has
    # the same type going in and coming out. The bracket starts from plain
    # constants (not varying) but the body updates it from r_ij (varying), so
    # without help the loop fails to trace. pcast just re-tags the constants
    # as varying; it computes nothing. The axes are read off r_ij_d, so this
    # is a no-op outside shard_map and does not care what the mesh axis is
    # called.
    varying = tuple(jax.typeof(r_ij_d).manual_axis_type.varying)
    r_lo = jax.lax.pcast(jnp.zeros(num_atoms, dtype=r_ij.dtype), varying, to="varying")
    r_hi = jax.lax.pcast(
        jnp.full(num_atoms, cutoff, dtype=r_ij.dtype), varying, to="varying"
    )

    # 10 iterations converge to fp32 precision (upstream's choice). Newton
    # steps that would leave the bracket (flat shoulders between bumps) fall
    # back to the bracket midpoint. fori_loop keeps the HLO compact (vs. 10x
    # unroll); safe here because the loop sits entirely on detached inputs,
    # so autodiff never needs to enter it.
    def newton_bisection_step(_, carry):
        r_lo, r_hi, r = carry
        n, dn = _n_total_and_dn_dr(
            r,
            r_ij_d,
            centers,
            pair_mask,
            num_atoms,
            cutoff_width,
            inv_cutoff,
            num_neighbors,
            segment_sum,
        )
        f = n - num_neighbors
        below = f <= 0
        r_lo = jnp.where(below, r, r_lo)
        r_hi = jnp.where(below, r_hi, r)
        r_newton = r - f / jnp.clip(dn, 1e-6, None)
        inside = (r_newton >= r_lo) & (r_newton <= r_hi)
        return r_lo, r_hi, jnp.where(inside, r_newton, 0.5 * (r_lo + r_hi))

    _, _, r = jax.lax.fori_loop(0, 10, newton_bisection_step, (r_lo, r_hi, 0.5 * r_hi))
    _, dn_root = _n_total_and_dn_dr(
        r,
        r_ij_d,
        centers,
        pair_mask,
        num_atoms,
        cutoff_width,
        inv_cutoff,
        num_neighbors,
        segment_sum,
    )

    # IFT step: r and dn_root are constants; gradients attach only through the
    # residual (live r_ij). The clamps mirror upstream: the derivative floor
    # bounds the correction in pathological geometries, cutoff/16 is the
    # physical lower bound on the adapted cutoff.
    n_residual = (
        _n_total(
            r,
            r_ij,
            centers,
            pair_mask,
            num_atoms,
            cutoff_width,
            inv_cutoff,
            num_neighbors,
            segment_sum,
        )
        - num_neighbors
    )
    return jnp.clip(r - n_residual / jnp.clip(dn_root, 1e-6, None), cutoff / 16, cutoff)


def _n_total(
    r_per_atom,
    r_ij,
    centers,
    pair_mask,
    num_atoms,
    cutoff_width,
    inv_cutoff,
    num_neighbors,
    segment_sum=None,
):
    """Smoothed neighbor count plus cubic baseline, evaluated at per-atom
    cutoff ``r_per_atom``. Padded pairs are masked out (upstream has no
    padding; the mask is the only deviation)."""
    if segment_sum is None:
        segment_sum = _per_center_sum("segment_sum", centers, num_atoms)
    per_edge = cutoff_bump(r_ij, r_per_atom[centers], cutoff_width) * pair_mask
    n = segment_sum(per_edge)
    x = r_per_atom * inv_cutoff
    return n + num_neighbors * x**3


def _n_total_and_dn_dr(
    r_per_atom,
    r_ij,
    centers,
    pair_mask,
    num_atoms,
    cutoff_width,
    inv_cutoff,
    num_neighbors,
    segment_sum,
):
    """``n_total`` and its analytic d/dr in one pass, for the Newton steps.

    Closed form of the bump in its active region ``s = (d - r + w)/w`` in (0, 1):
    ``f = 0.5 * (1 + tanh(cot(pi s)))``, ``df/dr = (pi / 2w) sech^2(cot(pi s)) /
    sin^2(pi s)``; outside, f saturates to 1 (below) / 0 (above) with df/dr = 0.
    The clamp of ``s`` matches the eps in ``cutoff_bump`` so f agrees
    numerically with the forward pass."""
    scaled = (r_ij - (r_per_atom[centers] - cutoff_width)) / cutoff_width
    active = (scaled > 0.0) & (scaled < 1.0) & pair_mask
    smaller = (scaled <= 0.0) & pair_mask

    safe = jnp.clip(scaled, 1e-6, 1 - 1e-6)
    s = jnp.pi * safe
    sin_s = jnp.sin(s)
    tanh_cot = jnp.tanh(jnp.cos(s) / sin_s)

    f = jnp.where(active, 0.5 * (1.0 + tanh_cot), smaller.astype(scaled.dtype))
    df_dr = jnp.where(
        active,
        (0.5 * jnp.pi / cutoff_width) * (1.0 - tanh_cot**2) / (sin_s * sin_s),
        0.0,
    )

    n = segment_sum(f)
    dn = segment_sum(df_dr)

    x = r_per_atom * inv_cutoff
    n = n + num_neighbors * x**3
    dn = dn + 3.0 * num_neighbors * x**2 * inv_cutoff
    return n, dn


# -- per-center sums over the center-sorted flat list --


def _per_center_sum(solver, centers, num_atoms):
    """``x -> [num_atoms]`` per-center sums for the ``Express.solver`` variant.
    ``centers`` must be non-decreasing for every variant but the stock one."""
    if solver == "segment_sum":
        return partial(jax.ops.segment_sum, segment_ids=centers, num_segments=num_atoms)
    if solver == "sorted_segment_sum":
        return partial(
            jax.ops.segment_sum,
            segment_ids=centers,
            num_segments=num_atoms,
            indices_are_sorted=True,
        )
    if solver == "blocked_prefix_sum":
        queries = jnp.arange(num_atoms + 1)
        bounds = jnp.searchsorted(
            jnp.asarray(centers), queries, side="left", method="scan_unrolled"
        )
        return partial(_blocked_prefix_sum, bounds=bounds)
    raise ValueError(f"no per-center sum for solver {solver!r}")


def _blocked_prefix_sum(x, bounds, block=1024):
    """Sums of ``x`` over the contiguous segments ``[bounds[c], bounds[c + 1])``
    from a two-level prefix sum: an inclusive cumsum inside blocks of ``block``
    entries plus an exclusive cumsum of the block totals.

    A segment spanning at most two blocks (any segment of at most ``block``
    entries) never differences the long prefix: the outer term is the same
    float twice, exactly zero. Its rounding is then bounded by the running sum
    over one block, not over the list. A plain prefix-sum difference is not:
    at the end of a long fp32 list both prefixes are large, and their
    difference loses the low bits of the segment's sum (cutoffs off by ~1e-4 A).
    The bound is still looser than the scatter's when a block holds many short
    segments (a molecule's list fits in one block), by about the number of
    segments per block."""
    P = x.shape[0]
    pad = (-P) % block
    if pad:
        x = jnp.concatenate([x, jnp.zeros(pad, x.dtype)])
    num_blocks = x.shape[0] // block
    inner = jnp.cumsum(x.reshape(num_blocks, block), axis=1)
    totals = inner[:, -1]
    inner = inner.reshape(-1)
    outer = jnp.concatenate([jnp.zeros(1, x.dtype), jnp.cumsum(totals)])

    starts, ends = bounds[:-1], bounds[1:]
    nonempty = ends > starts
    last = jnp.where(nonempty, ends - 1, 0)
    first_block, last_block = starts // block, last // block
    at_boundary = starts % block == 0
    before = jnp.where(at_boundary, 0.0, inner[jnp.where(at_boundary, 0, starts - 1)])
    through = inner[last]
    between = (
        outer[jnp.minimum(last_block, num_blocks)]
        - outer[jnp.minimum(first_block + 1, num_blocks)]
    )
    within = through - before
    across = (totals[jnp.minimum(first_block, num_blocks - 1)] - before) + through + between
    return jnp.where(nonempty, jnp.where(first_block == last_block, within, across), 0.0)


# -- selection mask: adaptive cutoffs, r_ij <= pair_cutoff --


def _select_edges(
    R_ij,
    centers,
    others,
    pair_mask,
    num_atoms,
    num_neighbors_adaptive,
    cutoff,
    cutoff_width_adaptive,
    method="grid",
    no_shadow=False,
    express=None,
):
    """Shared selection core: consumed by ``_select_edges_kernel`` (host-side)
    and ``truncate_edges`` (forward). Returns ``(atomic_cutoffs, pair_cutoffs,
    selected)``.

    ``method`` picks the per-atom cutoff algorithm (a trace-time branch);
    grid and solver take the same arguments. ``express.solver`` picks the
    variant of the solver method; the grid method has none."""
    express = Express.from_any(express)
    r_ij = safe_norm(R_ij, axis=-1)
    if method == "grid":
        adaptive_fn = get_adaptive_cutoffs
    elif method == "solver":
        if express.solver == "sorted_window" or express.seed_cutoffs:
            raise NotImplementedError(
                f"solver={express.solver!r}, seed_cutoffs={express.seed_cutoffs} "
                "is not implemented yet"
            )
        adaptive_fn = partial(
            get_adaptive_cutoffs_solver,
            segment_sum=_per_center_sum(express.solver, centers, num_atoms),
        )
    else:
        raise ValueError(
            f"adaptive_cutoff_method must be 'grid' or 'solver', got {method!r}"
        )
    atomic_cutoffs = adaptive_fn(
        centers,
        r_ij,
        pair_mask,
        num_neighbors_adaptive,
        num_atoms,
        cutoff,
        cutoff_width_adaptive,
    )
    if no_shadow:
        atomic_cutoffs = jax.lax.stop_gradient(atomic_cutoffs)
    pair_cutoffs = (atomic_cutoffs[centers] + atomic_cutoffs[others]) / 2
    selected = (r_ij <= pair_cutoffs) & pair_mask
    return atomic_cutoffs, pair_cutoffs, selected


# -- pack flat selection into the [N_padded * k_sel] rectangular layout --


def _pack_selected_to_flat(selected, centers, N_padded, k_sel):
    """Pack a flat selection mask over sorted-by-center pairs into a flat
    ``[P_sel = N_padded * k_sel]`` array that is logically ``[N_padded, k_sel]``
    (row-major; row index = center, column index = packed position within row).
    All returned arrays are 1-D; reshape to 2-D is the model's choice.

    Sortedness of ``centers`` is required: the cumsum/segment_sum scheme
    that computes the within-row column index for each surviving pair
    relies on it. Padded pairs (``selected = False``) contribute zero to
    every count and never get written to a real slot.

    Returns:
      slot:           ``[N_pair_padded]`` — for selected-and-fitting pairs,
                      slot in ``[P_sel]`` where they go. ``P_sel - 1``
                      otherwise (the masked-out sentinel).
      sel_to_pair:    ``[P_sel]`` — inverse map. Unused P_sel slots point
                      at ``N_pair_padded - 1`` (the padded-pair sentinel).
      pair_mask_sel:  ``[P_sel]`` bool.
      counts:         ``[N_padded]`` — selected pairs per center, unclipped.
      overflow:       scalar bool — True iff any selected pair didn't fit
                      because its center already had ≥ ``k_sel`` selected.
    """
    N_pair_padded = selected.shape[0]
    P_sel = N_padded * k_sel

    sel_int = selected.astype(int)
    count_per_center = jax.ops.segment_sum(
        sel_int, centers, num_segments=N_padded, indices_are_sorted=True
    )
    # prefix[c] = number of selected pairs with center < c
    prefix = jnp.concatenate(
        [jnp.zeros(1, count_per_center.dtype), jnp.cumsum(count_per_center)[:-1]]
    )
    global_csum = jnp.cumsum(sel_int)
    # Column index within row centers[p] (valid where selected[p]).
    column = global_csum - 1 - prefix[centers]

    fits = selected & (column < k_sel)
    overflow = jnp.any(selected & (column >= k_sel))

    slot = jnp.where(fits, centers * k_sel + column, P_sel - 1)

    counts_clipped = jnp.minimum(count_per_center, k_sel)
    pair_mask_sel = (jnp.arange(k_sel)[None, :] < counts_clipped[:, None]).reshape(P_sel)

    sel_to_pair = jnp.full(P_sel, N_pair_padded - 1, dtype=centers.dtype)
    sel_to_pair = sel_to_pair.at[slot].set(jnp.arange(N_pair_padded, dtype=centers.dtype))

    return slot, sel_to_pair, pair_mask_sel, count_per_center, overflow
