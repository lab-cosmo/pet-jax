"""Adaptive-cutoff machine for UPETCalculator — the in-JIT pipeline half.

Given a structure dict (from ``structure.to_structure``), produce a
``truncated`` dict keyed to ``UPET.__call__``'s parameters so the model
forward is ``model.apply(params, **truncated)``.

``truncate`` splits at the displacement boundary: it derives ``R_ij`` via
``edge_displacements`` (single structure, one cell) and delegates to
``truncate_edges``, the layout-agnostic selection+pack core. Consumers with
precomputed per-edge displacements — e.g. training pipelines batching several
structures, where no single cell exists — call ``truncate_edges`` directly.
``pack_edges`` is the selection-free sibling: fixed-width packing of every
unmasked pair, for NLs already trimmed upstream (e.g. a kNN pass).

Two distinct JIT contexts touch this module:
  - ``_k_sel_kernel`` (``@jax.jit``, the sizing path) — CPU-pinned by
    ``determine_k_sel``, runs once per NL-rebuild.
  - ``truncate`` is **undecorated** and traced by ``predict.predict_fn`` —
    runs on the default device, every step.

Rule: do not ``@jax.jit`` the traced helpers below — that nests jit inside
``predict_fn``. Decorate only entry points.
"""

import jax
import jax.numpy as jnp

from functools import partial

from .utils import cutoff_bump, edge_displacements, safe_norm

# -- truncate: structure -> (truncated dict, overflow) — the per-step entry --


def truncate(
    structure,
    num_neighbors_adaptive,
    cutoff,
    cutoff_width_adaptive,
    method="grid",
    no_shadow=False,
):
    """Adaptive selection + pack for a single-structure dict: derive ``R_ij``
    from ``(positions, cell_shifts, cell)``, read ``k_sel`` off the
    ``k_sel_sizer`` carrier, delegate to ``truncate_edges``."""
    R_ij = edge_displacements(
        structure["positions"],
        structure["centers"],
        structure["others"],
        structure["cell_shifts"],
        structure["cell"],
    )
    return truncate_edges(
        R_ij,
        structure["centers"],
        structure["others"],
        structure["reverse"],
        structure["pair_mask"],
        structure["species"],
        structure["atom_mask"],
        structure["k_sel_sizer"].shape[-1],
        num_neighbors_adaptive,
        cutoff,
        cutoff_width_adaptive,
        method=method,
        no_shadow=no_shadow,
    )


def truncate_edges(
    R_ij,
    centers,
    others,
    reverse,
    pair_mask,
    species,
    atom_mask,
    k_sel,
    num_neighbors_adaptive,
    cutoff,
    cutoff_width_adaptive,
    method="grid",
    no_shadow=False,
):
    """Adaptive cutoff selection on a flat NL with precomputed displacements:
    pack survivors into the rectangular ``[N * k_sel]`` layout, return the
    truncated dict keyed to ``UPET.__call__``'s parameter names (so the forward
    is ``model.apply(params, **truncated)``).

    ``cutoff`` is the trained maximum cutoff and ``cutoff_width_adaptive`` the
    selection taper width only — the final ``cutoff_bump`` taper inside the
    model runs on ``cutoff_width``. Both are Python floats (trace-time
    constants), as is the ``method`` choice ("grid" or "solver").

    Layout-agnostic: works on a single structure or on several concatenated
    ones (atoms sample-contiguous, padding at the tail), since the selection
    couples pairs only through their center/other atoms. Requires ``centers``
    non-decreasing; ``k_sel`` must be a static int under jit.
    """
    N = species.shape[0]

    pair_cutoffs, selected = _select_edges(
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
    )
    slot, sel_to_pair, pair_mask_sel, overflow = _pack_selected_to_flat(
        selected, centers, N, k_sel
    )

    # Reverse map in P_sel space. Under overflow a pair can fit while its
    # reverse does not; then slot[reverse] = P_sel - 1 (the masked sentinel).
    truncated = {
        "R_ij": R_ij[sel_to_pair],
        "centers": centers[sel_to_pair],
        "neighbors": others[sel_to_pair],
        "species": species,
        "reverse": slot[reverse[sel_to_pair]],
        "pair_mask": pair_mask_sel,
        "atom_mask": atom_mask,
        "pair_cutoffs": pair_cutoffs[sel_to_pair],
    }
    return truncated, overflow


def pack_edges(R_ij, centers, others, reverse, pair_mask, species, atom_mask, k):
    """Fixed-width pack on a flat NL with precomputed displacements — no
    selection: every unmasked pair goes into the rectangular ``[N * k]``
    layout. Returns the truncated dict keyed like ``truncate_edges``'s, with
    ``pair_cutoffs=None`` (the model applies its plain scalar-cutoff bump).

    ``overflow`` is True iff a center holds more than ``k`` unmasked pairs —
    that pair is silently dropped, so trim upstream to ``<= k`` per center.
    Same layout requirements as ``truncate_edges``: ``centers``
    non-decreasing, ``k`` a static int under jit.
    """
    N = species.shape[0]
    slot, sel_to_pair, pair_mask_sel, overflow = _pack_selected_to_flat(
        pair_mask, centers, N, k
    )
    truncated = {
        "R_ij": R_ij[sel_to_pair],
        "centers": centers[sel_to_pair],
        "neighbors": others[sel_to_pair],
        "species": species,
        "reverse": slot[reverse[sel_to_pair]],
        "pair_mask": pair_mask_sel,
        "atom_mask": atom_mask,
        "pair_cutoffs": None,
    }
    return truncated, overflow


# -- k_sel sizing: CPU-pinned standalone jit kernel; called via determine_k_sel --


def determine_k_sel(
    structure,
    num_neighbors_adaptive,
    cutoff,
    cutoff_width_adaptive,
    method="grid",
):
    """Trial adaptive cutoff to size k_sel. Runs the sizing kernel on CPU (the
    structure dict is moved with one ``jax.device_put``). Kept off the GPU to
    avoid contention with the forward and to read the result back without a
    device→host sync.

    ``jax.devices("cpu")`` is resolved lazily here — *not* at module import —
    so a process that only imports ``petjax`` (e.g. a grain pool worker doing
    preprocessing) never triggers a JAX backend init at import time. The cheap
    repeated lookup is amortised by ``determine_k_sel`` running at most once per
    NL rebuild.

    Returns ``(k_sel, max_cutoff)``: the host-int k_sel and the largest selected
    adaptive cutoff (host float) — the selection's "real reach", for tuning
    ``cutoff_override``.
    """
    cpu = jax.devices("cpu")[0]
    cpu_structure = jax.device_put(structure, cpu)
    max_count, max_cutoff = _k_sel_kernel(
        cpu_structure, num_neighbors_adaptive, cutoff, cutoff_width_adaptive, method=method
    )
    return max(int(max_count), 1), float(max_cutoff)


# jit on the inner k_sel kernel is fine in steady state: everything but the
# structure is a per-model constant (hence static args — the probe grid is
# built from cutoff / width at trace time), so the kernel compiles once per
# Calculator and is reused. Across-shape calls (e.g. a Calculator reused on
# different-sized structures) re-compile.
@partial(
    jax.jit,
    static_argnames=(
        "num_neighbors_adaptive",
        "cutoff",
        "cutoff_width_adaptive",
        "method",
    ),
)
def _k_sel_kernel(
    structure,
    num_neighbors_adaptive,
    cutoff,
    cutoff_width_adaptive,
    method="grid",
):
    R_ij = edge_displacements(
        structure["positions"],
        structure["centers"],
        structure["others"],
        structure["cell_shifts"],
        structure["cell"],
    )
    pair_cutoffs, selected = _select_edges(
        R_ij,
        structure["centers"],
        structure["others"],
        structure["pair_mask"],
        structure["positions"].shape[0],
        num_neighbors_adaptive,
        cutoff,
        cutoff_width_adaptive,
        method=method,
    )
    counts = jax.ops.segment_sum(
        selected.astype(int),
        structure["centers"],
        num_segments=structure["positions"].shape[0],
        indices_are_sorted=True,
    )
    # Largest adaptive cutoff among selected pairs — the real reach of the
    # selection. 0.0 if nothing is selected (degenerate: isolated atom).
    max_cutoff = jnp.max(jnp.where(selected, pair_cutoffs, 0.0))
    return counts.max(), max_cutoff


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
    centers, r_ij, pair_mask, num_neighbors, num_atoms, cutoff, cutoff_width
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
    """
    inv_cutoff = 1.0 / cutoff
    r_ij_d = jax.lax.stop_gradient(r_ij)

    # Bracket [r_lo, r_hi] with f(r_lo) <= 0 <= f(r_hi): n_total(0) = 0 and the
    # baseline alone reaches num_neighbors at r = cutoff.
    r_lo = jnp.zeros(num_atoms, dtype=r_ij.dtype)
    r_hi = jnp.full(num_atoms, cutoff, dtype=r_ij.dtype)

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
        r, r_ij_d, centers, pair_mask, num_atoms, cutoff_width, inv_cutoff, num_neighbors
    )

    # IFT step: r and dn_root are constants; gradients attach only through the
    # residual (live r_ij). The clamps mirror upstream: the derivative floor
    # bounds the correction in pathological geometries, cutoff/16 is the
    # physical lower bound on the adapted cutoff.
    n_residual = (
        _n_total(
            r, r_ij, centers, pair_mask, num_atoms, cutoff_width, inv_cutoff, num_neighbors
        )
        - num_neighbors
    )
    return jnp.clip(r - n_residual / jnp.clip(dn_root, 1e-6, None), cutoff / 16, cutoff)


def _n_total(
    r_per_atom, r_ij, centers, pair_mask, num_atoms, cutoff_width, inv_cutoff, num_neighbors
):
    """Smoothed neighbor count plus cubic baseline, evaluated at per-atom
    cutoff ``r_per_atom``. Padded pairs are masked out (upstream has no
    padding; the mask is the only deviation)."""
    per_edge = cutoff_bump(r_ij, r_per_atom[centers], cutoff_width) * pair_mask
    n = jax.ops.segment_sum(per_edge, centers, num_atoms)
    x = r_per_atom * inv_cutoff
    return n + num_neighbors * x**3


def _n_total_and_dn_dr(
    r_per_atom, r_ij, centers, pair_mask, num_atoms, cutoff_width, inv_cutoff, num_neighbors
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

    n = jax.ops.segment_sum(f, centers, num_atoms)
    dn = jax.ops.segment_sum(df_dr, centers, num_atoms)

    x = r_per_atom * inv_cutoff
    n = n + num_neighbors * x**3
    dn = dn + 3.0 * num_neighbors * x**2 * inv_cutoff
    return n, dn


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
):
    """Shared selection core: consumed by ``_k_sel_kernel`` (sizing) and
    ``truncate_edges`` (forward). Returns ``(pair_cutoffs, selected)``.

    ``method`` picks the per-atom cutoff algorithm (a trace-time branch);
    grid and solver take the same arguments."""
    r_ij = safe_norm(R_ij, axis=-1)
    if method == "grid":
        adaptive_fn = get_adaptive_cutoffs
    elif method == "solver":
        adaptive_fn = get_adaptive_cutoffs_solver
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
    return pair_cutoffs, selected


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

    return slot, sel_to_pair, pair_mask_sel, overflow
