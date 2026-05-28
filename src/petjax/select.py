"""Adaptive-cutoff machine for UPETCalculator — the in-JIT pipeline half.

Given a structure dict (from ``structure.to_structure``), produce a
``truncated`` dict keyed to ``UPET.__call__``'s parameters so the model
forward is ``model.apply(params, **truncated)``.

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


def truncate(structure, probes, cutoff_width, num_neighbors_adaptive, no_shadow=False):
    """Run the adaptive cutoff selection, pack survivors into the rectangular
    ``[N_padded * k_sel]`` layout, return the truncated dict keyed to
    ``UPET.__call__``'s parameter names (so the forward is
    ``model.apply(params, **truncated)``).
    """
    centers = structure["centers"]
    others = structure["others"]
    reverse_pair = structure["reverse"]
    N_padded = structure["positions"].shape[0]
    k_sel = structure["k_sel_sizer"].shape[-1]

    R_ij_flat, pair_cutoffs_flat, selected = _selection(
        structure, probes, cutoff_width, num_neighbors_adaptive, no_shadow=no_shadow
    )
    slot, sel_to_pair, pair_mask_sel, overflow = _pack_selected_to_flat(
        selected, centers, N_padded, k_sel
    )

    # Reverse map in P_sel space. Under overflow a pair can fit while its
    # reverse does not; then slot[reverse] = P_sel - 1 (the masked sentinel).
    truncated = {
        "R_ij": R_ij_flat[sel_to_pair],
        "centers": centers[sel_to_pair],
        "neighbors": others[sel_to_pair],
        "species": structure["species"],
        "reverse": slot[reverse_pair[sel_to_pair]],
        "pair_mask": pair_mask_sel,
        "atom_mask": structure["atom_mask"],
        "pair_cutoffs": pair_cutoffs_flat[sel_to_pair],
    }
    return truncated, overflow


# -- k_sel sizing: CPU-pinned standalone jit kernel; called via determine_k_sel --


def determine_k_sel(structure, probes, num_neighbors_adaptive, cutoff_width):
    """Trial adaptive cutoff to size k_sel. Runs the sizing kernel on CPU (the
    structure dict is moved with one ``jax.device_put``), returns a host int.
    Kept off the GPU to avoid contention with the forward and to read out the
    int without a device→host sync.

    ``jax.devices("cpu")`` is resolved lazily here — *not* at module import —
    so a process that only imports ``petjax`` (e.g. a grain pool worker doing
    preprocessing) never triggers a JAX backend init at import time. The
    cheap repeated lookup is amortised by ``determine_k_sel`` running at most
    once per NL rebuild.
    """
    cpu = jax.devices("cpu")[0]
    cpu_structure = jax.device_put(structure, cpu)
    cpu_probes = jax.device_put(probes, cpu)
    max_count = _k_sel_kernel(
        cpu_structure, cpu_probes, cutoff_width, num_neighbors_adaptive
    )
    return max(int(max_count), 1)


# jit on the inner k_sel kernel is fine in steady state: in MD the static arg
# num_neighbors_adaptive and the input shapes are constant across steps, so the
# kernel compiles once per Calculator and is reused. Across-shape calls (e.g. a
# Calculator reused on different-sized structures) re-compile.
@partial(jax.jit, static_argnames=("num_neighbors_adaptive",))
def _k_sel_kernel(structure, probes, cutoff_width, num_neighbors_adaptive):
    _, _, selected = _selection(structure, probes, cutoff_width, num_neighbors_adaptive)
    counts = jax.ops.segment_sum(
        selected.astype(int),
        structure["centers"],
        num_segments=structure["positions"].shape[0],
        indices_are_sorted=True,
    )
    return counts.max()


# -- adaptive per-atom cutoffs --


def get_adaptive_cutoffs(
    centers, r_ij, pair_mask, num_neighbors, num_atoms, probes, cutoff_width
):
    """Compute per-atom adaptive cutoffs via probe-based Gaussian selection."""
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


# -- selection mask: displacements, adaptive cutoffs, r_ij <= pair_cutoff --


def _selection(structure, probes, cutoff_width, num_neighbors_adaptive, no_shadow=False):
    """Shared selection: consumed by ``_k_sel_kernel`` (sizing) and ``truncate``
    (forward). Returns ``(R_ij, pair_cutoffs, selected)``."""
    positions = structure["positions"]
    centers = structure["centers"]
    others = structure["others"]
    pair_mask = structure["pair_mask"]
    N_padded = positions.shape[0]

    R_ij = edge_displacements(
        positions, centers, others, structure["cell_shifts"], structure["cell"]
    )
    r_ij = safe_norm(R_ij, axis=-1)
    atomic_cutoffs = get_adaptive_cutoffs(
        centers, r_ij, pair_mask, num_neighbors_adaptive, N_padded, probes, cutoff_width
    )
    if no_shadow:
        atomic_cutoffs = jax.lax.stop_gradient(atomic_cutoffs)
    pair_cutoffs = (atomic_cutoffs[centers] + atomic_cutoffs[others]) / 2
    selected = (r_ij <= pair_cutoffs) & pair_mask
    return R_ij, pair_cutoffs, selected


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
