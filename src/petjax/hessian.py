"""Positions-Hessian of a PET energy: the PET glue around ``petjax.sparse``
(sparse, via asdex decompression) and chunked forward-over-reverse HVPs (dense).

Only the sparse factory needs the ``sparse`` extra; asdex is imported inside it.

Both factories return a jittable ``hessian(params, positions, structure) ->
(H, overflow)``. The function handed to the AD machinery differentiates a
single ``(N, 3)`` positions array; ``params`` and ``structure`` are arguments
of the returned ``hessian``, captured as tracers under ``jax.jit``, never baked
into the executable as constants. ``overflow`` is PET's k_sel-overflow flag,
returned as the energy's aux; in ``fwd_over_rev`` mode the aux rides the
linearize forward pass, so surfacing it costs nothing.

``UPETCalculator.hessian`` orchestrates these (structure build, selection,
coloring cache, overflow retry, padding strip). The pieces are public so a
training pipeline or a custom driver can compose them directly.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .predict import _select_and_predict

__all__ = [
    "get_dense_hessian_fn",
    "get_energy_fn",
    "get_sparse_hessian_fn",
    "selected_adjacency",
]


# -- energy fn: what gets differentiated --


def get_energy_fn(model, *, no_shadow, num_neighbors_adaptive=None):
    """Build ``energy_fn(params, positions, structure) -> (energy, overflow)``.

    ``energy`` is the scaled model total exactly as ``predict_fn`` computes it,
    with the adaptive selection re-run inside every call over the raw NL in
    ``structure`` (``positions`` override ``structure["positions"]``).
    Composition shifts are linear in atom count and contribute nothing to the
    Hessian, so they are never added.

    ``no_shadow=True`` stop-gradients the adaptive cutoff, confining every
    coupling to the selected neighbour list. The sparse path requires it: the
    sparsity pattern is derived from that list and is wrong with shadow
    coupling. The dense path accepts either.
    """
    if num_neighbors_adaptive is None:
        num_neighbors_adaptive = model.num_neighbors_adaptive

    def energy_fn(params, positions, structure):
        energy, aux = _select_and_predict(
            model,
            params,
            {**structure, "positions": positions},
            num_neighbors_adaptive,
            model.cutoff,
            model.cutoff_width_adaptive,
            model.adaptive_cutoff_method,
            no_shadow=no_shadow,
        )
        return energy, aux["overflow"]

    return energy_fn


# -- sparse: asdex decompression on a colored pattern --


def get_sparse_hessian_fn(energy_fn, coloring, *, chunk_size=None, remat=False):
    """Build a jittable sparse positions-Hessian from a colored pattern.

    Returns ``hessian(params, positions, structure) -> (BCOO (N, 3, N, 3),
    overflow)``. ``coloring.num_colors`` is the per-call HVP count.

    Args:
        energy_fn: From ``get_energy_fn``.
        coloring: An ``asdex.ColoredPattern`` from
            ``petjax.sparse.hessian_coloring``, built on the same padded atom
            count ``N`` as ``structure``.
        chunk_size: Colors processed in parallel per HVP batch; bounds peak AD
            memory. ``None`` runs all colors in one batch.
        remat: Wrap the energy in ``jax.checkpoint``.
    """
    import asdex

    def hessian(params, positions, structure):
        def raw_energy(p):
            return energy_fn(params, p, structure)

        energy = jax.checkpoint(raw_energy) if remat else raw_energy
        hess_fn = asdex.hessian_from_coloring(
            energy, coloring, has_aux=True, chunk_size=chunk_size
        )
        return hess_fn(positions)

    return hessian


# -- dense: chunked HVPs against the identity basis --


def get_dense_hessian_fn(energy_fn, *, chunk_size=None, remat=False):
    """Build a jittable dense positions-Hessian.

    Returns ``hessian(params, positions, structure) -> (H (N, 3, N, 3),
    overflow)``. The per-call HVP count is ``3N``: the "coloring" is the
    identity, one seed per coordinate. Seeds are basis *indices*, with the
    one-hot tangent built inside the HVP, so the ``(3N, 3N)`` identity is never
    materialised.

    Args:
        energy_fn: From ``get_energy_fn``.
        chunk_size: Coordinates (HVPs) processed in parallel per batch; bounds
            peak AD memory. ``None`` runs all ``3N`` in one batch.
        remat: Wrap the energy in ``jax.checkpoint``.
    """

    def hessian(params, positions, structure):
        n = positions.shape[0]
        dim = 3 * n

        def raw_energy(p):
            return energy_fn(params, p, structure)

        energy = jax.checkpoint(raw_energy) if remat else raw_energy

        # fwd-over-rev: linearise the gradient once, then apply its JVP to each
        # identity basis vector. The residuals of `linearize(grad(energy))` are
        # computed once and shared across the whole sweep; `remat` controls
        # whether they are held live or recomputed per HVP.
        _, hvp_fn, aux = jax.linearize(
            jax.grad(energy, has_aux=True), positions, has_aux=True
        )

        def single_hvp(idx):
            tangent = jax.nn.one_hot(idx, dim, dtype=positions.dtype).reshape(n, 3)
            return hvp_fn(tangent).reshape(dim)

        indices = jnp.arange(dim)
        if chunk_size is None or chunk_size >= dim:
            compressed = jax.vmap(single_hvp)(indices)  # (dim, dim)
        else:
            compressed = jax.lax.map(single_hvp, indices, batch_size=chunk_size)
        # compressed[k, i] = H[i, k]; transpose so the seed axis is the column,
        # giving the standard (N, 3, N, 3) layout H[a, alpha, b, beta].
        return compressed.T.reshape(n, 3, n, 3), aux

    return hessian


# -- the pattern's 1-hop adjacency --


def selected_adjacency(structure, selected, n_real):
    """Real-to-real selected pairs of a structure dict: the 1-hop adjacency the
    sparsity pattern is built on.

    Args:
        structure: A ``to_structure`` dict (raw padded NL).
        selected: Its per-pair mask from ``select_edges``.
        n_real: Number of real atoms; slots at or beyond it are padding.

    Returns:
        ``(centers, others)`` host int arrays, both directions of every edge.
    """
    centers = np.asarray(structure["centers"])
    others = np.asarray(structure["others"])
    keep = np.asarray(selected, dtype=bool) & (centers < n_real) & (others < n_real)
    return centers[keep], others[keep]
