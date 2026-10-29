"""The ``sorted_window`` adaptive-cutoff solve as CuTe DSL kernels, forward and backward.

The stock ``select.get_adaptive_cutoffs_window`` sorts the whole pair list
twice (two radix sorts), gathers each atom's bracket and window, and runs the
Newton-bisection as a handful of XLA fusions per step, with the
implicit-function derivative as their transpose. Here one kernel per direction
does it with a warp per atom (``_solver_cute``): the atom's raw pairs are
sorted inside the warp, so no global sort, no gather and no scatter are left,
and the result is the stock window solve to round-off: the same bracket, the
same window, the same ``iterations`` Newton-bisection steps and the same
implicit-function step, summed in another order.

What it does not give: the center-sorted ``SortedPairs`` of the stock variant,
from which the padded token layout packs each row by distance
(``_pack_sorted_candidates``); with the kernel the padded pack is the stock
list-order one, the same set of pairs per row. Shapes are static per compile;
the atom with the most raw pairs must fit ``capacity`` (a power of two up to
256; more raises ``overflow``, like a window too small for the bracket).
"""

import numpy as np
import jax
import jax.numpy as jnp

import functools
import math

WARP = 32


def window_cutoffs(
    r_ij,
    centers,
    pair_mask,
    num_atoms,
    num_neighbors,
    cutoff,
    cutoff_width,
    iterations=4,
    window_capacity=64,
    capacity=256,
):
    """``(cutoffs [N], overflow)`` of the window solve on the center-sorted
    pairs ``r_ij [P]`` (live, fp32) with ``centers [P]`` (non-decreasing) and
    ``pair_mask [P]``, the stock arguments otherwise. Differentiable in
    ``r_ij`` through the implicit-function derivative on the window."""
    if r_ij.dtype != jnp.float32:
        raise TypeError(f"window_cutoffs runs in float32, got {r_ij.dtype}")
    KN = int(math.ceil(num_neighbors)) + 2
    if KN > WARP:
        raise ValueError(
            f"num_neighbors={num_neighbors} needs more than {WARP} nearest pairs"
        )
    if window_capacity % WARP or window_capacity < WARP:
        raise ValueError(
            f"window_capacity must be a multiple of {WARP}, got {window_capacity}"
        )
    if capacity & (capacity - 1) or not WARP <= capacity <= 256:
        raise ValueError(
            f"capacity must be a power of two in [{WARP}, 256], got {capacity}"
        )
    if window_capacity >= capacity:
        raise ValueError(
            "capacity must exceed window_capacity (the window plus its check entry)"
        )
    N = int(num_atoms)
    centers = jnp.asarray(centers)
    bounds = jnp.searchsorted(
        centers, jnp.arange(N + 1, dtype=centers.dtype), side="left", method="scan_unrolled"
    ).astype(jnp.int32)
    starts, counts = bounds[:-1], bounds[1:] - bounds[:-1]
    pmask = jnp.asarray(pair_mask).astype(jnp.int32)
    # r_p = cutoff (1 - p/k)^(1/3), the radius where the baseline leaves room for p
    p = np.arange(KN, dtype=np.float32)
    rp = np.float32(cutoff) * np.cbrt(
        np.clip(np.float32(1.0) - p / np.float32(num_neighbors), 0, 1)
    )
    fn = _fused(
        float(num_neighbors),
        float(cutoff),
        float(cutoff_width),
        int(iterations),
        int(window_capacity),
        int(capacity),
        KN,
    )
    cutoffs, flags = fn(r_ij, pmask, starts, counts, jnp.asarray(rp.astype(np.float32)))
    return cutoffs, jnp.any(flags > 0)


@functools.cache
def _fused(k, cutoff, width, iterations, WC, CAP, KN):
    """The custom_vjp function for one parameter set."""
    constants = dict(
        k=k, cutoff=cutoff, width=width, iterations=iterations, WC=WC, CAP=CAP, KN=KN
    )

    @jax.custom_vjp
    def f(r_ij, pmask, starts, counts, rp):
        cutoffs, flags, _ = forward_call(r_ij, pmask, starts, counts, rp, **constants)
        return cutoffs, flags

    def fwd(r_ij, pmask, starts, counts, rp):
        cutoffs, flags, saved = forward_call(r_ij, pmask, starts, counts, rp, **constants)
        return (cutoffs, flags), (r_ij, starts, counts, saved)

    def bwd(res, cotangents):
        r_ij, starts, counts, saved = res
        d_cutoffs, _ = cotangents
        d_r = backward_call(r_ij, starts, counts, saved, d_cutoffs, WC=WC, width=width)
        zero_int = lambda x: np.zeros(x.shape, dtype=jax.dtypes.float0)
        return (
            d_r,
            zero_int(starts),
            zero_int(starts),
            zero_int(counts),
            jnp.zeros((KN,), jnp.float32),
        )

    f.defvjp(fwd, bwd)
    return f


def forward_call(
    r_ij, pmask, starts, counts, rp, *, k, cutoff, width, iterations, WC, CAP, KN
):
    """The forward kernel: ``(cutoffs, flags, (root, slope, win))``."""
    import cutlass.jax as cjax

    from . import _solver_cute as kern

    N = starts.shape[0]
    f32 = jnp.float32
    cutoffs, root, slope, win, flags = cjax.cutlass_call(
        kern.launch_forward,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((N,), f32),
            jax.ShapeDtypeStruct((N,), f32),
            jax.ShapeDtypeStruct((N,), f32),
            jax.ShapeDtypeStruct((N, WC), jnp.int32),
            jax.ShapeDtypeStruct((N,), f32),
        ),
        use_static_tensors=True,
        CAP=CAP,
        WC=WC,
        KN=KN,
        NA=N,
        k=k,
        cutoff=cutoff,
        width=width,
        iterations=iterations,
    )(r_ij, pmask, starts, counts, rp)
    return cutoffs, flags, (root, slope, win)


def backward_call(r_ij, starts, counts, saved, d_cutoffs, *, WC, width):
    """The backward kernel: ``d r_ij`` from the cutoffs' cotangent."""
    import cutlass.jax as cjax

    from . import _solver_cute as kern

    root, slope, win = saved
    N = starts.shape[0]
    (d_r,) = cjax.cutlass_call(
        kern.launch_backward,
        output_shape_dtype=(jax.ShapeDtypeStruct(r_ij.shape, jnp.float32),),
        use_static_tensors=True,
        WC=WC,
        NA=N,
        width=width,
    )(r_ij, starts, counts, root, slope, win, d_cutoffs)
    return d_r
