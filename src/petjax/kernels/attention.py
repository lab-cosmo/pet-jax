"""Fused attention over per-atom tiles, forward and backward, as CuTe DSL kernels.

The stock path (``model.Attention``) gathers the projected tokens into tiles
``[T, W, H, d]`` and runs ``flax.linen.attention.dot_product_attention`` on
them, which XLA lowers to a dozen kernels per layer in the forward and more in
the backward, plus the gathers and their transposes. Here one kernel per
direction does the whole thing, gather included: it reads the tokens through
the tile map, computes each tile's attention and writes the output tokens in
place, everything in fp32 FFMA with no tensor cores, so the result is the stock
one to round-off.

Semantics, as in ``flax``: ``s = (q / temperature / sqrt(d)) k^T + bias``,
``s = where(mask, s, finfo(f32).min)``, ``p = softmax(s)``, ``out = p v``. A
fully masked row (a padded atom) comes out uniform and finite, and its
gradients on ``q``, ``k`` and ``bias`` are zero. The forward also returns the
softmax's row max and row sum, from which the backward recomputes ``p``
instead of storing it.

Layout: ``qkv [NP, 3 F]`` is the projected flat token list as the ``Dense``
layer emits it (``F = H d``, columns ``[q | k | v]``, head ``h`` in
``h d .. (h + 1) d`` of each), ``rows [T, W]`` the token behind each tile slot
(``model.TokenTiles.rows``; an entry ``>= NP`` is an empty slot, read as
zeros), ``bias [T, W]`` and ``mask [T, W]`` (as fp32 0/1) per key slot. The
result is ``out [NP, F]``, zero for a token in no tile. Every token sits in at
most one slot, so the writes are unique. The padded layout's ``[N, T, F]``
tokens are the same thing with the identity map. Shapes are static per compile.

``fused_attention`` wraps the two kernels in a ``jax.custom_vjp``. A second
derivative through the kernel backward is not available (``backward="jax"``
gives one, through the pure-JAX reference, for tests and for the HVP path).
"""

import numpy as np
import jax
import jax.numpy as jnp

import functools
import math

BIG_NEG = float(np.finfo(np.float32).min)
WARP = 32


# -- the public entry point --


def fused_attention(qkv, rows, bias, mask, num_heads, temperature=1.0, backward="cute"):
    """``[NP, F]`` attention output for the flat tokens ``qkv [NP, 3 F]`` tiled by
    ``rows [T, W]`` (int), with ``bias [T, W]`` (log cutoff) and ``mask [T, W]``
    (bool) per slot, with the ``flax`` semantics above. ``backward`` is
    ``"cute"`` (the backward kernel) or ``"jax"`` (the pure-JAX reference's VJP)."""
    if qkv.dtype != jnp.float32:
        raise TypeError(f"fused_attention runs in float32, got {qkv.dtype}")
    F = qkv.shape[-1] // 3
    if F % num_heads:
        raise ValueError(f"F={F} not divisible by {num_heads} heads")
    W = rows.shape[1]
    if W > 2 * WARP:
        raise ValueError(
            f"fused_attention holds at most {2 * WARP} tokens per tile, got {W}"
        )
    maskf = mask.astype(jnp.float32)
    fn = _fused(num_heads, float(temperature), backward)
    return fn(qkv, rows.astype(jnp.int32), bias, maskf)


@functools.cache
def _fused(num_heads, temperature, backward):
    """The custom_vjp function for one (heads, temperature, backward) setting."""

    @jax.custom_vjp
    def f(qkv, rows, bias, maskf):
        out, _ = _forward(qkv, rows, bias, maskf, num_heads, temperature)
        return out

    def fwd(qkv, rows, bias, maskf):
        out, stats = _forward(qkv, rows, bias, maskf, num_heads, temperature)
        return out, (qkv, rows, bias, maskf, stats)

    def bwd(res, d_out):
        qkv, rows, bias, maskf, stats = res
        if backward == "jax":
            ref = functools.partial(
                reference_attention, num_heads=num_heads, temperature=temperature
            )
            _, vjp = jax.vjp(lambda q, b, m: ref(q, rows, b, m), qkv, bias, maskf)
            d_qkv, d_bias, _ = vjp(d_out)
        else:
            d_qkv, d_bias = _backward(
                qkv, rows, bias, maskf, stats, d_out, num_heads, temperature
            )
        # an integer input's cotangent is a float0 zero
        d_rows = np.zeros(rows.shape, dtype=jax.dtypes.float0)
        return d_qkv, d_rows, d_bias, jnp.zeros_like(maskf)

    f.defvjp(fwd, bwd)
    return f


def reference_attention(qkv, rows, bias, maskf, num_heads, temperature=1.0):
    """The stock math in plain JAX on the same layout, for tests and for
    ``backward="jax"``: the tile gather, ``flax.linen.attention.dot_product_attention``
    with ``q / temperature``, ``bias`` and ``mask``, the scatter back."""
    NP, F3 = qkv.shape
    T, W = rows.shape
    F = F3 // 3
    H = num_heads
    d = F // H
    tiles = qkv.at[rows].get(mode="fill", fill_value=0)
    q, k, v = jnp.split(tiles, 3, axis=-1)
    q = q.reshape(T, W, H, d) / jnp.asarray(temperature, qkv.dtype)
    q = q / jnp.sqrt(jnp.asarray(d, qkv.dtype))
    k = k.reshape(T, W, H, d)
    v = v.reshape(T, W, H, d)
    s = jnp.einsum("tqhd,tkhd->thqk", q, k, precision=jax.lax.Precision.HIGHEST)
    s = s + bias[:, None, None, :]
    s = jnp.where(maskf[:, None, None, :] > 0.5, s, jnp.finfo(qkv.dtype).min)
    p = jax.nn.softmax(s, axis=-1)
    out = jnp.einsum("thqk,tkhd->tqhd", p, v, precision=jax.lax.Precision.HIGHEST)
    out = out.reshape(T * W, F)
    live = (rows < NP).reshape(-1)
    return (
        jnp.zeros((NP, F), qkv.dtype)
        .at[jnp.where(live, rows.reshape(-1), NP)]
        .add(out, mode="drop")
    )


# -- the kernels (imported lazily: cutlass is optional) --


def _forward(qkv, rows, bias, maskf, num_heads, temperature):
    import cutlass.jax as cjax

    from . import _attention_cute as k

    NP, F3 = qkv.shape
    T, W = rows.shape
    F = F3 // 3
    d = F // num_heads
    scale = 1.0 / (temperature * math.sqrt(d))
    out, rowmax, rowsum = cjax.cutlass_call(
        k.launch_forward,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((NP, F), qkv.dtype),
            jax.ShapeDtypeStruct((T, num_heads, W), qkv.dtype),
            jax.ShapeDtypeStruct((T, num_heads, W), qkv.dtype),
        ),
        input_output_aliases={4: 0},
        use_static_tensors=True,
        W=W,
        H=num_heads,
        D=d,
        NP=NP,
        scale=scale,
    )(qkv, rows, bias, maskf, jnp.zeros((NP, F), qkv.dtype))
    return out, (rowmax, rowsum)


def _backward(qkv, rows, bias, maskf, stats, d_out, num_heads, temperature):
    import cutlass.jax as cjax

    from . import _attention_cute as k

    NP, F3 = qkv.shape
    T, W = rows.shape
    F = F3 // 3
    d = F // num_heads
    scale = 1.0 / (temperature * math.sqrt(d))
    d_qkv, d_bias_h = cjax.cutlass_call(
        k.launch_backward,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((NP, F3), qkv.dtype),
            jax.ShapeDtypeStruct((T, num_heads, k.WARPS_PER_CTA, W), qkv.dtype),
        ),
        input_output_aliases={7: 0},
        use_static_tensors=True,
        W=W,
        H=num_heads,
        D=d,
        NP=NP,
        NW=k.WARPS_PER_CTA,
        scale=scale,
    )(qkv, rows, bias, maskf, *stats, d_out, jnp.zeros((NP, F3), qkv.dtype))
    return d_qkv, d_bias_h.sum(axis=(1, 2))
