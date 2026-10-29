"""The CuTe DSL kernels against the stock math, on a GPU with ``pet-jax[cute]``."""

import numpy as np
import jax
import jax.numpy as jnp

import pytest

from petjax import kernels

pytestmark = pytest.mark.skipif(
    not kernels.available() or jax.devices()[0].platform != "gpu",
    reason="needs a GPU and the CuTe DSL (pet-jax[cute])",
)

# (tiles, tile width, heads, head dim): PET-MAD-s, PET-MAD-xs, a two-key-per-lane
# edge, a short single-slot tile, and fewer heads.
CASES = [(88, 44, 8, 32), (40, 28, 8, 16), (16, 64, 8, 32), (8, 20, 4, 32), (5, 33, 8, 32)]


def inputs(T, W, H, D, seed=0):
    """Flat tokens and a tile map as the model builds them: tile ``t`` holds its
    centre token ``t`` in slot 0 and a prefix of edge tokens, empty slots point past
    the list, the last tile is a padded atom (all slots masked), and the list ends
    with tokens in no tile (the capacity's slack)."""
    k1, k2, k3, k4 = jax.random.split(jax.random.key(seed), 4)
    F = H * D
    counts = jax.random.randint(k1, (T,), 1, W + 1)
    mask = jnp.arange(W)[None, :] < counts[:, None]
    mask = mask.at[-1].set(False)
    # edge tokens packed tile by tile behind the T centre tokens, then 7 dead tokens
    edges = jnp.maximum(counts - 1, 0)
    offsets = T + jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(edges)[:-1]])
    NP = int(T + edges.sum()) + 7
    col = jnp.arange(W)[None, :]
    rows = jnp.where(col == 0, jnp.arange(T)[:, None], offsets[:, None] + col - 1)
    rows = jnp.where(mask, rows, NP + jnp.arange(T * W).reshape(T, W)).astype(jnp.int32)
    qkv = jax.random.normal(k2, (NP, 3 * F), jnp.float32)
    cutoffs = jax.random.uniform(k3, (T, W), minval=0.0, maxval=1.0)
    cutoffs = jnp.where(mask, cutoffs, 0.0).at[:, 0].set(1.0)
    bias = jnp.log(jnp.clip(cutoffs, 1e-15, None))
    cot = jax.random.normal(k4, (NP, F), jnp.float32)
    return qkv, rows, bias, mask, cot


@pytest.mark.parametrize("T,W,H,D", CASES)
def test_forward(T, W, H, D):
    from petjax.kernels.attention import fused_attention, reference_attention

    qkv, rows, bias, mask, _ = inputs(T, W, H, D)
    out = jax.jit(lambda *a: fused_attention(*a, num_heads=H))(qkv, rows, bias, mask)
    ref = reference_attention(qkv, rows, bias, mask.astype(jnp.float32), H)
    assert np.all(np.isfinite(np.asarray(out)))
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-5, atol=2e-6)
    # tokens in no tile stay zero
    assert np.all(np.asarray(out)[-7:] == 0.0)


@pytest.mark.parametrize("T,W,H,D", CASES)
@pytest.mark.parametrize("backward", ["cute", "jax"])
def test_backward(T, W, H, D, backward):
    from petjax.kernels.attention import fused_attention, reference_attention

    qkv, rows, bias, mask, cot = inputs(T, W, H, D)
    maskf = mask.astype(jnp.float32)

    def loss(f):
        return lambda qkv, bias: jnp.sum(f(qkv, bias) * cot)

    fused = lambda qkv, bias: fused_attention(qkv, rows, bias, mask, H, backward=backward)
    ref = lambda qkv, bias: reference_attention(qkv, rows, bias, maskf, H)
    g = jax.jit(jax.grad(loss(fused), argnums=(0, 1)))(qkv, bias)
    g_ref = jax.jit(jax.grad(loss(ref), argnums=(0, 1)))(qkv, bias)
    for name, a, b in zip(("d_qkv", "d_bias"), g, g_ref):
        a, b = np.asarray(a), np.asarray(b)
        assert np.all(np.isfinite(a)), name
        scale = np.abs(b).max()
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5 * scale, err_msg=name)
    # padded rows (all keys masked) get no gradient on q, k and bias
    assert np.all(np.asarray(g[1])[-1] == 0.0)
    assert np.all(np.asarray(g[0])[-7:] == 0.0)
