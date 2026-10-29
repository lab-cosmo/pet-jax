"""The CuTe DSL kernels against the stock math, on a GPU with ``pet-jax[cute]``."""

import numpy as np
import jax
import jax.numpy as jnp

import functools

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


# -- the window solver --

# (num_neighbors_adaptive, cutoff, cutoff_width_adaptive): PET-MAD-s, the v1.6 xs
# models, the grid-era xs value.
SOLVER_HYPERS = {
    "k24_w1": (24, 8.0, 1.0),
    "k16_w1": (16, 7.5, 1.0),
    "k8_w05": (8, 7.5, 0.5),
}


def _liquid(n=64, rho=0.09, dmin=1.8, seed=0):
    """Random positions at a dense liquid's density with a minimum distance."""
    from ase import Atoms

    rng = np.random.default_rng(seed)
    L = (n / rho) ** (1 / 3)
    positions = []
    while len(positions) < n:
        p = rng.uniform(0, L, 3)
        if all(np.linalg.norm((p - q + L / 2) % L - L / 2) >= dmin for q in positions):
            positions.append(p)
    numbers = rng.choice([40, 8, 8], size=n)
    return Atoms(numbers=numbers, positions=positions, cell=np.eye(3) * L, pbc=True)


@pytest.fixture(scope="module")
def solver_structures(mini_xyz):
    from ase import Atoms
    from ase.io import read

    from petjax import to_structure

    atoms = [read(str(mini_xyz), index=i) for i in (0, 5, 6, 11)]
    atoms += [_liquid(n=40, rho=0.03)]
    atoms += [Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.74], [20, 20, 20]])]
    return [to_structure(a, cutoff=7.5, skin=0.0) for a in atoms]


def _solve(structure, hypers, kernel, capacity=256):
    """Cutoffs, overflow and d(sum_i w_i c_i)/d positions of the window solve."""
    from petjax.kernels.solver import window_cutoffs
    from petjax.select import get_adaptive_cutoffs_window
    from petjax.utils import edge_displacements, safe_norm

    k, cutoff, width = hypers
    N = structure["positions"].shape[0]
    weights = jnp.asarray(np.random.default_rng(1).uniform(0.5, 1.5, N), jnp.float32)
    centers, others = structure["centers"], structure["others"]
    mask = structure["pair_mask"]

    def cutoffs(positions):
        R_ij = edge_displacements(
            positions, centers, others, structure["cell_shifts"], structure["cell"]
        )
        r_ij = safe_norm(R_ij, axis=-1)
        if kernel:
            c, overflow = window_cutoffs(
                r_ij, centers, mask, N, k, cutoff, width, capacity=capacity
            )
        else:
            c, _, overflow = get_adaptive_cutoffs_window(
                centers, r_ij, mask, k, N, cutoff, width
            )
        return jnp.sum(weights * c), (c, overflow)

    positions = jnp.asarray(structure["positions"], jnp.float32)
    grad, (c, overflow) = jax.jit(jax.grad(cutoffs, has_aux=True))(positions)
    return np.asarray(c), bool(overflow), np.asarray(grad)


@pytest.mark.parametrize("hypers", list(SOLVER_HYPERS), ids=list(SOLVER_HYPERS))
def test_solver(hypers, solver_structures):
    """Same cutoffs, overflow and implicit-function gradient as the stock window
    solve, to round-off, on the mini frames, a liquid and an isolated atom next
    to a dimer."""
    for structure in solver_structures:
        c0, o0, g0 = _solve(structure, SOLVER_HYPERS[hypers], kernel=False)
        c1, o1, g1 = _solve(structure, SOLVER_HYPERS[hypers], kernel=True)
        assert o1 == o0 and not o1
        np.testing.assert_allclose(c1, c0, rtol=0, atol=4e-6)
        assert np.all(np.isfinite(g1))
        np.testing.assert_allclose(g1, g0, rtol=0, atol=1e-4 * np.abs(g0).max())


def test_solver_capacity_overflow():
    """An atom with more raw pairs than the kernel holds raises overflow."""
    from petjax import to_structure
    from petjax.kernels.solver import window_cutoffs

    liquid = to_structure(_liquid(n=40, rho=0.05), cutoff=7.5, skin=0.0)
    counts = np.bincount(np.asarray(liquid["centers"])[np.asarray(liquid["pair_mask"])])
    assert 64 < counts.max() <= 256
    r_ij = jnp.linalg.norm(
        jnp.asarray(liquid["positions"])[liquid["others"]]
        - jnp.asarray(liquid["positions"])[liquid["centers"]]
        + liquid["cell_shifts"] @ liquid["cell"],
        axis=-1,
    ).astype(jnp.float32)
    N = liquid["positions"].shape[0]
    args = (r_ij, liquid["centers"], liquid["pair_mask"], N, 16, 7.5, 1.0)
    _, overflow = window_cutoffs(*args, window_capacity=32, capacity=64)
    assert bool(overflow)
    _, overflow = window_cutoffs(*args, capacity=256)
    assert not bool(overflow)


# -- the fused dense layer --

# (M, K, N): the token GEMMs of a ~100-atom system (compress, qkv / glu in), per-atom GEMMs,
# a ragged M, and an N narrower than the default tile
DENSE_SHAPES = [
    (3915, 256, 768),
    (3915, 256, 256),
    (87, 1024, 4096),
    (100, 512, 256),
    (37, 256, 64),
]
DENSE_EPILOGUES = {
    "plain": dict(),
    "bias": dict(bias=True),
    "bias_silu": dict(bias=True, activation="silu"),
    "bias_glu": dict(bias=True, activation="glu"),
    "scale_bias_res_mask": dict(row_scale=True, bias=True, residual=True, row_mask=True),
    "glu_res": dict(activation="glu", residual=True),
    "bias_pre_rank1_res2": dict(
        bias=True, pre_add=True, rank1=True, residual=True, residual2=True
    ),
    "silu_save_res2_mask": dict(
        activation="silu", save_pre=True, residual2=True, row_mask=True
    ),
    "scale_glu_pre_rank1_save": dict(
        row_scale=True,
        bias=True,
        activation="glu",
        pre_add=True,
        rank1=True,
        save_pre=True,
        row_mask=True,
    ),
    "t_scale_bias_res_mask": dict(
        transpose_w=True, row_scale=True, bias=True, residual=True, row_mask=True
    ),
    "t_bias_glu_rank1_save": dict(
        transpose_w=True, bias=True, activation="glu", rank1=True, save_pre=True
    ),
    # the activation's derivative as the epilogue of a backward GEMM
    "dact_silu": dict(dact="silu"),
    "dact_silu_scale_mask": dict(dact="silu", row_scale=True, row_mask=True),
    "dact_glu": dict(dact="glu"),
    "t_dact_glu_scale_mask": dict(
        dact="glu", row_scale=True, row_mask=True, transpose_w=True
    ),
    # an RMSNorm of x folded into the epilogue from its rows' sums of squares
    "rstd_bias_silu": dict(row_rstd=True, bias=True, activation="silu"),
    "t_rstd_glu_res_mask": dict(
        transpose_w=True, row_rstd=True, activation="glu", residual=True, row_mask=True
    ),
}


def dense_inputs(M, K, N, epilogue, seed=0):
    """``x``, the weight and ``fused_dense``'s keyword arguments; with
    ``transpose_w`` the weight is ``W.T`` materialised. The mask drops random rows
    and the last three."""
    k = jax.random.split(jax.random.key(seed), 10)
    N_out = N // 2 if epilogue.get("activation") == "glu" else N
    x = jax.random.normal(k[0], (M, K), jnp.float32)
    w = jax.random.normal(k[1], (K, N), jnp.float32) / np.sqrt(K)
    kw = dict(activation=epilogue.get("activation"))
    if epilogue.get("bias"):
        kw["bias"] = jax.random.normal(k[2], (N,), jnp.float32)
    if epilogue.get("row_scale"):
        kw["row_scale"] = jax.random.uniform(k[3], (M,), jnp.float32, 0.5, 1.5)
    if epilogue.get("row_rstd"):
        # x's own sums of squares, perturbed so the factor is not exactly the norm
        spread = jax.random.uniform(k[3], (M,), jnp.float32, 0.5, 1.5)
        kw["row_rstd"] = jnp.sum(x * x, axis=1) * spread
    if epilogue.get("pre_add"):
        kw["pre_add"] = jax.random.normal(k[6], (M, N), jnp.float32)
    if epilogue.get("rank1"):
        kw["rank1"] = (
            jax.random.normal(k[7], (M,), jnp.float32),
            jax.random.normal(k[8], (N,), jnp.float32),
        )
    if epilogue.get("residual"):
        kw["residual"] = jax.random.normal(k[4], (M, N_out), jnp.float32)
    if epilogue.get("residual2"):
        kw["residual2"] = jax.random.normal(k[9], (M, N_out), jnp.float32)
    if epilogue.get("row_mask"):
        keep = jax.random.bernoulli(k[5], 0.8, (M,)).at[-3:].set(False)
        kw["row_mask"] = keep.astype(jnp.float32)
    for flag in ("save_pre", "row_sumsq", "row_stats"):
        if epilogue.get(flag):
            kw[flag] = True
    if epilogue.get("dact"):
        act = epilogue["dact"]
        h = jax.random.normal(k[6], (M, 2 * N if act == "glu" else N), jnp.float32)
        kw["dact"] = (2 * h, act)
    if epilogue.get("transpose_w"):
        kw["transpose_w"] = True
        w = jnp.array(w.T)
    return x, w, kw


def _check_dense(M, K, N, epilogue, tile, **extra):
    """``fused_dense`` (with ``extra`` arguments, e.g. ``stages`` or ``splits``)
    against the reference; returns its output."""
    from petjax.kernels.dense import fused_dense, reference_dense

    if isinstance(epilogue, str):
        epilogue = DENSE_EPILOGUES[epilogue]
    x, w, kw = dense_inputs(M, K, N, epilogue)
    out = jax.jit(lambda x, w: fused_dense(x, w, tile=tile, **kw, **extra))(x, w)
    ref = reference_dense(x, w, **kw)
    outs, refs = jax.tree.leaves(out), jax.tree.leaves(ref)
    assert len(outs) == len(refs)
    for o, r in zip(outs, refs):
        o, r = np.asarray(o), np.asarray(r)
        assert o.shape == r.shape
        assert np.all(np.isfinite(o))
        np.testing.assert_allclose(o, r, rtol=1e-5, atol=1e-5 * np.abs(r).max())
    if "row_mask" in kw:
        assert np.all(np.asarray(outs[0])[np.asarray(kw["row_mask"]) == 0.0] == 0.0)
    if kw.get("transpose_w"):
        # the same weight stored the other way round: the same FFMAs in the same order
        plain = dict(kw, transpose_w=False)
        again = jax.jit(lambda x, w: fused_dense(x, w, tile=tile, **plain, **extra))(x, w.T)
        for o, a in zip(jax.tree.leaves(out), jax.tree.leaves(again)):
            np.testing.assert_array_equal(np.asarray(o), np.asarray(a))
    return out


@pytest.mark.parametrize("epilogue", list(DENSE_EPILOGUES))
@pytest.mark.parametrize("M,K,N", DENSE_SHAPES)
def test_dense(M, K, N, epilogue):
    _check_dense(M, K, N, epilogue, (64, 64, 8), splits=1)


@pytest.mark.parametrize("tile", [(64, 32, 8), (128, 64, 8)])
@pytest.mark.parametrize("epilogue", ["plain", "bias_glu", "scale_bias_res_mask"])
@pytest.mark.parametrize("M,K,N", [(3915, 256, 768), (37, 256, 64)])
def test_dense_tiles(M, K, N, epilogue, tile):
    _check_dense(M, K, N, epilogue, tile, splits=1)


# cuBLAS's SIMT configurations at these shapes are 64x32x16 and 128x64x16 with 3
# stages and 64x64x8 with 5; 128x64x16 at 6 stages is the largest pipeline
@pytest.mark.parametrize("stages", [3, 4, 5, 6])
@pytest.mark.parametrize("tile", [(64, 64, 8), (64, 32, 16), (64, 64, 16), (128, 64, 16)])
@pytest.mark.parametrize("epilogue", ["scale_bias_res_mask", "t_bias_glu_rank1_save"])
@pytest.mark.parametrize("M,K,N", [(3915, 256, 768), (37, 256, 64)])
def test_dense_pipeline(M, K, N, epilogue, tile, stages):
    _check_dense(M, K, N, epilogue, tile, stages=stages, splits=1)


# per-thread register tiles (tm, tn) other than the default: 128 threads at 64x64
# (8x4) and 128x64 (8x8), 64 at 64x32, 4x8 at 128x64
@pytest.mark.parametrize(
    "tile",
    [
        (64, 64, 8, 8, 4),
        (64, 64, 16, 8, 4),
        (64, 32, 8, 8, 4),
        (128, 64, 16, 4, 8),
        (128, 64, 16, 8, 8),
        (128, 128, 16, 8, 8),
    ],
)
@pytest.mark.parametrize(
    "epilogue", ["scale_bias_res_mask", "t_bias_glu_rank1_save", "t_dact_glu_scale_mask"]
)
@pytest.mark.parametrize("M,K,N", [(3915, 256, 768), (37, 256, 64)])
def test_dense_micro(M, K, N, epilogue, tile):
    _check_dense(M, K, N, epilogue, tile, splits=1)


@pytest.mark.parametrize("splits", [2, 4])
@pytest.mark.parametrize(
    "epilogue",
    [
        "scale_bias_res_mask",
        "t_scale_bias_res_mask",
        "scale_glu_pre_rank1_save",
        "t_bias_glu_rank1_save",
        "t_dact_glu_scale_mask",
    ],
)
@pytest.mark.parametrize("M,K,N", [(88, 2048, 1024), (88, 4096, 1024), (37, 256, 64)])
def test_dense_split(M, K, N, epilogue, splits):
    """Split-K against the reference and against the unsplit kernel."""
    split = _check_dense(M, K, N, epilogue, (64, 32, 8), splits=splits)
    plain = _check_dense(M, K, N, epilogue, (64, 32, 8), splits=1)
    for s, p in zip(jax.tree.leaves(split), jax.tree.leaves(plain)):
        s, p = np.asarray(s), np.asarray(p)
        np.testing.assert_allclose(s, p, rtol=1e-5, atol=1e-5 * np.abs(p).max())


# epilogues for the gradient test: every differentiable input, the save_pre
# output's own cotangent, and the transposed weight
DENSE_GRAD_EPILOGUES = {
    "plain": dict(),
    "scale_bias_silu_res_mask": dict(
        row_scale=True, bias=True, activation="silu", residual=True, row_mask=True
    ),
    "everything_glu": dict(
        row_scale=True,
        bias=True,
        activation="glu",
        pre_add=True,
        rank1=True,
        residual=True,
        residual2=True,
        row_mask=True,
    ),
    "pre_rank1_silu_save_mask": dict(
        activation="silu", pre_add=True, rank1=True, save_pre=True, row_mask=True
    ),
    "t_scale_bias_glu_mask": dict(
        transpose_w=True, row_scale=True, bias=True, activation="glu", row_mask=True
    ),
    "rstd_bias_silu_mask": dict(row_rstd=True, bias=True, activation="silu", row_mask=True),
    "rstd_glu_res": dict(row_rstd=True, activation="glu", residual=True),
}
DIFFERENTIABLE = (
    "bias",
    "row_scale",
    "row_rstd",
    "pre_add",
    "rank1",
    "residual",
    "residual2",
)


@pytest.mark.parametrize("epilogue", list(DENSE_GRAD_EPILOGUES))
# automatic tiles and splits: M <= 128 runs split-K
@pytest.mark.parametrize(
    "M,K,N", [(100, 256, 128), (37, 256, 64), (3915, 256, 768), (88, 2048, 1024)]
)
def test_dense_grad(M, K, N, epilogue):
    _check_dense_grad(M, K, N, DENSE_GRAD_EPILOGUES[epilogue])


def _check_dense_grad(M, K, N, epilogue, **fused_kw):
    """``jax.grad`` through ``fused_dense`` (with ``fused_kw`` on top) against the
    reference, for a random cotangent on every output; returns the gradients."""
    from petjax.kernels.dense import fused_dense, reference_dense

    x, w, kw = dense_inputs(M, K, N, epilogue)
    params = dict(x=x, w=w) | {n: kw.pop(n) for n in DIFFERENTIABLE if n in kw}
    shapes = jax.tree.leaves(jax.eval_shape(lambda p: reference_dense(**p, **kw), params))
    keys = jax.random.split(jax.random.key(1), len(shapes))
    cots = [jax.random.normal(k, s.shape, jnp.float32) for k, s in zip(keys, shapes)]

    def loss(fn):
        def f(p):
            outs = jax.tree.leaves(fn(**p, **kw))
            return sum(jnp.sum(o * c) for o, c in zip(outs, cots))

        return jax.jit(jax.grad(f))

    grads = loss(functools.partial(fused_dense, **fused_kw))(params)
    refs = loss(reference_dense)(params)
    for (path, g), r in zip(jax.tree.leaves_with_path(grads), jax.tree.leaves(refs)):
        g, r = np.asarray(g), np.asarray(r)
        assert g.shape == r.shape, path
        np.testing.assert_allclose(
            g, r, rtol=1e-4, atol=1e-5 * np.abs(r).max(), err_msg=jax.tree_util.keystr(path)
        )
    if "row_mask" in kw and not kw.get("save_pre"):  # h is not masked
        assert np.all(np.asarray(grads["x"])[np.asarray(kw["row_mask"]) == 0.0] == 0.0)
    return grads


@pytest.mark.parametrize(
    "epilogue", ["plain", "scale_bias_silu_res_mask", "everything_glu"]
)
@pytest.mark.parametrize("M,K,N", [(100, 256, 128), (3915, 256, 768)])
def test_dense_w_t(M, K, N, epilogue):
    """A prepared ``W.T`` for the backward gives the same gradients (its own is
    zero), and the mirror case is refused."""
    from petjax.kernels.dense import fused_dense

    ep = DENSE_GRAD_EPILOGUES[epilogue]
    _, w, _ = dense_inputs(M, K, N, ep)
    w_t = jnp.array(w.T)
    plain = _check_dense_grad(M, K, N, ep)
    prepared = _check_dense_grad(M, K, N, ep, w_t=w_t)
    for (path, a), b in zip(jax.tree.leaves_with_path(prepared), jax.tree.leaves(plain)):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(
            a, b, rtol=1e-5, atol=1e-6 * np.abs(b).max(), err_msg=jax.tree_util.keystr(path)
        )
    g = jax.grad(lambda wt: jnp.sum(fused_dense(jnp.ones((M, K)), w, w_t=wt)))(w_t)
    assert np.all(np.asarray(g) == 0.0)
    with pytest.raises(ValueError, match="w_t"):
        fused_dense(jnp.ones((M, N)), w, transpose_w=True, w_t=w)


# row statistics for the next RMSNorm: whole-row tiles from the kernel, split-K
# (M <= 128) from the XLA epilogue
DENSE_STATS_EPILOGUES = {
    "sumsq_res_mask": dict(row_sumsq=True, bias=True, residual=True, row_mask=True),
    "stats_glu_scale": dict(row_stats=True, row_scale=True, activation="glu"),
    "stats_silu_save_res2": dict(
        row_stats=True, activation="silu", save_pre=True, residual2=True
    ),
    # the RMSNorm chain: rstd in from the previous GEMM, sums of squares out
    "rstd_sumsq_res": dict(row_rstd=True, row_sumsq=True, residual=True),
}


@pytest.mark.parametrize("epilogue", list(DENSE_STATS_EPILOGUES))
@pytest.mark.parametrize(
    "M,K,N", [(2304, 256, 256), (2395, 256, 256), (2304, 256, 128), (37, 256, 256)]
)
def test_dense_row_stats(M, K, N, epilogue):
    _check_dense(M, K, N, DENSE_STATS_EPILOGUES[epilogue], None)


@pytest.mark.parametrize(
    "tile", [(64, 256, 8, 8, 8), (32, 256, 8, 4, 8), (16, 256, 8, 4, 8)]
)
def test_dense_row_stats_tiles(tile):
    _check_dense(2395, 256, 256, DENSE_STATS_EPILOGUES["sumsq_res_mask"], tile)


@pytest.mark.parametrize("epilogue", list(DENSE_STATS_EPILOGUES))
@pytest.mark.parametrize("M,K,N", [(300, 256, 256), (37, 256, 128)])
def test_dense_row_stats_grad(M, K, N, epilogue):
    _check_dense_grad(M, K, N, DENSE_STATS_EPILOGUES[epilogue])


# pick_tile's large-M choices: 128x64x16 and 128x128x16 with 2 CTAs per SM for
# N <= 256, 128x128x16 for the K-major weight, the CTA-count rule for glu
DENSE_LARGE = [
    ((8200, 256, 256), "scale_bias_res_mask", (128, 64, 16, 8, 4, 2)),
    ((100_003, 256, 256), "bias", (128, 128, 16, 8, 8, 2)),
    ((8200, 256, 256), "t_scale_bias_res_mask", (128, 128, 16)),
    ((8200, 256, 512), "bias_glu", (128, 128, 16)),
    ((8200, 512, 256), "rstd_bias_silu", (128, 64, 16, 8, 4, 2)),
]


@pytest.mark.parametrize("shape,epilogue,tile", DENSE_LARGE)
def test_dense_large(shape, epilogue, tile):
    from petjax.kernels.dense import pick_tile

    M, K, N = shape
    ep = DENSE_EPILOGUES[epilogue]
    glu = ep.get("activation") == "glu"
    assert pick_tile(M, N // 2 if glu else N, K, ep.get("transpose_w", False), glu) == tile
    _check_dense(M, K, N, epilogue, None)


@pytest.mark.parametrize(
    "tile", [(64, 64, 8), (128, 64, 16), (128, 128, 16), (128, 128, 8), (64, 32, 16, 8, 4)]
)
@pytest.mark.parametrize("epilogue", ["scale_bias_res_mask", "t_bias_glu_rank1_save"])
@pytest.mark.parametrize("M,K,N", [(3915, 256, 768), (37, 256, 64)])
def test_dense_fragment_slots(M, K, N, epilogue, tile, monkeypatch):
    """The k-loop with a two-slot register double buffer instead of the full k-tile."""
    from petjax.kernels import dense

    monkeypatch.setattr(dense, "FRAGMENT_SLOTS", 2)
    _check_dense(M, K, N, epilogue, tile, splits=1)


def test_dense_large_grad():
    """The backward at large M: the forward with 2 CTAs per SM, the dx K-major."""
    _check_dense_grad(8200, 256, 256, DENSE_GRAD_EPILOGUES["scale_bias_silu_res_mask"])


# -- the fused layer: calculator parity --


def test_dense_layer(mini_xyz, assets_dir):
    """The recipe with ``dense="cute"``, ``solver_kernel="cute"`` and the compress
    fold against the recipe with the attention kernel alone, fp32 on the GPU,
    on the mini frames: energies and forces to round-off."""
    from ase.io import read

    from petjax import UPETCalculator
    from petjax.express import Express

    checkpoint = assets_dir / "checkpoints" / "pet-mad-xs-v1.6"
    if not (checkpoint / "model.msgpack").exists():
        pytest.skip("pet-mad-xs-v1.6 checkpoint missing")
    base = Express(
        tokens="flat", solver="sorted_window", reverse_unique=True, attention="cute"
    )
    arm = Express(
        tokens="flat",
        solver="sorted_window",
        reverse_unique=True,
        attention="cute",
        solver_kernel="cute",
        fold_compress=True,
        dense="cute",
    )
    frames = [read(str(mini_xyz), index=i) for i in (0, 6, 11)]
    for express in (base, arm):
        calc = UPETCalculator.from_checkpoint(
            str(checkpoint), express=express, matmul_precision="highest"
        )
        out = []
        for atoms in frames:
            atoms = atoms.copy()
            atoms.calc = calc
            out.append((atoms.get_potential_energy(), atoms.get_forces()))
        if express is base:
            reference = out
    for (e0, f0), (e1, f1) in zip(reference, out):
        n = len(f0)
        assert abs(e1 - e0) < 2e-5 * n, (e0, e1)
        np.testing.assert_allclose(f1, f0, rtol=0, atol=1e-4 * max(1.0, np.abs(f0).max()))


# -- two Dense layers with one backward --

MLP_CASES = {
    # the compress stage (layer 0: no first GEMM, the pre-activation is folded)
    "compress0": dict(activation="silu", x=False, pre_add=True, b1=False, row_mask=True),
    "compress": dict(activation="silu", pre_add=True, b1=False, row_mask=True),
    # the centre MLP, the edge MLP (with the next norm's statistics), the comb MLP
    "center": dict(activation="glu", row_scale=True, residual=True),
    "edge": dict(
        activation="glu", row_scale=True, residual=True, row_mask=True, row_sumsq=True
    ),
    # the edge MLP of the last attention layer: both statistics for the comb LayerNorm
    "edge_stats": dict(
        activation="glu", row_rstd=True, residual=True, row_mask=True, row_stats=True
    ),
    # the same with the norm's statistics from the previous GEMM
    "center_rstd": dict(activation="glu", row_rstd=True, residual=True),
    "edge_rstd": dict(
        activation="glu", row_rstd=True, residual=True, row_mask=True, row_sumsq=True
    ),
    "comb": dict(
        activation="silu",
        row_scale=True,
        rank1=True,
        residual=True,
        residual2=True,
        row_mask=True,
    ),
}
MLP_SHAPES = {
    "compress0": [(2304, 256, 256, 256)],
    "compress": [(2304, 256, 256, 256), (37, 256, 256, 256)],
    "center": [(88, 1024, 2048, 1024), (37, 512, 1024, 512)],
    "edge": [(2304, 256, 512, 256), (2395, 256, 512, 256), (37, 256, 512, 256)],
    "edge_stats": [(2304, 256, 512, 256), (37, 256, 512, 256)],
    "center_rstd": [(88, 1024, 2048, 1024)],
    "edge_rstd": [(2304, 256, 512, 256), (37, 256, 512, 256)],
    "comb": [(2304, 512, 512, 256), (37, 512, 512, 256)],
}


def mlp_inputs(M, K, F, N2, case, seed=0, prepared=False):
    """``fused_mlp``'s arguments for a case: the differentiable arrays in one dict,
    the rest (the mask, flags, prepared transposes) in another."""
    k = jax.random.split(jax.random.key(seed), 12)
    act = case["activation"]
    N1 = 2 * F if act == "glu" else F
    params, kw = {}, dict(activation=act)
    if case.get("x", True):
        params["x"] = jax.random.normal(k[0], (M, K), jnp.float32)
        params["w1"] = jax.random.normal(k[1], (K, N1), jnp.float32) / np.sqrt(K)
        if prepared:
            kw["w1_t"] = jnp.array(params["w1"].T)
    else:
        params["x"] = params["w1"] = None
    if case.get("b1", True):
        params["b1"] = jax.random.normal(k[2], (N1,), jnp.float32)
    else:
        params["b1"] = None
    params["w2"] = jax.random.normal(k[3], (F, N2), jnp.float32) / np.sqrt(F)
    params["b2"] = jax.random.normal(k[4], (N2,), jnp.float32)
    if prepared:
        kw["w2_t"] = jnp.array(params["w2"].T)
    if case.get("row_scale"):
        params["row_scale"] = jax.random.uniform(k[5], (M,), jnp.float32, 0.5, 1.5)
    if case.get("row_rstd"):
        spread = jax.random.uniform(k[5], (M,), jnp.float32, 0.5, 1.5)
        params["row_rstd"] = jnp.sum(params["x"] ** 2, axis=1) * spread
    if case.get("pre_add"):
        params["pre_add"] = jax.random.normal(k[6], (M, N1), jnp.float32)
    if case.get("rank1"):
        params["rank1"] = (
            jax.random.normal(k[7], (M,), jnp.float32),
            jax.random.normal(k[8], (N1,), jnp.float32),
        )
    if case.get("residual"):
        params["residual"] = jax.random.normal(k[9], (M, N2), jnp.float32)
    if case.get("residual2"):
        params["residual2"] = jax.random.normal(k[10], (M, N2), jnp.float32)
    if case.get("row_mask"):
        keep = jax.random.bernoulli(k[11], 0.8, (M,)).at[-3:].set(False)
        kw["row_mask"] = keep.astype(jnp.float32)
    for flag in ("row_sumsq", "row_stats"):
        if case.get(flag):
            kw[flag] = True
    return params, kw


def _mlp_cases():
    return [(name, shape) for name, shapes in MLP_SHAPES.items() for shape in shapes]


@pytest.mark.parametrize("case,shape", _mlp_cases())
def test_mlp(case, shape):
    """``fused_mlp`` against the reference, forward."""
    from petjax.kernels.dense import fused_mlp, reference_mlp

    params, kw = mlp_inputs(*shape, MLP_CASES[case])
    out = jax.jit(lambda p: fused_mlp(**p, **kw))(params)
    ref = reference_mlp(**params, **kw)
    for o, r in zip(jax.tree.leaves(out), jax.tree.leaves(ref)):
        o, r = np.asarray(o), np.asarray(r)
        assert o.shape == r.shape
        np.testing.assert_allclose(o, r, rtol=1e-5, atol=1e-5 * np.abs(r).max())
    if "row_mask" in kw:
        first = np.asarray(jax.tree.leaves(out)[0])
        assert np.all(first[np.asarray(kw["row_mask"]) == 0.0] == 0.0)


@pytest.mark.parametrize("prepared", [False, True], ids=["w", "w_t"])
@pytest.mark.parametrize("case,shape", _mlp_cases())
def test_mlp_grad(case, shape, prepared):
    """``jax.grad`` through ``fused_mlp`` against the reference, for a random
    cotangent on every output, with the weights read as stored or through the
    prepared transposes."""
    from petjax.kernels.dense import fused_mlp, reference_mlp

    params, kw = mlp_inputs(*shape, MLP_CASES[case], prepared=prepared)
    params = {n: a for n, a in params.items() if a is not None}
    fixed = {n: None for n in ("x", "w1", "b1") if n not in params}
    shapes = jax.tree.leaves(
        jax.eval_shape(lambda p: reference_mlp(**p, **fixed, **kw), params)
    )
    keys = jax.random.split(jax.random.key(1), len(shapes))
    cots = [jax.random.normal(k, s.shape, jnp.float32) for k, s in zip(keys, shapes)]

    def loss(fn):
        def f(p):
            outs = jax.tree.leaves(fn(**p, **fixed, **kw))
            return sum(jnp.sum(o * c) for o, c in zip(outs, cots))

        return jax.jit(jax.grad(f))

    grads = loss(fused_mlp)(params)
    refs = loss(reference_mlp)(params)
    for (path, g), r in zip(jax.tree.leaves_with_path(grads), jax.tree.leaves(refs)):
        g, r = np.asarray(g), np.asarray(r)
        assert g.shape == r.shape, path
        np.testing.assert_allclose(
            g, r, rtol=1e-4, atol=1e-5 * np.abs(r).max(), err_msg=jax.tree_util.keystr(path)
        )
