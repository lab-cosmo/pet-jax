"""Dense layers with a fused epilogue as a CuTe DSL kernel, with a backward.

The model's token MLPs are a ``Dense`` followed by elementwise work (an RMSNorm's
row scale before, a bias, SiLU or SwiGLU, a residual add and the padding mask
after), which XLA runs as a GEMM plus a few memory-bound passes over ``[M, N]``.
``fused_dense`` does it in one fp32 FFMA GEMM whose epilogue applies all of it
before the single store; the semantics are ``reference_dense``. The kernel and
its constraints are documented in ``_gemm_cute``.

The backward is a ``jax.custom_vjp``: ``dx`` is one more ``fused_dense`` call
(the forward weight read transposed, ``transpose_w`` flipped, the row scale in
its epilogue), the weight and vector cotangents are plain XLA, and dead at
inference. There is no second derivative.

``dact`` is the piece a layer-level backward needs to drop the remaining
elementwise pass: for two consecutive layers ``L`` and ``L + 1`` (``L + 1``
consuming ``L``'s output), the dx GEMM of ``L + 1`` produces the cotangent of
``L``'s output, and ``L``'s backward multiplies it by its mask and ``act'(h_L)``.
``fused_dense(dh_next, w_next, transpose_w=True, row_scale=s_next,
dact=(h_L, act_L), row_mask=mask_L)`` gives ``dh_L`` in one call. The
``custom_vjp`` here sees one layer at a time, so it cannot do this itself; a
backward written for the whole block (and keeping the plain cotangent where a
residual branch also needs it) can.
"""

import jax
import jax.numpy as jnp

import functools

ACTIVATIONS = (None, "silu", "glu")
ALGORITHMS = ("simt", "tf32x3", "tf32")
HIGHEST = jax.lax.Precision.HIGHEST
RSTD_EPS = float(jnp.finfo(jnp.float32).eps)
# the tensor-core kernel: split x into its TF32 high and low parts in the kernel
# (False: the low part as an XLA fusion and a second operand, to compare); split K
# for M <= 128 (False: one split); the row count from which its row statistics come
# from the kernel, below from XLA
# (both in-kernel splits, warp 0 alone and warps 2-3 with their own mbarrier, measured
# slower than XLA's split at every size tried)
TF32_SPLIT_A = False
TF32_SPLIT_K = True
# split K above M = 128 too, from this K on (0: only M <= 128); glu's tile width
# (0: the general rule); the shared memory one CTA's pipeline may take
TF32_SPLIT_MIN_K = 1024
TF32_GLU_BN = 64
# (110 KB, two CTAs per SM at bN 64, measured slower almost everywhere)
TF32_SMEM_LIMIT = 200 * 1024
TF32_STATS_MIN_ROWS = 131072
# 16-byte loads and stores in the kernel's epilogue (False: scalar ones, to compare)
VECTOR_EPILOGUE = True
# CTAs per SM ptxas should leave room for (it caps the registers; 0: no hint)
MIN_BLOCKS_PER_SM = 0
# register slots of the k-loop's operand fragments: 0 all of the k-tile (ptxas
# hoists the shared loads freely: 178 registers at 128x128x16, one CTA per SM),
# 2 a true double buffer (fewer registers, to compare)
FRAGMENT_SLOTS = 0
# the optional inputs, in the kernel's argument order
OPTIONAL = (
    "bias",
    "row_scale",
    "pre_add",
    "rank1_row",
    "rank1_col",
    "residual",
    "residual2",
    "row_mask",
)


def fused_dense(
    x,
    w,
    bias=None,
    *,
    row_scale=None,
    activation=None,
    pre_add=None,
    rank1=None,
    residual=None,
    residual2=None,
    row_mask=None,
    row_rstd=None,
    rstd_eps=RSTD_EPS,
    transpose_w=False,
    save_pre=False,
    row_sumsq=False,
    row_stats=False,
    w_t=None,
    dact=None,
    tile=None,
    stages=3,
    splits=None,
    algorithm="simt",
    w_lo=None,
    w_t_lo=None,
):
    """``epilogue(x @ W)`` for ``x [M, K]`` and the weight ``W [K, N]``, passed as
    ``w = W`` or, with ``transpose_w``, as ``w = W.T [N, K]`` (no copy either way).
    The epilogue: ``h = (x @ W) * row_scale[:, None] + bias + pre_add
    + rank1[0][:, None] * rank1[1]``, the activation (``None``, ``"silu"`` or
    ``"glu"``, the SwiGLU ``v * sigmoid(g)`` of the two column halves of ``h``),
    ``+ residual + residual2``, ``* row_mask[:, None]``. ``bias``, ``pre_add`` and
    ``rank1`` span ``N``, the residuals the output width (``N // 2`` for ``"glu"``).
    Returns ``[M, N]`` or ``[M, N // 2]``, and with ``save_pre`` also ``h [M, N]``.

    ``row_rstd [M]`` replaces ``row_scale`` by an RMSNorm of ``x`` folded into the
    epilogue: it holds each row's sum of squares (a previous call's ``row_sumsq``)
    and the row factor is ``rsqrt(row_rstd / K + rstd_eps)``.

    ``row_sumsq`` adds ``sum(out**2, axis=1)`` ``[M]`` to the outputs (for the next
    RMSNorm's ``rsqrt(mean + eps)``), ``row_stats`` both ``sum(out, axis=1)`` and
    that, from the epilogue: the CTA owns whole rows, so the output width must be
    32, 64, 128 or 256 (``tile=None`` picks ``(32, 256, 8, 4, 8)`` and the like;
    an explicit tile needs ``bN`` equal to it). ``row_stats`` is for norms of one
    GEMM's output: the comb LayerNorm over a concatenation cannot use it. Both are
    differentiable (their cotangents add ``dsum + 2 out dsumsq`` to ``out``'s).
    Outputs come as ``(out, h, sum, sumsq)``, each extra only when asked for.

    ``w_t`` is an optional copy of ``W.T`` as a row-major ``[N, K]`` array for the
    backward (``transpose_w=False`` only): its dx GEMM then reads the weight
    N-major (the fast path) instead of ``w`` K-major. The forward ignores it, it
    gets a zero cotangent (``dW`` comes out through ``w``), so pass it as a
    precomputed, stop-gradient'ed array.

    ``dact=(h, act)`` replaces the epilogue by an activation's derivative for a
    backward GEMM: ``(x @ W) * row_scale[:, None] * act'(h) * row_mask[:, None]``
    with ``h`` a saved pre-activation, ``[M, N]`` for ``"silu"``, ``[M, 2 N]`` for
    ``"glu"``, whose output is then ``[M, 2 N]`` (the value and gate halves'
    cotangents); no other epilogue input, forward only (not differentiable).

    ``algorithm`` picks the GEMM: ``"simt"`` the fp32 FFMA kernel (``_gemm_cute``),
    ``"tf32x3"`` the SM100 tensor-core kernel (``_gemm_tf32_cute``) in 3xTF32,
    fp32-class, or ``"tf32"`` the same in one TF32 pass. For 3xTF32 ``w_lo`` is the
    weight's low part ``w - tf32(w)`` in ``w``'s storage (``tf32_low_part``,
    computed per call when absent) and ``x``'s is computed per call; the tensor-core
    kernel reads ``w`` either way round at the same speed, so its backward needs
    neither ``w_t`` nor ``w_t_lo`` (accepted, unused). ``tile`` is then ``(128, bN,
    bK)`` (``None``: ``pick_tile_tf32``); ``stages`` and ``splits`` are its own
    (``pick_splits_tf32``: split-K for ``M <= 128``).

    All fp32; ``K`` and ``N`` multiples of 32, ``M`` arbitrary. ``tile`` is the CTA
    tile ``(bM, bN, bK)``, optionally with the per-thread register tile ``(bM, bN,
    bK, tm, tn)`` (``tm, tn`` in 4, 8; else ``default_micro``) and the CTAs per SM
    ptxas should allow for, ``(bM, bN, bK, tm, tn, min_blocks)``; ``None`` is
    ``pick_tile``'s, for this GEMM and for the backward's. ``stages`` is the depth of the
    shared-memory pipeline and ``splits`` the split-K factor (``None``:
    ``pick_splits``; the partial sums are reduced, and the epilogue applied, in a
    second XLA fusion). Differentiable once in everything but ``row_mask`` (a constant: its cotangent
    is zero). The forward under ``jax.grad`` also stores ``h``, or the raw
    accumulator ``x @ W`` when there is a ``row_scale`` (its cotangent needs it;
    ``h`` is then recomputed elementwise in the backward)."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation!r}")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError(f"need 2-d x and w, got {x.shape} and {w.shape}")
    M, K = x.shape
    N, Kw = w.shape if transpose_w else w.shape[::-1]
    if Kw != K:
        layout = "[N, K] (transpose_w)" if transpose_w else "[K, N]"
        raise ValueError(f"need x [M, K] and w {layout}, got {x.shape} and {w.shape}")
    if K % 32 or N % 32:
        raise ValueError(f"K and N must be multiples of 32, got K={K}, N={N}")
    if rank1 is not None and len(rank1) != 2:
        raise ValueError("rank1 must be a (row [M], col [N]) pair")
    rank1_row, rank1_col = rank1 if rank1 is not None else (None, None)
    N_out = N // 2 if activation == "glu" else N
    if row_rstd is not None:
        if row_scale is not None:
            raise ValueError("pass either row_scale or row_rstd")
        row_scale = row_rstd
    rstd = float(rstd_eps) if row_rstd is not None else None
    opts = dict(
        bias=bias,
        row_scale=row_scale,
        pre_add=pre_add,
        rank1_row=rank1_row,
        rank1_col=rank1_col,
        residual=residual,
        residual2=residual2,
        row_mask=row_mask,
    )
    shapes = dict(
        bias=(N,),
        row_scale=(M,),
        pre_add=(M, N),
        rank1_row=(M,),
        rank1_col=(N,),
        residual=(M, N_out),
        residual2=(M, N_out),
        row_mask=(M,),
    )
    w_shape = w.shape
    for name, a in dict(x=x, w=w, w_lo=w_lo, w_t_lo=w_t_lo, **opts).items():
        if a is None:
            continue
        if a.dtype != jnp.float32:
            raise TypeError(f"fused_dense runs in float32, got {name} of {a.dtype}")
        want = dict(shapes, w_lo=w_shape, w_t_lo=(N, K)).get(name)
        if want is not None and a.shape != want:
            raise ValueError(f"{name} must have shape {want}, got {a.shape}")
    if row_sumsq and row_stats:
        raise ValueError("row_stats already includes row_sumsq")
    stats = "both" if row_stats else ("sumsq" if row_sumsq else "none")
    if tile is not None:
        tile = tuple(tile)
    # the forward's tile; tile itself (None: picked per GEMM) goes to the backward
    fwd_tile = tile
    if algorithm != "simt":
        splits = None  # pick_splits_tf32's, in _call_tf32
        in_kernel = stats != "none" and M >= TF32_STATS_MIN_ROWS
        bN = (fwd_tile or pick_tile_tf32(M, N_out, in_kernel, activation == "glu"))[1]
        if in_kernel and bN != N_out:
            raise ValueError(f"row statistics need the tile's bN = {N_out}, got {bN}")
    else:
        fwd_tile, splits = _simt_tile(
            M, K, N_out, activation, transpose_w, stats, fwd_tile, stages, splits
        )

    if dact is not None:
        h, act = dact
        if act not in ("silu", "glu"):
            raise ValueError(f"dact's activation must be 'silu' or 'glu', got {act!r}")
        others = [
            n
            for n, a in opts.items()
            if a is not None and n not in ("row_scale", "row_mask")
        ]
        if activation is not None or others or save_pre:
            raise ValueError(f"dact takes only row_scale and row_mask, got {others}")
        h_shape = (M, 2 * N if act == "glu" else N)
        if h.dtype != jnp.float32 or h.shape != h_shape:
            raise ValueError(f"dact's h must be float32 {h_shape}, got {h.dtype} {h.shape}")
        opts = {n: a for n, a in opts.items() if a is not None}
        if w_t is not None or stats != "none":
            raise ValueError("dact takes neither w_t nor row statistics")
        out, *_ = _call(
            x,
            w,
            opts,
            None,
            False,
            False,
            transpose_w=transpose_w,
            tile=tile,
            stages=stages,
            splits=splits,
            dact=dact,
            rstd_eps=rstd,
            algorithm=algorithm,
            w_lo=w_lo,
        )
        return out

    for name, a in (("w_lo", w_lo), ("w_t_lo", w_t_lo)):
        if a is not None:
            opts[name] = a
    if w_t is not None:
        if transpose_w:
            raise ValueError(
                "w_t is for transpose_w=False: with transpose_w the backward's dx "
                "already reads w on the fast (N-major) path"
            )
        if w_t.dtype != jnp.float32 or w_t.shape != (N, K):
            raise ValueError(f"w_t must be float32 [N, K] = {(N, K)}, got {w_t.shape}")
        opts["w_t"] = w_t
    names = tuple(
        n for n in (*OPTIONAL, "w_t", "w_lo", "w_t_lo") if opts.get(n) is not None
    )
    fn = _fused(
        names,
        activation,
        transpose_w,
        save_pre,
        stats,
        fwd_tile,
        tile,
        stages,
        splits,
        rstd,
        algorithm,
    )
    return fn(x, w, tuple(opts[n] for n in names))


def _simt_tile(M, K, N_out, activation, transpose_w, stats, tile, stages, splits):
    """The SIMT kernel's forward tile (``None``: picked per GEMM) and split-K factor
    for ``fused_dense``'s arguments, checked against the kernel's constraints."""
    if tile is None and stats != "none":
        tile = _stats_tile(N_out, M)
    glu = activation == "glu"
    bM, bN, bK, tm, tn, _ = _full_tile(tile or pick_tile(M, N_out, K, transpose_w, glu))
    if stats != "none" and splits is None:
        splits = pick_splits(M, K, N_out, (bM, bN, bK), stages)
    # under split-K the statistics come from the XLA epilogue, else from the kernel
    if stats != "none" and splits == 1:
        Tn = bN // tn
        if bN != N_out or Tn > 32 or Tn & (Tn - 1):
            raise ValueError(
                f"row statistics need bN = {N_out} and bN / tn a power of two up to "
                f"32, got the tile {(bM, bN, bK, tm, tn)}"
            )
    if K % bK:
        raise ValueError(f"K={K} is not a multiple of the tile's bK={bK}")
    if splits is not None and (K % (splits * bK) or K // (splits * bK) < stages - 1):
        raise ValueError(
            f"K={K} must split into {splits} ranges of at least {stages - 1} k-tiles of {bK}"
        )
    _check_smem((bM, bN, bK), stages, glu, transpose_w)
    return tile, splits


def tf32_low_part(a):
    """``a - tf32(a)`` by round-to-nearest: the small part of a 3xTF32 operand."""
    return a - jax.lax.reduce_precision(a, exponent_bits=8, mantissa_bits=10)


def pick_tile_tf32(M, N, stats=False, glu=False):
    """The tensor-core kernel's ``(128, bN, 32)`` tile for an ``[M, N]`` output
    (``N`` the GEMM's width, ``F`` for glu): ``bN = N`` for row statistics (whole
    rows per CTA), ``TF32_GLU_BN`` for glu at ``M <= 4096`` (its two B tiles per
    stage leave 2 stages at bN 128 in 3xTF32, 3 at 64; the deeper pipeline pays at
    moderate M and stops paying at large M), else 64 for ``N <= 256`` at
    ``M <= 4096``, 128 beyond (bN 256 was slower everywhere), the largest of those
    that divides ``N``. Tuned on a B200 for PET-MAD-sized layers."""
    if stats:
        if N not in (32, 64, 128, 256):
            raise ValueError(f"row statistics need an output width of 32 to 256, got {N}")
        return (128, N, 32)
    bN = 64 if N <= 256 and M <= 4096 else 128
    if glu and TF32_GLU_BN and M <= 4096:
        bN = TF32_GLU_BN
    while N % bN:
        bN //= 2
    return (128, bN, 32)


def pick_splits_tf32(M, K, N, bN, bK, stages, min_ctas=128, wide_min_ctas=64):
    """The tensor-core kernel's split-K factor: for ``M <= 128`` (the per-atom GEMMs:
    one row tile, ``N / bN`` CTAs) when the grid has fewer than ``min_ctas`` CTAs,
    and above that, from ``K >= TF32_SPLIT_MIN_K``, when it has fewer than
    ``wide_min_ctas`` (the wide-K centre GEMMs on a few hundred rows gain from it;
    a grid that already fills about the SMs does not, and the token GEMMs' ``K`` of
    256 and 512 stay unsplit). Then enough splits for ``min_ctas`` CTAs, at most 8,
    each split keeping at least ``stages`` k-tiles, dividing ``K`` into whole k-tiles
    (unsplit, the per-atom GEMMs leave most SMs idle)."""
    ctas = -(-M // 128) * (N // bN)
    if M <= 128:
        if ctas >= min_ctas:
            return 1
    elif ctas >= wide_min_ctas or not (TF32_SPLIT_MIN_K and K >= TF32_SPLIT_MIN_K):
        return 1
    s = max(1, min(8, -(-min_ctas // ctas), K // (bK * stages)))
    while s > 1 and K % (s * bK):
        s -= 1
    return s


def _stages_tf32(bN, bK, three_pass, glu, limit=200 * 1024):
    """The deepest operand pipeline (at most 4) that fits ``TF32_SMEM_LIMIT`` of
    shared memory (about 110 KB leaves room for two CTAs per SM), else ``limit``."""
    from ._gemm_tf32_cute import smem_bytes

    for cap in (TF32_SMEM_LIMIT, limit):
        for s in (4, 3, 2):
            if smem_bytes(bN, bK, s, three_pass, glu) <= cap:
                return s
    raise ValueError(f"the tile (128, {bN}, {bK}) does not fit two stages in shared memory")


def _stats_tile(N, M=None, rows=None):
    """The CTA tile for row statistics of an ``[M, N]`` output: whole rows, a row's
    threads one aligned group of lanes (``N / tn`` a power of two up to 32),
    ``tm = 4``, and ``rows`` rows per CTA: 16 or 32 (``(16, 256, 8, 4, 8)`` has 128
    threads, ``(32, 256, 8, 4, 8)`` 256), by default 32 when that still gives 100
    CTAs, else 16 (``M = 2304`` gets 144 CTAs instead of 72)."""
    if N not in (32, 64, 128, 256):
        raise ValueError(f"row statistics need an output width of 32 to 256, got {N}")
    if rows is None:
        rows = 32 if M is None or -(-M // 32) >= 100 else 16
    if rows not in (16, 32):
        raise ValueError(f"row statistics tiles have 16 or 32 rows, got {rows}")
    return (rows, N, 8, 4, 8 if N == 256 else 4)


def _check_smem(tile, stages, glu, transpose_w, limit=227 * 1024):
    """Raises if the tile's shared-memory pipeline does not fit an SM."""
    bM, bN, bK = tile
    pad = 4 if transpose_w else 0
    size = 4 * stages * bK * (bM + 4 + (bN + pad) * (2 if glu else 1))
    if size > limit:
        raise ValueError(f"tile {tile} x {stages} stages needs {size} B of shared memory")


def pick_tile(M, N, K, transpose_w=False, glu=False):
    """The CTA tile for an ``[M, N]`` output (``N`` the GEMM's width, ``F`` for glu)
    over ``K``, from in-situ timings against cuBLAS on a B200, for PET-MAD-sized
    layers on systems of about 100, 1000 and 10000 atoms:

    - ``M <= 128`` (the per-atom GEMMs): 64x64x8 and split-K (``pick_splits``);
      unsplit they leave most SMs idle.
    - ``M >= 8192``, ``N <= 256``, no glu: 128x64x16 below 100000 rows, 128x128x16
      above, both with 2 CTAs per SM (the sixth entry), which brings these shapes
      to about cuBLAS or below. Not for glu or wider N: two CTAs of 128x128 with
      glu spill, qkv and comb1 gain nothing.
    - ``transpose_w`` (K-major B, the dx without ``w_t``) at ``M >= 8192``:
      128x128x16; below, 128x64x16 for ``N >= 512``, else 64x64x8.
    - otherwise the first of 128x128x16, 128x64x16, 64x64x8, 64x32x8 with at least
      100 CTAs. The wide-N GEMMs (qkv, glu in, comb1) stay slower than cuBLAS: a
      128x128 tile at 8x8 per thread leaves one CTA per SM."""
    del K  # not part of the rule yet
    if M <= 128:
        return (64, 64, 8)
    large = M >= 8192
    if transpose_w:
        if large:
            return (128, 128, 16)
        return (128, 64, 16) if N >= 512 else (64, 64, 8)
    if large and N <= 256 and not glu:
        return (128, 128, 16, 8, 8, 2) if M >= 100_000 else (128, 64, 16, 8, 4, 2)
    for bM, bN, bK in ((128, 128, 16), (128, 64, 16), (64, 64, 8)):
        if -(-M // bM) * -(-N // bN) >= 100:
            return (bM, bN, bK)
    return (64, 32, 8)


def default_micro(bM, bN):
    """The per-thread register tile ``(tm, tn)`` without an explicit one: as many
    threads as give 4 x 4 each, at most 256 and at most 16 along N (64x64 and 64x32
    are 4x4, 128x64 is 8x4, 128x128 8x8)."""
    T = min(256, bM * bN // 16)
    Tn = min(16, bN // 4)
    return bM // (T // Tn), bN // Tn


def _full_tile(tile):
    """``(bM, bN, bK, tm, tn, min_blocks)`` from a 3-, 5- or 6-tuple: the register
    tile defaults to ``default_micro``, the CTAs per SM to ask ptxas for to 0 (no
    hint; ``MIN_BLOCKS_PER_SM`` overrides it when set)."""
    if len(tile) == 3:
        return (*tile, *default_micro(*tile[:2]), 0)
    if len(tile) == 5:
        return (*tile, 0)
    if len(tile) != 6:
        raise ValueError(
            f"tile must be (bM, bN, bK), (bM, bN, bK, tm, tn) or (bM, bN, bK, tm, tn, "
            f"min_blocks), got {tile}"
        )
    return tuple(tile)


def pick_splits(M, K, N, tile=(64, 64, 8), stages=3, min_ctas=128):
    """The split-K factor for an ``[M, N]`` output over ``K``: only for ``M <= 128``
    (the per-atom GEMMs), enough splits for ``min_ctas`` CTAs, at most 8 and at
    most ``K / (bK (stages - 1))``, and dividing ``K`` into whole k-tiles."""
    bM, bN, bK = tile[:3]
    if M > 128:
        return 1
    ctas = -(-M // bM) * -(-N // bN)
    s = max(1, min(8, -(-min_ctas // ctas), K // (bK * (stages - 1))))
    while s > 1 and K % (s * bK):
        s -= 1
    return s


@functools.cache
def _fused(
    names,
    activation,
    transpose_w,
    save_pre,
    row_stats,
    tile,
    dx_tile,
    stages,
    splits,
    rstd,
    algorithm="simt",
):
    """The custom_vjp function for one set of present inputs and flags; the
    optional inputs come in as a tuple in the order of ``names``. ``tile`` is the
    forward's, ``dx_tile`` the backward dx GEMM's (``None``: ``pick_tile``);
    ``rstd`` is the RMSNorm epsilon when ``row_scale`` holds sums of squares;
    ``algorithm`` is ``fused_dense``'s, for both GEMMs."""
    has_scale = "row_scale" in names
    # what the backward reads: the accumulator if there is a row scale (for its
    # cotangent, h is recomputed from it), else h if the activation needs it, and
    # the output for the row statistics' cotangents
    save_acc = has_scale
    keep_pre = save_pre or (activation is not None and not has_scale)
    call = functools.partial(
        _call,
        transpose_w=transpose_w,
        tile=tile,
        stages=stages,
        splits=splits,
        row_stats=row_stats,
        rstd_eps=rstd,
        algorithm=algorithm,
    )

    def outputs(out, pre, rsum, rsq):
        extra = [a for a in (pre, rsum, rsq) if a is not None]
        return (out, *extra) if extra else out

    @jax.custom_vjp
    def f(x, w, arrays):
        opts = dict(zip(names, arrays))
        out, pre, _, rsum, rsq = call(
            x, w, opts, activation, save_pre, False, w_lo=opts.get("w_lo")
        )
        return outputs(out, pre, rsum, rsq)

    def fwd(x, w, arrays):
        opts = dict(zip(names, arrays))
        out, pre, acc, rsum, rsq = call(
            x, w, opts, activation, keep_pre, save_acc, w_lo=opts.get("w_lo")
        )
        for n in ("residual", "residual2"):
            opts.pop(n, None)
        kept = out if row_stats != "none" else None
        primal = outputs(out, pre if save_pre else None, rsum, rsq)
        return primal, (x, w, opts, pre, acc, kept)

    def bwd(res, cot):
        x, w, opts, pre, acc, out = res
        cot = list(cot) if isinstance(cot, tuple) else [cot]
        d_out = cot.pop(0)
        d_pre = cot.pop(0) if save_pre else None
        if row_stats == "both":
            d_out = d_out + cot.pop(0)[:, None]
        if row_stats != "none":
            d_out = d_out + 2 * out * cot.pop(0)[:, None]
        K = x.shape[1]
        sumsq = opts.get("row_scale") if rstd is not None else None
        s = _effective_scale(opts, rstd, K)
        mask = opts.get("row_mask")
        g = d_out if mask is None else d_out * mask[:, None]
        h = pre
        if h is None and activation is not None:
            h = _epilogue_pre(acc, opts if s is None else dict(opts, row_scale=s))
        dh = _activation_vjp(h, g, activation)
        if d_pre is not None:
            dh = dh + d_pre

        # dx = (dh s) @ W.T: the same kernel with the weight read the other way, or
        # the prepared W.T read the fast way; the tensor-core kernel reads either
        # way at the same speed, so it takes w (and w_lo) as they are
        w_dx, dx_transposed, dx_lo = w, not transpose_w, opts.get("w_lo")
        if "w_t" in opts and algorithm == "simt":
            w_dx, dx_transposed = opts["w_t"], False
        dx, *_ = _call(
            dh,
            w_dx,
            {} if s is None else {"row_scale": s},
            None,
            False,
            False,
            transpose_w=dx_transposed,
            tile=dx_tile if algorithm == "simt" else None,
            stages=stages,
            splits=None,
            algorithm=algorithm,
            w_lo=dx_lo,
        )
        dh_s = dh if s is None else dh * s[:, None]
        if transpose_w:
            dw = jnp.dot(dh_s.T, x, precision=HIGHEST)
        else:
            dw = jnp.dot(x.T, dh_s, precision=HIGHEST)
        grads = dict(
            bias=lambda: dh.sum(0),
            row_scale=lambda: (
                jnp.sum(dh * acc, axis=1)
                if sumsq is None
                else jnp.sum(dh * acc, axis=1) * (-0.5 / K) * s**3
            ),
            pre_add=lambda: dh,
            rank1_row=lambda: jnp.sum(dh * opts["rank1_col"], axis=1),
            rank1_col=lambda: jnp.sum(dh * opts["rank1_row"][:, None], axis=0),
            residual=lambda: g,
            residual2=lambda: g,
            row_mask=lambda: jnp.zeros_like(mask),
            w_t=lambda: jnp.zeros_like(opts["w_t"]),
            w_lo=lambda: jnp.zeros_like(opts["w_lo"]),
            w_t_lo=lambda: jnp.zeros_like(opts["w_t_lo"]),
        )
        return dx, dw, tuple(grads[n]() for n in names)

    f.defvjp(fwd, bwd)
    return f


def _effective_scale(opts, rstd, K):
    """The row factor the kernel applies: ``row_scale`` itself, or with ``rstd``
    the RMSNorm's ``rsqrt(row_scale / K + rstd)``; ``None`` without one."""
    s = opts.get("row_scale")
    if s is None or rstd is None:
        return s
    return jax.lax.rsqrt(s / K + rstd)


def _epilogue_pre(acc, opts):
    """The pre-activation ``h`` from the raw accumulator, as the kernel forms it."""
    h = acc
    if "row_scale" in opts:
        h = h * opts["row_scale"][:, None]
    if "bias" in opts:
        h = h + opts["bias"]
    if "pre_add" in opts:
        h = h + opts["pre_add"]
    if "rank1_row" in opts:
        h = h + opts["rank1_row"][:, None] * opts["rank1_col"]
    return h


def _activation_vjp(h, g, activation):
    """The cotangent of the pre-activation ``h`` given that of the activation's
    output ``g``."""
    if activation is None:
        return g
    if activation == "silu":
        sg = jax.nn.sigmoid(h)
        return g * sg * (1 + h * (1 - sg))
    v, gate = jnp.split(h, 2, axis=-1)
    sg = jax.nn.sigmoid(gate)
    return jnp.concatenate([g * sg, g * v * sg * (1 - sg)], axis=-1)


def _epilogue(acc, opts, activation, dact):
    """The kernel's epilogue in plain JAX, for split-K: ``(out, h)``."""
    if dact is not None:
        g = acc if "row_scale" not in opts else acc * opts["row_scale"][:, None]
        out = _activation_vjp(dact[0], g, dact[1])
        h = None
    else:
        h = _epilogue_pre(acc, opts)
        out = h
        if activation == "silu":
            out = jax.nn.silu(h)
        elif activation == "glu":
            v, g = jnp.split(h, 2, axis=-1)
            out = v * jax.nn.sigmoid(g)
        for n in ("residual", "residual2"):
            if n in opts:
                out = out + opts[n]
    if "row_mask" in opts:
        out = out * opts["row_mask"][:, None]
    return out, h


def _call(
    x,
    w,
    opts,
    activation,
    save_pre,
    save_acc,
    *,
    transpose_w,
    tile,
    stages,
    splits,
    dact=None,
    row_stats="none",
    rstd_eps=None,
    algorithm="simt",
    w_lo=None,
):
    """The kernel launch: ``(out, h, acc, row sum, row sum of squares)``, ``None``
    for each not asked for (``row_stats`` is ``"none"``, ``"sumsq"`` or
    ``"both"``); absent inputs and outputs are ``[1]`` placeholders.
    ``splits=None`` is ``pick_splits``'s; ``rstd_eps`` makes ``row_scale`` sums of
    squares for an RMSNorm (see ``fused_dense``'s ``row_rstd``); ``algorithm`` other
    than ``"simt"`` goes to the tensor-core kernel (``_call_tf32``), with ``w_lo``
    the weight's low part for 3xTF32."""
    if algorithm != "simt":
        if tile is not None and len(tile) != 3:
            tile = None  # a SIMT tile (row statistics' default): pick our own
        return _call_tf32(
            x,
            w,
            opts,
            activation,
            save_pre,
            save_acc,
            transpose_w=transpose_w,
            tile=tile,
            dact=dact,
            row_stats=row_stats,
            rstd_eps=rstd_eps,
            three_pass=algorithm == "tf32x3",
            w_lo=w_lo,
        )
    import cutlass.jax as cjax

    from . import _gemm_cute as kernel

    M, K = x.shape
    N = w.shape[0] if transpose_w else w.shape[1]
    N_gemm = N // 2 if activation == "glu" else N
    if tile is None:
        tile = pick_tile(M, N_gemm, K, transpose_w, activation == "glu")
    bM, bN, bK, tm, tn, min_blocks = _full_tile(tile)
    if splits is None:
        splits = pick_splits(M, K, N_gemm, tile, stages)
    N_out = 2 * N if dact is not None and dact[1] == "glu" else N_gemm
    split = splits > 1
    placeholder = jnp.zeros((1,), jnp.float32)
    inputs = [opts.get(n, placeholder) for n in OPTIONAL]
    inputs.append(placeholder if dact is None else dact[0])

    def maybe(on, shape):
        return jax.ShapeDtypeStruct(shape if on else (1,), jnp.float32)

    # the kernel takes the weight as (N, K): the [K, N] array through a transposed
    # view, or the [N, K] array as it is
    w_spec = None if transpose_w else cjax.TensorSpec(mode=(1, 0))
    # under split-K the kernel only stores the partial accumulators
    kernel_opts = {} if split else opts
    stats = "none" if split else row_stats
    out, pre, acc, rsum, rsq = cjax.cutlass_call(
        kernel.launch_dense,
        output_shape_dtype=(
            maybe(not split, (M, N_out)),
            maybe(save_pre and not split, (M, N)),
            maybe(save_acc or split, (splits, M, N)),
            maybe(stats == "both", (M,)),
            maybe(stats != "none", (M,)),
        ),
        input_spec=(None, w_spec, *(None for _ in inputs)),
        use_static_tensors=True,
        bM=bM,
        bN=bN,
        bK=bK,
        stages=stages,
        tm=tm,
        tn=tn,
        transpose_w=transpose_w,
        has_bias="bias" in kernel_opts,
        has_row_scale="row_scale" in kernel_opts,
        has_pre_add="pre_add" in kernel_opts,
        has_rank1="rank1_row" in kernel_opts,
        save_pre=save_pre and not split,
        save_acc=save_acc or split,
        activation="glu"
        if activation == "glu"
        else ("none" if split else activation or "none"),
        has_residual="residual" in kernel_opts,
        has_residual2="residual2" in kernel_opts,
        has_row_mask="row_mask" in kernel_opts,
        dact="none" if dact is None or split else dact[1],
        splits=splits,
        row_stats=stats,
        row_scale_mode="scale" if rstd_eps is None else "rstd",
        rstd_eps=0.0 if rstd_eps is None else rstd_eps,
        vector_epilogue=VECTOR_EPILOGUE,
        min_blocks=MIN_BLOCKS_PER_SM or min_blocks,
        fragment_slots=FRAGMENT_SLOTS,
    )(x, w, *inputs)
    acc = acc.sum(0) if split else acc[0]
    if split:
        if rstd_eps is not None and "row_scale" in opts:
            opts = dict(opts, row_scale=_effective_scale(opts, rstd_eps, K))
        out, pre = _epilogue(acc, opts, activation, dact)
        rsum, rsq = out.sum(1), jnp.sum(out * out, axis=1)
    return (
        out,
        pre if save_pre else None,
        acc if save_acc else None,
        rsum if row_stats == "both" else None,
        rsq if row_stats != "none" else None,
    )


def _call_tf32(
    x,
    w,
    opts,
    activation,
    save_pre,
    save_acc,
    *,
    transpose_w,
    tile,
    dact,
    row_stats,
    rstd_eps,
    three_pass,
    w_lo,
):
    """``_call`` on the tensor-core kernel: the same outputs. With 3xTF32, ``x``'s
    low part is split in the kernel (``TF32_SPLIT_A``, else here by XLA) and
    ``w``'s is ``w_lo`` (computed here when None). For ``M <= 128`` the K loop is
    split (``pick_splits_tf32``): the kernel stores raw partials and XLA reduces
    them and applies the epilogue, as ``_call`` does for the SIMT kernel; the row
    statistics come from the kernel only from ``TF32_STATS_MIN_ROWS`` rows on (the
    whole-row tiles they need leave too few CTAs below), else from XLA."""
    import cutlass.jax as cjax

    from . import _gemm_tf32_cute as kernel

    M, K = x.shape
    N = w.shape[0] if transpose_w else w.shape[1]
    glu = activation == "glu"
    N_gemm = N // 2 if glu else N
    in_kernel_stats = row_stats != "none" and M >= TF32_STATS_MIN_ROWS
    _, bN, bK = (
        tile if tile is not None else pick_tile_tf32(M, N_gemm, in_kernel_stats, glu)
    )
    stages = _stages_tf32(bN, bK, three_pass, glu)
    splits = 1
    if TF32_SPLIT_K and not in_kernel_stats:
        splits = pick_splits_tf32(M, K, N_gemm, bN, bK, stages)
    split = splits > 1
    split_a = three_pass and TF32_SPLIT_A
    N_out = 2 * N if dact is not None and dact[1] == "glu" else N_gemm
    placeholder = jnp.zeros((1,), jnp.float32)
    x_lo = tf32_low_part(x) if three_pass and not split_a else placeholder
    if three_pass:
        w_lo = tf32_low_part(w) if w_lo is None else w_lo
    else:
        w_lo = placeholder
    inputs = [opts.get(n, placeholder) for n in OPTIONAL]
    inputs.append(placeholder if dact is None or split else dact[0])

    def maybe(on, shape):
        return jax.ShapeDtypeStruct(shape if on else (1,), jnp.float32)

    # under split-K the kernel only stores the partial accumulators
    kernel_opts = {} if split else opts
    stats = row_stats if in_kernel_stats else "none"
    # the weight as the (N, K) tensor: the [K, N] array through a transposed view
    # (MN-major B), or the [N, K] array as it is (K-major)
    w_spec = None if transpose_w else cjax.TensorSpec(mode=(1, 0))
    out, pre, acc, rsum, rsq = cjax.cutlass_call(
        kernel.launch_dense_tf32,
        output_shape_dtype=(
            maybe(not split, (M, N_out)),
            maybe(save_pre and not split, (M, N)),
            maybe(save_acc or split, (splits, M, N)),
            maybe(stats == "both", (M,)),
            maybe(stats != "none", (M,)),
        ),
        input_spec=(None, w_spec, None, w_spec if three_pass else None)
        + tuple(None for _ in inputs),
        use_static_tensors=True,
        bN=bN,
        bK=bK,
        stages=stages,
        b_k_major=transpose_w,
        three_pass=three_pass,
        subtiles=max(1, bN // 64),
        has_bias="bias" in kernel_opts,
        has_row_scale="row_scale" in kernel_opts,
        row_scale_mode="scale" if rstd_eps is None else "rstd",
        rstd_eps=0.0 if rstd_eps is None else rstd_eps,
        has_pre_add="pre_add" in kernel_opts,
        has_rank1="rank1_row" in kernel_opts,
        save_pre=save_pre and not split,
        save_acc=save_acc or split,
        activation="glu" if glu else ("none" if split else activation or "none"),
        dact="none" if dact is None or split else dact[1],
        has_residual="residual" in kernel_opts,
        has_residual2="residual2" in kernel_opts,
        has_row_mask="row_mask" in kernel_opts,
        row_stats=stats,
        splits=splits,
        split_a=split_a,
    )(x, w, x_lo, w_lo, *inputs)
    acc = acc.sum(0) if split else acc[0]
    if split:
        if rstd_eps is not None and "row_scale" in opts:
            opts = dict(opts, row_scale=_effective_scale(opts, rstd_eps, K))
        out, pre = _epilogue(acc, opts, activation, dact)
    if row_stats != "none" and not in_kernel_stats:
        rsum, rsq = out.sum(1), jnp.sum(out * out, axis=1)
    return (
        out,
        pre if save_pre else None,
        acc if save_acc else None,
        rsum if row_stats == "both" else None,
        rsq if row_stats != "none" else None,
    )


def reference_dense(
    x,
    w,
    bias=None,
    *,
    row_scale=None,
    activation=None,
    pre_add=None,
    rank1=None,
    residual=None,
    residual2=None,
    row_mask=None,
    row_rstd=None,
    rstd_eps=RSTD_EPS,
    transpose_w=False,
    save_pre=False,
    row_sumsq=False,
    row_stats=False,
    w_t=None,
    dact=None,
    precision=HIGHEST,
    algorithm=None,
    w_lo=None,
    w_t_lo=None,
):
    """The same math in plain JAX, for tests and benchmarks; ``precision`` of the
    GEMM is full fp32 by default (``None`` is what a ``flax`` ``Dense`` uses).
    ``w_t``, ``algorithm``, ``w_lo`` and ``w_t_lo`` are accepted and ignored, so a
    test can pass ``fused_dense``'s arguments as they are."""
    del w_t, algorithm, w_lo, w_t_lo
    if row_rstd is not None:
        row_scale = jax.lax.rsqrt(row_rstd / x.shape[1] + rstd_eps)
    h = jnp.dot(x, w.T if transpose_w else w, precision=precision)
    if dact is not None:
        if row_scale is not None:
            h = h * row_scale[:, None]
        h = _activation_vjp(dact[0], h, dact[1])
        return h if row_mask is None else h * row_mask[:, None]
    if row_scale is not None:
        h = h * row_scale[:, None]
    if bias is not None:
        h = h + bias
    if pre_add is not None:
        h = h + pre_add
    if rank1 is not None:
        h = h + rank1[0][:, None] * rank1[1]
    pre = h
    if activation == "silu":
        h = jax.nn.silu(h)
    elif activation == "glu":
        v, g = jnp.split(h, 2, axis=-1)
        h = v * jax.nn.sigmoid(g)
    if residual is not None:
        h = h + residual
    if residual2 is not None:
        h = h + residual2
    if row_mask is not None:
        h = h * row_mask[:, None]
    extra = [pre] if save_pre else []
    if row_stats:
        extra.append(h.sum(1))
    if row_stats or row_sumsq:
        extra.append(jnp.sum(h * h, axis=1))
    return (h, *extra) if extra else h


# -- two Dense layers with one backward --

MLP_ARRAYS = (
    "x",
    "w1",
    "b1",
    "w2",
    "b2",
    "row_scale",
    "pre_add",
    "rank1_row",
    "rank1_col",
    "residual",
    "residual2",
    "row_mask",
    "w1_t",
    "w2_t",
    "w1_lo",
    "w2_lo",
    "w1_t_lo",
    "w2_t_lo",
)


def fused_mlp(
    x,
    w1,
    b1,
    w2,
    b2,
    *,
    activation,
    row_scale=None,
    pre_add=None,
    rank1=None,
    residual=None,
    residual2=None,
    row_mask=None,
    row_rstd=None,
    rstd_eps=RSTD_EPS,
    w1_t=None,
    w2_t=None,
    row_sumsq=False,
    row_stats=False,
    algorithm="simt",
    w1_lo=None,
    w2_lo=None,
    w1_t_lo=None,
    w2_t_lo=None,
):
    """``out = (act(h) @ W2 + b2 + residual + residual2) * row_mask[:, None]`` with
    ``h = (x @ W1) * row_scale[:, None] + b1 + pre_add + rank1[0][:, None] *
    rank1[1]``: two ``fused_dense`` calls in the forward and one backward for
    both, so the activation's derivative and the mask are applied in the epilogue
    of the second layer's ``dx`` GEMM (``dact``) instead of a pass of their own.
    ``activation`` is ``"silu"`` or ``"glu"``; ``x`` and ``w1`` may both be
    ``None`` when the first layer is an elementwise ``h = pre_add + b1`` (the
    compress stage of the first GNN layer, whose inputs are all folded).
    ``row_rstd`` / ``rstd_eps`` are ``fused_dense``'s (an RMSNorm of ``x`` from
    its rows' sums of squares, in place of ``row_scale``), ``w1_t`` and ``w2_t``
    the prepared transposes for the backward, as in ``fused_dense``;
    ``row_sumsq`` adds ``sum(out**2, axis=1)``, ``row_stats`` both ``sum(out,
    axis=1)`` and that. Returns ``out``, ``(out, sumsq)`` or ``(out, sum,
    sumsq)``. Differentiable once, like ``fused_dense``. ``algorithm``, ``w1_lo``
    and ``w2_lo`` are ``fused_dense``'s ``algorithm`` and ``w_lo`` for the two
    GEMMs (``w1_t_lo`` / ``w2_t_lo`` accepted, unused: the tensor-core backward
    reads the forward weights)."""
    if activation not in ("silu", "glu"):
        raise ValueError(f"activation must be 'silu' or 'glu', got {activation!r}")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")
    if (x is None) != (w1 is None):
        raise ValueError("x and w1 are both given or both None")
    if w1 is None and pre_add is None:
        raise ValueError("without a first GEMM the pre-activation comes from pre_add")
    if w1 is None and any(
        a is not None for a in (w1_t, w1_lo, w1_t_lo, row_scale, row_rstd, rank1)
    ):
        raise ValueError("without a first GEMM only pre_add and b1 form h")
    if row_rstd is not None:
        if row_scale is not None:
            raise ValueError("pass either row_scale or row_rstd")
        row_scale = row_rstd
    rstd = float(rstd_eps) if row_rstd is not None else None
    rank1_row, rank1_col = rank1 if rank1 is not None else (None, None)
    arrays = dict(
        x=x,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        row_scale=row_scale,
        pre_add=pre_add,
        rank1_row=rank1_row,
        rank1_col=rank1_col,
        residual=residual,
        residual2=residual2,
        row_mask=row_mask,
        w1_t=w1_t,
        w2_t=w2_t,
        w1_lo=w1_lo,
        w2_lo=w2_lo,
        w1_t_lo=w1_t_lo,
        w2_t_lo=w2_t_lo,
    )
    arrays = {n: a for n, a in arrays.items() if a is not None}
    F, N2 = w2.shape
    N1 = 2 * F if activation == "glu" else F
    M = (x if x is not None else pre_add).shape[0]
    K = x.shape[1] if x is not None else None
    shapes = dict(
        x=(M, K),
        w1=(K, N1),
        b1=(N1,),
        w2=(F, N2),
        b2=(N2,),
        row_scale=(M,),
        pre_add=(M, N1),
        rank1_row=(M,),
        rank1_col=(N1,),
        residual=(M, N2),
        residual2=(M, N2),
        row_mask=(M,),
        w1_t=(N1, K),
        w2_t=(N2, F),
        w1_lo=(K, N1),
        w2_lo=(F, N2),
        w1_t_lo=(N1, K),
        w2_t_lo=(N2, F),
    )
    for name, a in arrays.items():
        if a.dtype != jnp.float32:
            raise TypeError(f"fused_mlp runs in float32, got {name} of {a.dtype}")
        if a.shape != shapes[name]:
            raise ValueError(f"{name} must have shape {shapes[name]}, got {a.shape}")
    if row_sumsq and row_stats:
        raise ValueError("row_stats already includes row_sumsq")
    stats = "both" if row_stats else ("sumsq" if row_sumsq else "none")
    names = tuple(n for n in MLP_ARRAYS if n in arrays)
    return _fused_mlp(names, activation, stats, rstd, algorithm)(arrays)


def _mlp_forward(a, activation, stats, rstd, keep, algorithm="simt"):
    """The two GEMMs: ``(out, h, act(h), acc1, sum, sumsq)``; ``acc1`` (the
    first layer's raw accumulator, for the row scale's cotangent) only with
    ``keep`` and a row scale, the row statistics as ``stats`` says (``"none"``,
    ``"sumsq"`` or ``"both"``); ``rstd`` is the RMSNorm epsilon when
    ``row_scale`` holds sums of squares."""
    first = {n: a[n] for n in ("row_scale", "pre_add", "rank1_row", "rank1_col") if n in a}
    if "b1" in a:
        first["bias"] = a["b1"]
    if "x" in a:
        h_act, h, acc1, _, _ = _call(
            a["x"],
            a["w1"],
            first,
            activation,
            True,
            keep and "row_scale" in a,
            transpose_w=False,
            tile=None,
            stages=3,
            splits=None,
            rstd_eps=rstd,
            algorithm=algorithm,
            w_lo=a.get("w1_lo"),
        )
    else:
        h = a["pre_add"] + a["b1"] if "b1" in a else a["pre_add"]
        h_act = jax.nn.silu(h) if activation == "silu" else _epilogue(h, {}, "glu", None)[0]
        acc1 = None
    second = {n: a[n] for n in ("residual", "residual2", "row_mask") if n in a}
    if "b2" in a:
        second["bias"] = a["b2"]
    M, N2 = h_act.shape[0], a["w2"].shape[1]
    out, _, _, rsum, sumsq = _call(
        h_act,
        a["w2"],
        second,
        None,
        False,
        False,
        transpose_w=False,
        tile=_stats_tile(N2, M) if stats != "none" and algorithm == "simt" else None,
        stages=3,
        splits=None,
        row_stats=stats,
        algorithm=algorithm,
        w_lo=a.get("w2_lo"),
    )
    return out, h, h_act, acc1, rsum, sumsq


@functools.cache
def _fused_mlp(names, activation, stats, rstd, algorithm="simt"):
    """The custom_vjp function of ``fused_mlp`` for one set of present arrays
    (a dict keyed by ``names``)."""
    simt = algorithm == "simt"

    def outputs(out, rsum, sumsq):
        extra = [a for a in (rsum, sumsq) if a is not None]
        return (out, *extra) if extra else out

    @jax.custom_vjp
    def f(a):
        out, _, _, _, rsum, sumsq = _mlp_forward(
            a, activation, stats, rstd, False, algorithm
        )
        return outputs(out, rsum, sumsq)

    def fwd(a):
        out, h, h_act, acc1, rsum, sumsq = _mlp_forward(
            a, activation, stats, rstd, True, algorithm
        )
        saved = {n: a[n] for n in names if n not in ("residual", "residual2")}
        return outputs(out, rsum, sumsq), (saved, h, h_act, acc1, out)

    def bwd(res, cot):
        a, h, h_act, acc1, out = res
        cot = list(cot) if isinstance(cot, tuple) else [cot]
        d_out = cot.pop(0)
        if stats == "both":
            d_out = d_out + cot.pop(0)[:, None]
        if stats != "none":
            d_out = d_out + 2 * out * cot.pop(0)[:, None]
        mask = a.get("row_mask")
        g = d_out if mask is None else d_out * mask[:, None]
        # d act(h) = (d_out mask) @ W2.T, then the activation's derivative, in one
        # kernel: W2 read N-major through the prepared transpose, else K-major
        # (the tensor-core kernel reads W2 K-major at full speed, with its low part)
        if "w2_t" in a and simt:
            w2_dx, transposed = a["w2_t"], False
        else:
            w2_dx, transposed = a["w2"], True
        dh, *_ = _call(
            d_out,
            w2_dx,
            {} if mask is None else {"row_mask": mask},
            None,
            False,
            False,
            transpose_w=transposed,
            tile=None,
            stages=3,
            splits=None,
            dact=(h, activation),
            algorithm=algorithm,
            w_lo=a.get("w2_lo"),
        )
        grads = dict(
            w2=lambda: jnp.dot(h_act.T, g, precision=HIGHEST),
            b2=lambda: g.sum(0),
            residual=lambda: g,
            residual2=lambda: g,
            row_mask=lambda: jnp.zeros_like(mask),
            w1_t=lambda: jnp.zeros_like(a["w1_t"]),
            w2_t=lambda: jnp.zeros_like(a["w2_t"]),
            w1_lo=lambda: jnp.zeros_like(a["w1_lo"]),
            w2_lo=lambda: jnp.zeros_like(a["w2_lo"]),
            w1_t_lo=lambda: jnp.zeros_like(a["w1_t_lo"]),
            w2_t_lo=lambda: jnp.zeros_like(a["w2_t_lo"]),
            b1=lambda: dh.sum(0),
            pre_add=lambda: dh,
            rank1_row=lambda: jnp.sum(dh * a["rank1_col"], axis=1),
            rank1_col=lambda: jnp.sum(dh * a["rank1_row"][:, None], axis=0),
        )
        if "x" in a:
            K = a["x"].shape[1]
            s = _effective_scale(a, rstd, K)
            if "w1_t" in a and simt:
                w1_dx, transposed = a["w1_t"], False
            else:
                w1_dx, transposed = a["w1"], True
            dx, *_ = _call(
                dh,
                w1_dx,
                {} if s is None else {"row_scale": s},
                None,
                False,
                False,
                transpose_w=transposed,
                tile=None,
                stages=3,
                splits=None,
                algorithm=algorithm,
                w_lo=a.get("w1_lo"),
            )
            dh_s = dh if s is None else dh * s[:, None]
            grads["x"] = lambda: dx
            grads["w1"] = lambda: jnp.dot(a["x"].T, dh_s, precision=HIGHEST)
            if rstd is None:
                grads["row_scale"] = lambda: jnp.sum(dh * acc1, axis=1)
            else:
                # d sumsq from d s, s = rsqrt(sumsq / K + eps)
                grads["row_scale"] = lambda: jnp.sum(dh * acc1, axis=1) * (-0.5 / K) * s**3
        return ({n: grads[n]() for n in names},)

    f.defvjp(fwd, bwd)
    return f


def reference_mlp(
    x,
    w1,
    b1,
    w2,
    b2,
    *,
    activation,
    row_scale=None,
    pre_add=None,
    rank1=None,
    residual=None,
    residual2=None,
    row_mask=None,
    row_rstd=None,
    rstd_eps=RSTD_EPS,
    w1_t=None,
    w2_t=None,
    row_sumsq=False,
    row_stats=False,
    precision=HIGHEST,
    algorithm=None,
    w1_lo=None,
    w2_lo=None,
    w1_t_lo=None,
    w2_t_lo=None,
):
    """``fused_mlp`` in plain JAX (the prepared weights and ``algorithm`` are
    accepted and ignored)."""
    del w1_t, w2_t, algorithm, w1_lo, w2_lo, w1_t_lo, w2_t_lo
    if row_rstd is not None:
        row_scale = jax.lax.rsqrt(row_rstd / x.shape[1] + rstd_eps)
    if x is None:
        h = pre_add if b1 is None else pre_add + b1
        h_act = jax.nn.silu(h) if activation == "silu" else _epilogue(h, {}, "glu", None)[0]
    else:
        h_act = reference_dense(
            x,
            w1,
            b1,
            row_scale=row_scale,
            pre_add=pre_add,
            rank1=rank1,
            activation=activation,
            precision=precision,
        )
    return reference_dense(
        h_act,
        w2,
        b2,
        residual=residual,
        residual2=residual2,
        row_mask=row_mask,
        row_sumsq=row_sumsq,
        row_stats=row_stats,
        precision=precision,
    )
