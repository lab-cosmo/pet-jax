"""The CuTe DSL kernels behind ``kernels.attention``; the semantics and the
layout are documented there. Imported lazily, so ``cutlass`` is needed only
when a kernel runs.

The kernels gather and scatter themselves: tokens live in a flat list
``tok [NP, C]`` and a tile map ``rows [T, W]`` names the token behind each
tile slot (an entry ``>= NP`` is an empty slot, which reads as zeros and is
never written). One CTA per (tile ``t``, head ``h``), everything in fp32
FFMA, the tile's rows of ``q``, ``k``, ``v`` (and ``dout``) staged in shared
memory with coalesced 16-byte copies.

Forward: a lane per query, ``ceil(W / 32)`` warps. Every lane walks the keys
with the key's row coming in as a warp-wide broadcast of 128-bit loads, so the
scores, the softmax and the contraction with ``v`` are lane-local arithmetic
with no warp reduction. The row max and row sum are saved for the backward
(kept apart: on a fully masked row the max is ``finfo.min`` and a log-sum-exp
would absorb the sum).

Backward: ``WARPS_PER_CTA`` warps that split the queries; lane ``j`` holds
``k_j`` in registers (two keys per lane for ``W > 32``), per query the warp
broadcasts ``q_i`` and ``dout_i``, recomputes ``p``, forms
``dp = dout_i . v_j``, ``delta_i`` by a warp sum and ``ds = p (dp - delta)``
with ``ds = 0`` on masked keys (no gradient through the mask's constant
branch); ``p`` and ``ds`` go to shared memory in both orientations, and the
three contractions for ``dq``, ``dk``, ``dv`` run with a lane per feature.
(A lane-per-query backward like the forward spills at head dim 32.)
"""

import numpy as np

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr, range_constexpr

ADD = cute.ReductionOp.ADD
BIG_NEG = float(np.finfo(np.float32).min)
WARP = 32
WARPS_PER_CTA = 4  # the backward's query-split warps


def warps_for(W):
    """The forward's warps per CTA: a lane per query."""
    return (W + WARP - 1) // WARP


def _warp_max(x):
    return cute.arch.warp_reduction(x, cute.arch.fmax)


def _warp_sum(x):
    return cute.arch.warp_reduction(x, _add)


def _add(a, b):
    return a + b


def _round4(n):
    return (n + 3) // 4 * 4


def _dot(a, b):
    """Dot product of two register tensors of the same length: a vector multiply
    and a tree reduction."""
    return (a.load() * b.load()).reduce(ADD, 0.0, 0)


def _as_chunks(tensor, shape, stride):
    """The tensor's memory as ``shape + (4,)`` with the last mode contiguous: rows of
    16-byte chunks, so a ``[..., None]`` slice is a four-float vector copy."""
    return cute.make_tensor(
        tensor.iterator, cute.make_layout(shape + (4,), stride=stride + (1,))
    )


@cute.jit
def _stage_rows(
    dst: cute.Tensor,
    tok: cute.Tensor,
    rows: cute.Tensor,
    t: cutlass.Int32,
    col0: cutlass.Int32,
    W: cutlass.Constexpr[int],
    D: cutlass.Constexpr[int],
    NP: cutlass.Constexpr[int],
    tidx: cutlass.Int32,
    nthreads: cutlass.Constexpr[int],
    scalar_store: cutlass.Constexpr[bool],
):
    """``dst[w, e] = tok[rows[t, w], col0 + e]`` (zeros for an empty slot) for the
    ``W x D`` block, cooperatively, in 16-byte chunks; the loop is unrolled so a
    thread's loads are all in flight at once. ``scalar_store`` for a destination
    whose rows are not 16-byte aligned (padded). A ``cute.jit`` function: a staged
    loop is only preprocessed inside one."""
    NPs, C = tok.shape
    tok4 = _as_chunks(tok, (NPs, C // 4), (C, 4))
    if const_expr(not scalar_store):
        dst4 = _as_chunks(dst, (W, D // 4), (D, 4))
    chunks = W * (D // 4)
    per_row = D // 4
    r = cute.make_rmem_tensor(4, Float32)
    for it in range_constexpr((chunks + nthreads - 1) // nthreads):
        c = tidx + it * nthreads
        if c < chunks:
            w = c // per_row
            cc = c % per_row
            row = rows[t, w]
            for k in range_constexpr(4):
                r[k] = Float32(0.0)
            if row < NP:
                cute.autovec_copy(tok4[row, col0 // 4 + cc, None], r)
            if const_expr(scalar_store):
                for k in range_constexpr(4):
                    dst[w, 4 * cc + k] = r[k]
            else:
                cute.autovec_copy(r, dst4[w, cc, None])


@cute.jit
def _load_row(
    dst: cute.Tensor,
    tok: cute.Tensor,
    row: cutlass.Int32,
    col0: cutlass.Int32,
    D: cutlass.Constexpr[int],
    NP: cutlass.Constexpr[int],
):
    """``dst[e] = tok[row, col0 + e]`` into a register tensor of ``D`` as 16-byte
    loads; zeros for an empty slot."""
    NPs, C = tok.shape
    tok4 = _as_chunks(tok, (NPs, C // 4), (C, 4))
    dst4 = _as_chunks(dst, (D // 4,), (4,))
    for e in range_constexpr(D):
        dst[e] = Float32(0.0)
    if row < NP:
        for c in range_constexpr(D // 4):
            cute.autovec_copy(tok4[row, col0 // 4 + c, None], dst4[c, None])


@cute.jit
def _store_row(
    tok: cute.Tensor,
    src: cute.Tensor,
    row: cutlass.Int32,
    col0: cutlass.Int32,
    D: cutlass.Constexpr[int],
):
    """``tok[row, col0 + e] = src[e]`` from a register tensor of ``D``, as 16-byte stores."""
    NPs, C = tok.shape
    tok4 = _as_chunks(tok, (NPs, C // 4), (C, 4))
    src4 = _as_chunks(src, (D // 4,), (4,))
    for c in range_constexpr(D // 4):
        cute.autovec_copy(src4[c, None], tok4[row, col0 // 4 + c, None])


# -- forward --


@cute.kernel
def forward_kernel(
    qkv: cute.Tensor,
    rows: cute.Tensor,
    bias: cute.Tensor,
    maskf: cute.Tensor,
    out: cute.Tensor,
    rowmax: cute.Tensor,
    rowsum: cute.Tensor,
    W: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    D: cutlass.Constexpr[int],
    NP: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
):
    F = H * D
    NW = warps_for(W)
    tidx, _, _ = cute.arch.thread_idx()
    t, h, _ = cute.arch.block_idx()
    i = tidx  # this lane's query; lanes past W only help with the staging
    ic = cutlass.min(i, W - 1)

    smem = cutlass.memory.SmemAllocator()
    sK = smem.allocate_tensor(
        Float32, cute.make_layout((W, D), stride=(D, 1)), byte_alignment=16
    )
    sV = smem.allocate_tensor(
        Float32, cute.make_layout((W, D), stride=(D, 1)), byte_alignment=16
    )
    _stage_rows(sK, qkv, rows, t, F + h * D, W, D, NP, tidx, NW * WARP, False)
    _stage_rows(sV, qkv, rows, t, 2 * F + h * D, W, D, NP, tidx, NW * WARP, False)
    row_i = rows[t, ic]
    rQ = cute.make_rmem_tensor(D, Float32)
    _load_row(rQ, qkv, row_i, h * D, D, NP)
    cute.arch.sync_threads()

    # scores against every key; the key's row is one broadcast per warp
    rK = cute.make_rmem_tensor(D, Float32)
    s = []
    for j in range_constexpr(W):
        cute.autovec_copy(sK[j, None], rK)
        sj = _dot(rQ, rK) * scale + bias[t, j]
        if maskf[t, j] < 0.5:
            sj = Float32(BIG_NEG)
        s.append(sj)
    # softmax, as flax: masked keys keep exp(min - m), 0 next to a live key, uniform on a dead row
    m = s[0]
    for j in range_constexpr(1, W):
        m = cute.arch.fmax(m, s[j])
    l = Float32(0.0)
    for j in range_constexpr(W):
        s[j] = cute.math.exp(s[j] - m)
        l = l + s[j]
    inv_l = Float32(1.0) / l

    # out_i = sum_j p_ij v_j, the value row again a broadcast
    rO = cute.make_rmem_tensor(D, Float32)
    for e in range_constexpr(D):
        rO[e] = Float32(0.0)
    rV = cute.make_rmem_tensor(D, Float32)
    for j in range_constexpr(W):
        cute.autovec_copy(sV[j, None], rV)
        pj = s[j] * inv_l
        for e in range_constexpr(D):
            rO[e] = rO[e] + pj * rV[e]
    if i < W:
        rowmax[t, h, i] = m
        rowsum[t, h, i] = l
        if row_i < NP:
            _store_row(out, rO, row_i, h * D, D)


@cute.jit
def launch_forward(
    stream: cuda.CUstream,
    qkv: cute.Tensor,
    rows: cute.Tensor,
    bias: cute.Tensor,
    maskf: cute.Tensor,
    out: cute.Tensor,
    rowmax: cute.Tensor,
    rowsum: cute.Tensor,
    *,
    W: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    D: cutlass.Constexpr[int],
    NP: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
):
    # ``out`` is the zero-initialised input that cutlass_call aliases to the output
    forward_kernel(qkv, rows, bias, maskf, out, rowmax, rowsum, W, H, D, NP, scale).launch(
        grid=[rows.shape[0], H, 1], block=[warps_for(W) * WARP, 1, 1], stream=stream
    )


# -- backward --


@cute.kernel
def backward_kernel(
    qkv: cute.Tensor,
    rows: cute.Tensor,
    bias: cute.Tensor,
    maskf: cute.Tensor,
    rowmax: cute.Tensor,
    rowsum: cute.Tensor,
    dout: cute.Tensor,
    dqkv: cute.Tensor,
    dbias_h: cute.Tensor,
    W: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    D: cutlass.Constexpr[int],
    NP: cutlass.Constexpr[int],
    NW: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
):
    F = H * D
    WP = _round4(W)
    tidx, _, _ = cute.arch.thread_idx()
    t, h, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()

    smem = cutlass.memory.SmemAllocator()
    sQ = smem.allocate_tensor(
        Float32, cute.make_layout((W, D), stride=(D, 1)), byte_alignment=16
    )
    sK = smem.allocate_tensor(
        Float32, cute.make_layout((W, D), stride=(D, 1)), byte_alignment=16
    )
    sV = smem.allocate_tensor(
        Float32, cute.make_layout((W, D), stride=(D + 1, 1)), byte_alignment=16
    )
    sdO = smem.allocate_tensor(
        Float32, cute.make_layout((W, D), stride=(D, 1)), byte_alignment=16
    )
    sP = smem.allocate_tensor(
        Float32, cute.make_layout((W, WP), stride=(WP, 1)), byte_alignment=16
    )
    sPT = smem.allocate_tensor(
        Float32, cute.make_layout((W, WP), stride=(WP, 1)), byte_alignment=16
    )
    sdS = smem.allocate_tensor(
        Float32, cute.make_layout((W, WP), stride=(WP, 1)), byte_alignment=16
    )
    sdST = smem.allocate_tensor(
        Float32, cute.make_layout((W, WP), stride=(WP, 1)), byte_alignment=16
    )

    _stage_rows(sQ, qkv, rows, t, h * D, W, D, NP, tidx, NW * WARP, False)
    _stage_rows(sK, qkv, rows, t, F + h * D, W, D, NP, tidx, NW * WARP, False)
    _stage_rows(sV, qkv, rows, t, 2 * F + h * D, W, D, NP, tidx, NW * WARP, True)
    _stage_rows(sdO, dout, rows, t, h * D, W, D, NP, tidx, NW * WARP, False)
    # the transposed buffers' padding columns are never written below
    for idx in cutlass.range(tidx, W * WP, NW * WARP):
        sPT[idx // WP, idx % WP] = Float32(0.0)
        sdST[idx // WP, idx % WP] = Float32(0.0)
    cute.arch.sync_threads()

    j0 = cutlass.min(lane, W - 1)
    rK0 = cute.make_rmem_tensor(D, Float32)
    cute.autovec_copy(sK[j0, None], rK0)
    v0 = maskf[t, j0]
    b0 = bias[t, j0]
    if lane >= W:
        v0 = Float32(0.0)
    if const_expr(W > WARP):
        j1 = cutlass.min(lane + WARP, W - 1)
        rK1 = cute.make_rmem_tensor(D, Float32)
        cute.autovec_copy(sK[j1, None], rK1)
        v1 = maskf[t, j1]
        b1 = bias[t, j1]
        if lane + WARP >= W:
            v1 = Float32(0.0)

    # phase A, a lane per key: p and ds of this warp's queries, into both orientations
    db0 = Float32(0.0)
    db1 = Float32(0.0)
    rQ = cute.make_rmem_tensor(D, Float32)
    rdO = cute.make_rmem_tensor(D, Float32)
    for i in cutlass.range(warp, W, NW):
        cute.autovec_copy(sQ[i, None], rQ)
        cute.autovec_copy(sdO[i, None], rdO)
        m_i = rowmax[t, h, i]
        l_i = rowsum[t, h, i]
        s0 = _dot(rQ, rK0) * scale + b0
        dp0 = Float32(0.0)
        for e in range_constexpr(D):
            dp0 = dp0 + rdO[e] * sV[j0, e]
        if v0 < 0.5:
            s0 = Float32(BIG_NEG)
        p0 = cute.math.exp(s0 - m_i) / l_i
        if lane >= W:
            p0 = Float32(0.0)
        p1 = Float32(0.0)
        dp1 = Float32(0.0)
        if const_expr(W > WARP):
            s1 = _dot(rQ, rK1) * scale + b1
            for e in range_constexpr(D):
                dp1 = dp1 + rdO[e] * sV[j1, e]
            if v1 < 0.5:
                s1 = Float32(BIG_NEG)
            p1 = cute.math.exp(s1 - m_i) / l_i
            if lane + WARP >= W:
                p1 = Float32(0.0)
        delta = _warp_sum(p0 * dp0 + p1 * dp1)
        ds0 = p0 * (dp0 - delta)
        if v0 < 0.5:
            ds0 = Float32(0.0)
        db0 = db0 + ds0
        if lane < WP:
            sP[i, lane] = p0
            sdS[i, lane] = ds0
        if lane < W:
            sPT[lane, i] = p0
            sdST[lane, i] = ds0
        if const_expr(W > WARP):
            ds1 = p1 * (dp1 - delta)
            if v1 < 0.5:
                ds1 = Float32(0.0)
            db1 = db1 + ds1
            if lane + WARP < WP:
                sP[i, lane + WARP] = p1
                sdS[i, lane + WARP] = ds1
            if lane + WARP < W:
                sPT[lane + WARP, i] = p1
                sdST[lane + WARP, i] = ds1
    # dbias over this warp's queries; the warps' partial sums are added by the host
    if lane < W:
        dbias_h[t, h, warp, lane] = db0
    if const_expr(W > WARP):
        if lane + WARP < W:
            dbias_h[t, h, warp, lane + WARP] = db1
    cute.arch.sync_threads()

    # phase B, a lane per feature: dq from ds rows and k columns, dk from ds^T rows and q
    # columns, dv from p^T rows and dout columns; rows of the flat list written in place
    if lane < D:
        rRow = cute.make_rmem_tensor(WP, Float32)
        rCol = cute.make_rmem_tensor(WP, Float32)
        for jj in range_constexpr(WP):
            if const_expr(jj < W):
                rCol[jj] = sK[jj, lane]
            else:
                rCol[jj] = Float32(0.0)
        for i in cutlass.range(warp, W, NW):
            cute.autovec_copy(sdS[i, None], rRow)
            row = rows[t, i]
            if row < NP:
                dqkv[row, h * D + lane] = _dot(rRow, rCol) * scale
        for ii in range_constexpr(W):
            rCol[ii] = sQ[ii, lane]
        for j in cutlass.range(warp, W, NW):
            cute.autovec_copy(sdST[j, None], rRow)
            row = rows[t, j]
            if row < NP:
                dqkv[row, F + h * D + lane] = _dot(rRow, rCol) * scale
        for ii in range_constexpr(W):
            rCol[ii] = sdO[ii, lane]
        for j in cutlass.range(warp, W, NW):
            cute.autovec_copy(sPT[j, None], rRow)
            row = rows[t, j]
            if row < NP:
                dqkv[row, 2 * F + h * D + lane] = _dot(rRow, rCol)


@cute.jit
def launch_backward(
    stream: cuda.CUstream,
    qkv: cute.Tensor,
    rows: cute.Tensor,
    bias: cute.Tensor,
    maskf: cute.Tensor,
    rowmax: cute.Tensor,
    rowsum: cute.Tensor,
    dout: cute.Tensor,
    dqkv: cute.Tensor,
    dbias_h: cute.Tensor,
    *,
    W: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    D: cutlass.Constexpr[int],
    NP: cutlass.Constexpr[int],
    NW: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
):
    # ``dqkv`` is the zero-initialised input that cutlass_call aliases to the output
    backward_kernel(
        qkv, rows, bias, maskf, rowmax, rowsum, dout, dqkv, dbias_h, W, H, D, NP, NW, scale
    ).launch(grid=[rows.shape[0], H, 1], block=[NW * WARP, 1, 1], stream=stream)
