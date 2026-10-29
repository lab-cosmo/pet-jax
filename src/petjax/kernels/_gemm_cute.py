# Derived from NVIDIA CUTLASS, examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/sgemm.py:
#
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Modifications Copyright (c) 2026 Marcel Langer, under the MIT License (LICENSE).

"""The CuTe DSL kernel behind ``kernels.dense``: an fp32 SIMT GEMM with a fused
epilogue, adapted from CUTLASS's Ampere ``sgemm.py`` example. Imported lazily, so
``cutlass`` is needed only when a kernel runs.

``out[M, N_out] = epilogue(x[M, K] @ W[K, N])`` with FFMA accumulation (no tensor
cores). Operands as JAX holds them: ``x`` row-major (K contiguous), ``out``
row-major, and the weight in one of two storages. The kernel always sees it as a
``(N, K)`` tensor (sgemm's B):

- ``w [K, N]`` row-major (the ``Dense`` kernel ``[in, out]``), through a view of
  strides ``(1, N)`` (``TensorSpec`` mode ``(1, 0)``, no copy): "N-major B", whose
  tiles load with vectorized ``cp.async`` into an unpadded N-major shared tile;
- ``wt [N, K]`` row-major (``transpose_w``, the weight stored transposed, so a
  backward ``dh @ W.T`` reads the forward ``W`` as is), strides ``(K, 1)``:
  "K-major B", which loads element by element into an N-major shared tile padded
  by 4 against bank conflicts, exactly as ``x`` (K-major A) does.

Epilogue, in this order, each step switched by a compile-time flag:
``acc *= row_scale[m]`` (with ``row_scale_mode="rstd"``: ``row_scale`` holds a
row's sum of squares over the normed width ``K``, and the factor is
``rsqrt(row_scale[m] / K + rstd_eps)``, an RMSNorm folded into the GEMM),
``acc += bias[n]``, ``acc += pre_add[m, n]``,
``acc += rank1_row[m] rank1_col[n]``, an optional store of this pre-activation
``h`` (``save_pre``; ``save_acc`` stores the raw accumulator instead or as well),
the activation (``none``, ``silu`` ``x sigmoid(x)``, or ``glu``),
``acc += residual[m, n]``, ``acc += residual2[m, n]``, ``acc *= row_mask[m]``.
For ``glu`` the CTA holds two accumulators over the rows ``n`` and ``F + n`` of
the ``(N, K)`` weight (``F = N / 2``, the value and gate halves, i.e. column
blocks of ``w`` or row blocks of ``wt``; two B tiles in shared memory, one A
tile) and writes ``v sigmoid(g)``, ``N_out = F``. The column inputs (``bias``,
``rank1_col``), ``pre_add`` and the saved ``[M, N]`` outputs span the full
accumulator width ``N``; the residuals are ``[M, N_out]``. A switched off input
or output is a ``[1]`` placeholder and is never touched. The saved accumulator
is ``[splits, M, N]`` (``splits = 1`` without split-K).

``dact`` (``silu`` or ``glu``, with ``activation`` none and only ``row_scale``
and ``row_mask`` besides) is the activation's derivative applied to a backward
GEMM: ``acc * act'(dact_h)`` with ``dact_h`` the saved pre-activation of the
layer whose cotangent this is; for ``glu`` the GEMM is ``[M, F]``, ``dact_h``
``[M, 2F] = [v | g]`` and the output ``[M, 2F] = [acc sigmoid(g) |
acc v sigmoid(g) (1 - sigmoid(g))]``.

``row_stats`` (``sumsq`` or ``both``) also stores each output row's sum of squares
(and sum) of the final values, for the next RMSNorm: the CTA must own whole rows
(``bN = N_out``, one column block) and a row's ``Tn = bN / tn`` threads must be
consecutive lanes of one warp (``Tn`` a power of two up to 32; the thread grid is
N-fastest), reduced by butterfly shuffles.

Split-K (``splits > 1``): the grid's third dimension splits K into contiguous
ranges, and each CTA stores only its raw partial accumulators into ``acc``
(``[splits, M, N]``); the caller reduces them and applies the epilogue.

Constraints: shapes static per compile, ``K`` a multiple of ``splits bK`` with at
least ``stages - 1`` k-tiles per split, ``N`` (``F`` for ``glu``) a multiple of
4; ``M`` and the N edge are predicated. Shared memory is
``4 stages bK (bM + 4 + (bN + pad) (2 for glu))`` bytes, ``pad`` 4 for
``transpose_w``.

The CTA tile ``(bM, bN, bK)`` and the per-thread register tile ``(tm, tn)`` (4 or
8 each) are compile-time: ``(bM / tm) (bN / tn)`` threads, a multiple of 32, each
holding ``tm x tn`` accumulators as 4 x 4 blocks (``dense.default_micro`` is
the choice without an explicit one).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr, range_constexpr
from cutlass.tensor_utils import LayoutEnum


@cute.kernel
def dense_kernel(
    mA: cute.Tensor,
    mW: cute.Tensor,
    bias: cute.Tensor,
    row_scale: cute.Tensor,
    pre_add: cute.Tensor,
    rank1_row: cute.Tensor,
    rank1_col: cute.Tensor,
    residual: cute.Tensor,
    residual2: cute.Tensor,
    row_mask: cute.Tensor,
    dact_h: cute.Tensor,
    mC: cute.Tensor,
    mPre: cute.Tensor,
    mAcc: cute.Tensor,
    mSum: cute.Tensor,
    mSumsq: cute.Tensor,
    sA_layout: cute.Layout,
    sB_layout: cute.Layout,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    bM: cutlass.Constexpr[int],
    bN: cutlass.Constexpr[int],
    bK: cutlass.Constexpr[int],
    has_bias: cutlass.Constexpr[bool],
    has_row_scale: cutlass.Constexpr[bool],
    has_pre_add: cutlass.Constexpr[bool],
    has_rank1: cutlass.Constexpr[bool],
    save_pre: cutlass.Constexpr[bool],
    save_acc: cutlass.Constexpr[bool],
    activation: cutlass.Constexpr[str],
    has_residual: cutlass.Constexpr[bool],
    has_residual2: cutlass.Constexpr[bool],
    has_row_mask: cutlass.Constexpr[bool],
    dact: cutlass.Constexpr[str],
    splits: cutlass.Constexpr[int],
    row_stats: cutlass.Constexpr[str],
    row_threads: cutlass.Constexpr[int],
    row_scale_mode: cutlass.Constexpr[str],
    rstd_inv_k: cutlass.Constexpr[float],
    rstd_eps: cutlass.Constexpr[float],
    vector_epilogue: cutlass.Constexpr[bool],
    fragment_slots: cutlass.Constexpr[int],
):
    """One ``(bM, bN)`` block of the GEMM over the k-range of split ``z``: the
    sgemm mainloop (a ``stages``-deep ``cp.async`` pipeline over k-tiles, a
    register double buffer over k-blocks), then the epilogue, or with
    ``splits > 1`` only the store of the raw partial accumulator. The body does
    not depend on the weight's storage: that is all in the shared layout and copy
    of B that ``launch_dense`` builds."""
    glu = activation == "glu"
    M, K = mA.shape
    # the GEMM's output width (F for glu)
    N_out = mW.shape[0]
    if const_expr(glu):
        N_out = N_out // 2
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    tiler = (bM, bN, bK)
    coord = (bidx, bidy, None)
    thr_mma = tiled_mma.get_slice(tidx)

    # the value and gate halves of the weight for glu, each (F, K)
    mB = mW
    if const_expr(glu):
        mB = cute.local_tile(mW, (N_out, K), (0, 0))
        mBg = cute.local_tile(mW, (N_out, K), (1, 0))

    # split-K: this CTA's contiguous K / splits columns of A and B
    if const_expr(splits > 1):
        Ks = K // splits
        mA = cute.local_tile(mA, (M, Ks), (0, bidz))
        mB = cute.local_tile(mB, (N_out, Ks), (0, bidz))
        if const_expr(glu):
            mBg = cute.local_tile(mBg, (N_out, Ks), (0, bidz))

    # gA: (bM, bK, k), gB: (bN, bK, k)
    gA = cute.local_tile(mA, tiler=tiler, coord=coord, proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=tiler, coord=coord, proj=(None, 1, 1))

    smem = cutlass.memory.SmemAllocator()
    sA = smem.allocate_tensor(Float32, sA_layout, byte_alignment=16)
    sB = smem.allocate_tensor(Float32, sB_layout, byte_alignment=16)
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)
    if const_expr(glu):
        gBg = cute.local_tile(mBg, tiler=tiler, coord=coord, proj=(None, 1, 1))
        sBg = smem.allocate_tensor(Float32, sB_layout, byte_alignment=16)
        tBgBg = thr_copy_B.partition_S(gBg)
        tBsBg = thr_copy_B.partition_D(sBg)

    # predicates on the M edge of A and the N edge of B; K needs none (K % bK == 0)
    cA = cute.local_tile(
        cute.make_identity_tensor(mA.shape), tiler=tiler, coord=coord, proj=(1, None, 1)
    )
    cB = cute.local_tile(
        cute.make_identity_tensor(mB.shape), tiler=tiler, coord=coord, proj=(None, 1, 1)
    )
    tAcA = thr_copy_A.partition_S(cA)
    tBcB = thr_copy_B.partition_S(cB)
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
            stride=(cute.size(tAsA, mode=[1]), 1, 0),
        ),
        cutlass.Boolean,
    )
    tBpB = cute.make_rmem_tensor(
        cute.make_layout(
            (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
            stride=(cute.size(tBsB, mode=[1]), 1, 0),
        ),
        cutlass.Boolean,
    )
    for rest_v in range_constexpr(tApA.shape[0]):
        for m in range_constexpr(tApA.shape[1]):
            tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0])
    for rest_v in range_constexpr(tBpB.shape[0]):
        for n in range_constexpr(tBpB.shape[1]):
            tBpB[rest_v, n, 0] = cute.elem_less(tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0])

    # prologue: k-tiles 0 .. stages - 2 in flight
    k_pipe_max = cute.size(tAsA, mode=[3])
    k_tile_count = cute.size(tAgA, mode=[3])
    gmem_pipe_read = Int32(0)
    for k_tile in range_constexpr(k_pipe_max - 1):
        if const_expr(k_tile < k_tile_count):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, k_tile],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, gmem_pipe_read],
                tBsB[None, None, None, k_tile],
                pred=tBpB,
            )
            if const_expr(glu):
                cute.copy(
                    tiled_copy_B,
                    tBgBg[None, None, None, gmem_pipe_read],
                    tBsBg[None, None, None, k_tile],
                    pred=tBpB,
                )
        gmem_pipe_read = (
            gmem_pipe_read + 1 if gmem_pipe_read + 1 < k_tile_count else Int32(0)
        )
        cute.arch.cp_async_commit_group()

    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    # the accumulators by shape (the output may be a placeholder under split-K)
    acc_shape = thr_mma.partition_shape_C((bM, bN))
    tCrC = cute.make_rmem_tensor(acc_shape, Float32)
    tCrC.fill(0.0)
    if const_expr(glu):
        tCsBg = thr_mma.partition_B(sBg)
        tCrBg = tiled_mma.make_fragment_B(tCsBg[None, None, None, 0])
        tCrG = cute.make_rmem_tensor(acc_shape, Float32)
        tCrG.fill(0.0)

    smem_pipe_read = Int32(0)
    smem_pipe_write = Int32(k_pipe_max - 1)
    tCsA_p = tCsA[None, None, None, smem_pipe_read]
    tCsB_p = tCsB[None, None, None, smem_pipe_read]
    if const_expr(glu):
        tCsBg_p = tCsBg[None, None, None, smem_pipe_read]

    # register prefetch of k-block 0
    k_block_max = cute.size(tCrA, mode=[2])
    if const_expr(k_block_max > 1):
        cute.arch.cp_async_wait_group(k_pipe_max - 2)
        cute.arch.sync_threads()
        cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
        cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])
        if const_expr(glu):
            cute.autovec_copy(tCsBg_p[None, None, 0], tCrBg[None, None, 0])

    # mainloop: smem pipeline over k-tiles, register double buffer over k-blocks
    for _ in cutlass.range(k_tile_count):
        for k_block in range_constexpr(k_block_max):
            if const_expr(k_block == k_block_max - 1):
                tCsA_p = tCsA[None, None, None, smem_pipe_read]
                tCsB_p = tCsB[None, None, None, smem_pipe_read]
                if const_expr(glu):
                    tCsBg_p = tCsBg[None, None, None, smem_pipe_read]
                cute.arch.cp_async_wait_group(k_pipe_max - 2)
                cute.arch.sync_threads()

            # the register slots of this k-block and the next (all of K, or two)
            k_block_next = (k_block + 1) % k_block_max
            r_now = _slot(k_block, fragment_slots)
            r_next = _slot(k_block_next, fragment_slots)
            cute.autovec_copy(tCsA_p[None, None, k_block_next], tCrA[None, None, r_next])
            cute.autovec_copy(tCsB_p[None, None, k_block_next], tCrB[None, None, r_next])
            if const_expr(glu):
                cute.autovec_copy(
                    tCsBg_p[None, None, k_block_next], tCrBg[None, None, r_next]
                )

            # copy A, gemm, then copy B, to interleave memory and math
            if const_expr(k_block == 0):
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, gmem_pipe_read],
                    tAsA[None, None, None, smem_pipe_write],
                    pred=tApA,
                )

            cute.gemm(
                tiled_mma, tCrC, tCrA[None, None, r_now], tCrB[None, None, r_now], tCrC
            )
            if const_expr(glu):
                cute.gemm(
                    tiled_mma,
                    tCrG,
                    tCrA[None, None, r_now],
                    tCrBg[None, None, r_now],
                    tCrG,
                )

            if const_expr(k_block == 0):
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, smem_pipe_write],
                    pred=tBpB,
                )
                if const_expr(glu):
                    cute.copy(
                        tiled_copy_B,
                        tBgBg[None, None, None, gmem_pipe_read],
                        tBsBg[None, None, None, smem_pipe_write],
                        pred=tBpB,
                    )
                cute.arch.cp_async_commit_group()
                smem_pipe_write = smem_pipe_read
                smem_pipe_read = smem_pipe_read + 1
                if smem_pipe_read == k_pipe_max:
                    smem_pipe_read = Int32(0)
                # past the last k-tile the copies wrap to tile 0: in bounds, never consumed
                gmem_pipe_read = (
                    gmem_pipe_read + 1 if gmem_pipe_read + 1 < k_tile_count else Int32(0)
                )

    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # epilogue, by rows and 4-column blocks of this thread's accumulators (the
    # fragment is (1, rows, columns) for the FFMA atom, its columns in blocks of four
    # consecutive ones, see the permutation in launch_dense): per row the row-wise
    # inputs once, per block 16-byte vector loads and stores (or scalar ones without
    # vector_epilogue). Reads use clamped indices; a block is all in or all out of
    # the N edge (N_out % 4 == 0), so only the stores are predicated
    tCcC = thr_mma.partition_C(cute.make_identity_tensor((bM, bN)))
    m0 = bidx * bM
    n0 = bidy * bN
    assert cute.size(tCrC, mode=[0]) == 1 and cute.size(tCrC, mode=[2]) % 4 == 0
    vec = vector_epilogue
    N_acc = mW.shape[0]  # the accumulator width (2 F for glu)
    N_dact = N_out  # the output's and dact_h's width (2 F for the glu derivative)
    if const_expr(dact == "glu"):
        N_dact = 2 * N_out
    if const_expr(splits == 1):
        C4 = _chunks(mC, M, N_dact)
    if const_expr(save_acc):
        Acc4 = _chunks(mAcc, splits * M, N_acc)
    if const_expr(save_pre):
        Pre4 = _chunks(mPre, M, N_acc)
    if const_expr(has_bias):
        bias4 = _chunks(bias, 1, N_acc)
    if const_expr(has_pre_add):
        pre_add4 = _chunks(pre_add, M, N_acc)
    if const_expr(has_rank1):
        rank1_col4 = _chunks(rank1_col, 1, N_acc)
    if const_expr(has_residual):
        residual4 = _chunks(residual, M, N_out)
    if const_expr(has_residual2):
        residual24 = _chunks(residual2, M, N_out)
    if const_expr(dact != "none"):
        dact4 = _chunks(dact_h, M, N_dact)
    rA = cute.make_rmem_tensor(4, Float32)
    rG = cute.make_rmem_tensor(4, Float32)
    rV = cute.make_rmem_tensor(4, Float32)
    rW = cute.make_rmem_tensor(4, Float32)
    for mi in range_constexpr(cute.size(tCrC, mode=[1])):
        row_sum = Float32(0.0)
        row_sumsq = Float32(0.0)
        m = m0 + tCcC[0, mi, 0][0]
        mc = cutlass.min(m, M - 1)
        if const_expr(has_row_scale):
            sc = row_scale[mc]
            if const_expr(row_scale_mode == "rstd"):
                sc = cute.math.rsqrt(sc * Float32(rstd_inv_k) + Float32(rstd_eps))
        if const_expr(has_rank1):
            r1 = rank1_row[mc]
        if const_expr(has_row_mask):
            keep = row_mask[mc]
        for nb in range_constexpr(cute.size(tCrC, mode=[2]) // 4):
            n = n0 + tCcC[0, mi, 4 * nb][1]
            nc = cutlass.min(n, N_out - 4)
            inside = cute.elem_less((m, n), (M, N_out))
            for e in range_constexpr(4):
                rA[e] = tCrC[0, mi, 4 * nb + e]
                if const_expr(glu):
                    rG[e] = tCrG[0, mi, 4 * nb + e]
            if const_expr(save_acc):
                if inside:
                    _store4(Acc4, bidz * M + m, n, rA, vec)
            if const_expr(save_acc and glu):
                if inside:
                    _store4(Acc4, bidz * M + m, N_out + n, rG, vec)
            if const_expr(has_row_scale):
                for e in range_constexpr(4):
                    rA[e] = rA[e] * sc
                    if const_expr(glu):
                        rG[e] = rG[e] * sc
            if const_expr(has_bias):
                _add4(rA, bias4, 0, nc, vec, rV)
                if const_expr(glu):
                    _add4(rG, bias4, 0, N_out + nc, vec, rV)
            if const_expr(has_pre_add):
                _add4(rA, pre_add4, mc, nc, vec, rV)
                if const_expr(glu):
                    _add4(rG, pre_add4, mc, N_out + nc, vec, rV)
            if const_expr(has_rank1):
                _load4(rV, rank1_col4, 0, nc, vec)
                for e in range_constexpr(4):
                    rA[e] = rA[e] + r1 * rV[e]
                if const_expr(glu):
                    _load4(rV, rank1_col4, 0, N_out + nc, vec)
                    for e in range_constexpr(4):
                        rG[e] = rG[e] + r1 * rV[e]
            if const_expr(save_pre):
                if inside:
                    _store4(Pre4, m, n, rA, vec)
            if const_expr(save_pre and glu):
                if inside:
                    _store4(Pre4, m, N_out + n, rG, vec)
            if const_expr(activation == "silu"):
                for e in range_constexpr(4):
                    rA[e] = rA[e] * (Float32(1.0) / (Float32(1.0) + cute.math.exp(-rA[e])))
            if const_expr(glu):
                for e in range_constexpr(4):
                    rA[e] = rA[e] * (Float32(1.0) / (Float32(1.0) + cute.math.exp(-rG[e])))
            if const_expr(dact == "silu"):
                _load4(rV, dact4, mc, nc, vec)
                for e in range_constexpr(4):
                    sg = Float32(1.0) / (Float32(1.0) + cute.math.exp(-rV[e]))
                    rA[e] = rA[e] * sg * (Float32(1.0) + rV[e] * (Float32(1.0) - sg))
            if const_expr(dact == "glu"):
                _load4(rV, dact4, mc, nc, vec)
                _load4(rW, dact4, mc, N_out + nc, vec)
                for e in range_constexpr(4):
                    sg = Float32(1.0) / (Float32(1.0) + cute.math.exp(-rW[e]))
                    rG[e] = rA[e] * rV[e] * sg * (Float32(1.0) - sg)
                    rA[e] = rA[e] * sg
            if const_expr(has_residual):
                _add4(rA, residual4, mc, nc, vec, rV)
            if const_expr(has_residual2):
                _add4(rA, residual24, mc, nc, vec, rV)
            if const_expr(has_row_mask):
                for e in range_constexpr(4):
                    rA[e] = rA[e] * keep
                    if const_expr(dact == "glu"):
                        rG[e] = rG[e] * keep
            if const_expr(splits == 1):
                if inside:
                    _store4(C4, m, n, rA, vec)
            if const_expr(dact == "glu" and splits == 1):
                if inside:
                    _store4(C4, m, N_out + n, rG, vec)
            if const_expr(row_stats != "none"):
                for e in range_constexpr(4):
                    row_sum = row_sum + rA[e]
                    row_sumsq = row_sumsq + rA[e] * rA[e]

        # the row's threads are row_threads consecutive lanes (N-fastest thread
        # grid, the CTA owns whole rows); the first of them stores
        if const_expr(row_stats != "none"):
            for b in range_constexpr(row_threads.bit_length() - 1):
                off = row_threads >> (b + 1)
                row_sumsq = row_sumsq + cute.arch.shuffle_sync_bfly(row_sumsq, offset=off)
                if const_expr(row_stats == "both"):
                    row_sum = row_sum + cute.arch.shuffle_sync_bfly(row_sum, offset=off)
            if tidx % row_threads == 0:
                if m < M:
                    mSumsq[m] = row_sumsq
            if const_expr(row_stats == "both"):
                if tidx % row_threads == 0:
                    if m < M:
                        mSum[m] = row_sum


def _slot(k_block, slots):
    """The register-fragment slot of a k-block: its own (``slots = 0``, the
    fragments hold all of the k-tile and ptxas schedules the shared loads freely)
    or ``k_block % slots`` (a true double buffer for ``slots = 2``: reusing the
    registers bounds how far ptxas hoists the loads, and so the register count)."""
    return k_block % slots if slots else k_block


def _chunks(t, rows, width):
    """``t``'s memory as ``[rows, width / 4, 4]`` (``t`` row-major ``[rows,
    width]``, or ``[width]`` with ``rows = 1``): ``[r, c, None]`` is the 16-byte
    vector of columns ``4 c .. 4 c + 3`` of row ``r``."""
    return cute.make_tensor(
        t.iterator, cute.make_layout((rows, width // 4, 4), stride=(width, 4, 1))
    )


def _load4(dst, t4, row, col, vec):
    """``dst[0:4] = t[row, col : col + 4]`` from a ``_chunks`` view, ``col`` a
    multiple of 4: one 16-byte load, or four scalar ones without ``vec``."""
    if vec:
        cute.autovec_copy(t4[row, col // 4, None], dst)
    else:
        for e in range(4):
            dst[e] = t4[row, col // 4, e]


def _add4(acc, t4, row, col, vec, tmp):
    """``acc[0:4] += t[row, col : col + 4]``, through ``tmp``."""
    _load4(tmp, t4, row, col, vec)
    for e in range(4):
        acc[e] = acc[e] + tmp[e]


def _store4(t4, row, col, src, vec):
    """``t[row, col : col + 4] = src[0:4]`` into a ``_chunks`` view."""
    if vec:
        cute.autovec_copy(src, t4[row, col // 4, None])
    else:
        for e in range(4):
            t4[row, col // 4, e] = src[e]


@cute.jit
def launch_dense(
    stream: cuda.CUstream,
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    row_scale: cute.Tensor,
    pre_add: cute.Tensor,
    rank1_row: cute.Tensor,
    rank1_col: cute.Tensor,
    residual: cute.Tensor,
    residual2: cute.Tensor,
    row_mask: cute.Tensor,
    dact_h: cute.Tensor,
    out: cute.Tensor,
    pre: cute.Tensor,
    acc: cute.Tensor,
    row_sum: cute.Tensor,
    row_sumsq: cute.Tensor,
    *,
    bM: cutlass.Constexpr[int],
    bN: cutlass.Constexpr[int],
    bK: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    tm: cutlass.Constexpr[int],
    tn: cutlass.Constexpr[int],
    transpose_w: cutlass.Constexpr[bool],
    has_bias: cutlass.Constexpr[bool],
    has_row_scale: cutlass.Constexpr[bool],
    has_pre_add: cutlass.Constexpr[bool],
    has_rank1: cutlass.Constexpr[bool],
    save_pre: cutlass.Constexpr[bool],
    save_acc: cutlass.Constexpr[bool],
    activation: cutlass.Constexpr[str],
    has_residual: cutlass.Constexpr[bool],
    has_residual2: cutlass.Constexpr[bool],
    has_row_mask: cutlass.Constexpr[bool],
    dact: cutlass.Constexpr[str],
    splits: cutlass.Constexpr[int],
    row_stats: cutlass.Constexpr[str],
    row_scale_mode: cutlass.Constexpr[str] = "scale",
    rstd_eps: cutlass.Constexpr[float] = 0.0,
    vector_epilogue: cutlass.Constexpr[bool] = True,
    min_blocks: cutlass.Constexpr[int] = 0,
    fragment_slots: cutlass.Constexpr[int] = 0,
):
    """Builds the shared-memory, copy and MMA layouts for the tile and launches a
    CTA per ``(bM, bN)`` block of the GEMM output and per split of K. ``w`` comes
    in as the ``(N, K)`` tensor: N-major (the ``[K, N]`` view), or K-major for
    ``transpose_w``."""
    assert LayoutEnum.from_tensor(x) == LayoutEnum.ROW_MAJOR
    if const_expr(splits == 1):
        assert LayoutEnum.from_tensor(out) == LayoutEnum.ROW_MAJOR
    if const_expr(transpose_w):
        assert LayoutEnum.from_tensor(w) == LayoutEnum.ROW_MAJOR, "w is not K-major"
    else:
        assert LayoutEnum.from_tensor(w) == LayoutEnum.COL_MAJOR, "w is not N-major"
    K = x.shape[1]
    assert w.shape[1] == K, f"w is {w.shape} as (N, K), x has K={K}"
    assert row_scale_mode in ("scale", "rstd"), f"row_scale_mode {row_scale_mode!r}"
    assert fragment_slots in (0, 2), f"fragment_slots {fragment_slots} is not 0 or 2"
    assert tm in (4, 8) and tn in (4, 8), f"register tile {tm}x{tn} not in 4, 8"
    assert bM % tm == 0 and bN % tn == 0, f"{bM}x{bN} not a multiple of {tm}x{tn}"
    Tm = bM // tm
    Tn = bN // tn
    T = Tm * Tn
    assert T % 32 == 0 and T <= 1024, f"{T} threads for {bM}x{bN} at {tm}x{tn}"
    if const_expr(row_stats != "none"):
        assert splits == 1 and bN == out.shape[1], "row stats need bN = N_out"
        assert Tn <= 32 and Tn & (Tn - 1) == 0, f"a row's {Tn} threads are not in a warp"
    assert splits >= 1 and K % (splits * bK) == 0, f"K={K} not split {splits} x bK={bK}"
    assert stages >= 3 and K // (splits * bK) >= stages - 1, (
        "needs stages >= 3 and K / (splits bK) >= stages - 1"
    )
    assert bM % (4 * Tm) == 0 and bN % (4 * Tn) == 0, f"no 4x4 thread grid for {bM}x{bN}"
    assert T % bK == 0 and bM % (T // bK) == 0, f"A copy layout does not tile {bM}x{bK}"

    # smem: M-major A padded by 4 (its gmem tile is K-major); N-major B, padded by
    # 4 too when its gmem tile is K-major
    sA_layout = cute.make_layout((bM, bK, stages), stride=(1, bM + 4, bK * (bM + 4)))
    pad_B = 0
    if const_expr(transpose_w):
        pad_B = 4
    sB_layout = cute.make_layout(
        (bN, bK, stages), stride=(1, bN + pad_B, bK * (bN + pad_B))
    )

    # K-major tiles (A, and B for transpose_w): one element per cp.async, threads
    # along K first; N-major B: vec elements along N per cp.async
    tA = cute.make_layout((T // bK, bK), stride=(bK, 1))
    vA = cute.make_layout((1, 1))
    atom_A = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=Float32.width
    )
    if const_expr(transpose_w):
        assert bN % (T // bK) == 0, f"B copy layout does not tile {bN}x{bK}"
        tB = cute.make_layout((T // bK, bK), stride=(bK, 1))
        vB = cute.make_layout((1, 1))
        atom_B = atom_A
    else:
        vec = min(4, bN * bK // T)
        assert vec in (1, 2, 4) and bK % (T * vec // bN) == 0, (
            f"B copy does not tile {bN}x{bK}"
        )
        tB = cute.make_layout((bN // vec, T // (bN // vec)), stride=(1, bN // vec))
        vB = cute.make_layout((vec, 1))
        atom_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=Float32.width * vec
        )
    tiled_copy_A = cute.make_tiled_copy_tv(atom_A, tA, vA)
    tiled_copy_B = cute.make_tiled_copy_tv(atom_B, tB, vB)

    # (Tm, Tn) threads, N fastest (row-major out); the permutation gives each thread
    # blocks of four consecutive rows and columns, 4 Tm rows and 4 Tn columns apart
    # (tm / 4 and tn / 4 of them), so every smem -> register load is one 128-bit
    # LDS and a warp's loads hit distinct banks (eight consecutive values per
    # thread would be two LDS.128 with a 2-way bank conflict on B)
    atoms_layout = cute.make_layout((Tm, Tn, 1), stride=(Tn, 1, 0))
    permutation_M = cute.make_layout((Tm, 4), stride=(4, 1))
    permutation_N = cute.make_layout((Tn, 4), stride=(4, 1))
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.MmaUniversalOp(Float32),
        atoms_layout,
        permutation_mnk=(permutation_M, permutation_N, None),
    )

    N_out = w.shape[0]
    if const_expr(activation == "glu"):
        N_out = N_out // 2
    grid = (*cute.ceil_div((x.shape[0], N_out), (bM, bN)), splits)
    kernel = dense_kernel(
        x,
        w,
        bias,
        row_scale,
        pre_add,
        rank1_row,
        rank1_col,
        residual,
        residual2,
        row_mask,
        dact_h,
        out,
        pre,
        acc,
        row_sum,
        row_sumsq,
        sA_layout,
        sB_layout,
        tiled_copy_A,
        tiled_copy_B,
        tiled_mma,
        bM,
        bN,
        bK,
        has_bias,
        has_row_scale,
        has_pre_add,
        has_rank1,
        save_pre,
        save_acc,
        activation,
        has_residual,
        has_residual2,
        has_row_mask,
        dact,
        splits,
        row_stats,
        Tn,
        row_scale_mode,
        1.0 / K,
        rstd_eps,
        vector_epilogue,
        fragment_slots,
    )
    # min_blocks > 0 asks ptxas for that many CTAs per SM (it caps the registers)
    if const_expr(min_blocks > 0):
        kernel.launch(
            grid=grid, block=[T, 1, 1], stream=stream, min_blocks_per_mp=min_blocks
        )
    else:
        kernel.launch(grid=grid, block=[T, 1, 1], stream=stream)
