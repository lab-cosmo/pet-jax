# Derived from NVIDIA CUTLASS, examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py:
#
# Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""A TF32 tensor-core GEMM for SM100 (tcgen05 UMMA, TMA loads, TMEM accumulators)
with ``kernels.dense``'s fused epilogue, in the CuTe DSL; the mainloop is adapted
from CUTLASS's Blackwell tutorial GEMM (``fp16_gemm_0``). Imported lazily, so
``cutlass`` is needed only when a kernel runs.

``out[M, N_out] = epilogue(x[M, K] @ W[K, N])`` with fp32 operands read as TF32
and fp32 accumulation, one pass or 3xTF32 (``three_pass``): with the low parts
``x_lo = x - tf32(x)`` and ``w_lo`` as two more operands, each k-block issues
``x_lo w + x w_lo + x w`` into one accumulator (CUTLASS's "big + small" split; the
tensor core truncates each operand to TF32, exact for the big parts), which is
fp32-class.

Operands as JAX holds them: ``x`` row-major (K-major A); the weight as the
``(N, K)`` tensor sgemm calls B, either ``W.T`` stored ``[N, K]`` row-major
(``b_k_major``) or ``W [K, N]`` row-major through a ``(1, 0)`` view (MN-major,
which TF32 UMMA supports with the 128B / 32B-atom swizzle the layout helper
picks); both run at the same speed, so the backward's ``dx`` reads the forward
weight as it is. TMA zero-fills the M and K edges of the loads; ``N_out`` must be
a multiple of ``bN``.

One CTA per ``(128, bN)`` output tile, 128 threads: warp 0 issues the TMA loads and
the MMAs through a ``stages``-deep mbarrier pipeline; then every thread owns one
output row of the tile and runs the epilogue on it, in column sub-tiles moved
TMEM -> registers (32x32b loads) -> global memory with 16-byte accesses, the
stores (and the loads of ``[M, N]`` inputs) predicated on the M edge. The
epilogue is ``_gemm_cute``'s, in its order (row scale or RMSNorm factor, bias,
``pre_add``, rank-1 term, saved pre-activation, activation or its derivative,
residuals, row mask, row statistics); ``glu`` keeps the value and gate tiles
(columns ``n`` and ``F + n`` of the weight) in two TMEM accumulators, and the row
statistics need the CTA to own whole rows (``bN = N_out``), so no reduction
across threads is needed.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, TFloat32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05

THREADS = 128
BM = 128


@functools.cache
def _shared_storage(stages):
    """The shared-memory struct of the barriers: a TMA/UMMA pair per stage, a
    split-done pair per stage (the in-kernel split of A), one accumulator pair,
    and the TMEM address."""

    @cute.struct
    class SharedStorage:
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, stages * 2]
        split_mbar_ptr: cute.struct.MemRange[cutlass.Int64, stages * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
        tmem_holding_buf: cutlass.Int32

    return SharedStorage


# the warp roles of the in-kernel split (split_a): warp 0 issues the TMA loads,
# warp 1 the MMAs, warps 2 and 3 split each landed A stage into its TF32 parts
SPLIT_WARP0 = 2
SPLIT_THREADS = 64


def tmem_columns(cols):
    """TMEM columns to allocate for ``cols`` fp32 accumulator columns: a power of
    two, at least 32."""
    n = 32
    while n < cols:
        n *= 2
    return n


def smem_bytes(bN, bK, stages, three_pass, glu):
    """Shared memory of the operand pipeline: per stage the A tile and one or two
    B tiles (``glu``), twice with the low parts."""
    per_stage = 4 * bK * (BM + bN * (2 if glu else 1))
    return stages * per_stage * (2 if three_pass else 1)


def _rows(t, rows, row_stride, width, offset=0):
    """``t``'s memory from element ``offset`` on as a row-major ``[rows, width]``
    tensor of row stride ``row_stride``: a column vector broadcast to every row
    with stride 0, the second half of a ``[rows, 2 width]`` tensor with ``offset =
    width``, the one split of a ``[1, rows, width]`` saved accumulator."""
    return cute.make_tensor(
        t.iterator + offset, cute.make_layout((rows, width), stride=(row_stride, 1))
    )


def _operand(smem, layout, dtype=TFloat32):
    """A staged, swizzled operand tile in shared memory (``Float32`` for an A tile
    TMA loads raw, to be split in place)."""
    return smem.allocate_tensor(
        element_type=dtype, layout=layout.outer, byte_alignment=128, swizzle=layout.inner
    )


def _tma_a(atom, mT, sT, thr_mma, mma_tiler, coord):
    """The TMA partition ``(smem, gmem)`` of this CTA's A tiles."""
    g = cute.local_tile(mT, mma_tiler, coord, proj=(1, None, 1))
    return cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(sT, 0, 3),
        cute.group_modes(thr_mma.partition_A(g), 0, 3),
    )


def _tma_b(atom, mT, sT, thr_mma, mma_tiler, coord):
    """The TMA partition ``(smem, gmem)`` of this CTA's B tiles at ``coord``."""
    g = cute.local_tile(mT, mma_tiler, coord, proj=(None, 1, 1))
    return cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(sT, 0, 3),
        cute.group_modes(thr_mma.partition_B(g), 0, 3),
    )


def _load(dst, src, i):
    """``dst`` from sub-tile ``i`` of the partitioned ``src``, as a TensorSSA."""
    cute.autovec_copy(src[None, None, i], dst)
    return dst.load()


def _epi(t, mma_tiler, coord, thr_mma, epi_tiler, thr_copy):
    """A ``[M, N_out]``-shaped gmem tensor partitioned like this thread's
    accumulator values: ``(TmemCpy, NumTmemCpy, sub-tile)``."""
    g = cute.local_tile(t, mma_tiler, coord, proj=(1, 1, None))
    return thr_copy.partition_D(cute.zipped_divide(thr_mma.partition_C(g), epi_tiler))


@cute.kernel
def dense_tf32_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    tma_atom_a_lo: cute.CopyAtom,
    mA_lo: cute.Tensor,
    tma_atom_b_lo: cute.CopyAtom,
    mB_lo: cute.Tensor,
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
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    mma_tiler: cutlass.Constexpr,
    M: cutlass.Constexpr[int],
    N_out: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    three_pass: cutlass.Constexpr[bool],
    tmem_cols: cutlass.Constexpr[int],
    k_tiles: cutlass.Constexpr[int],
    subtiles: cutlass.Constexpr[int],
    has_bias: cutlass.Constexpr[bool],
    has_row_scale: cutlass.Constexpr[bool],
    row_scale_mode: cutlass.Constexpr[str],
    rstd_inv_k: cutlass.Constexpr[float],
    rstd_eps: cutlass.Constexpr[float],
    has_pre_add: cutlass.Constexpr[bool],
    has_rank1: cutlass.Constexpr[bool],
    save_pre: cutlass.Constexpr[bool],
    save_acc: cutlass.Constexpr[bool],
    activation: cutlass.Constexpr[str],
    dact: cutlass.Constexpr[str],
    has_residual: cutlass.Constexpr[bool],
    has_residual2: cutlass.Constexpr[bool],
    has_row_mask: cutlass.Constexpr[bool],
    row_stats: cutlass.Constexpr[str],
    splits: cutlass.Constexpr[int],
    split_a: cutlass.Constexpr[bool],
):
    """One ``(128, bN)`` tile of ``out`` over the k-range of split ``z`` (with
    ``splits > 1`` only the raw partial accumulators are stored)."""
    glu = activation == "glu"
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, bidz = cute.arch.block_idx()
    coord = (bidx, bidy, None)
    # the gate half of the weight for glu: rows F + n of the (N, K) tensor
    coord_g = (bidx, bidy + N_out // mma_tiler[1], None)

    smem = utils.SmemAllocator()
    storage = smem.allocate(_shared_storage(stages))

    # with split_a, TMA loads A raw (fp32) and the threads split it in place (off by
    # default: both the one-warp and the warp-specialised split measured slower than
    # XLA's split of x)
    sA = _operand(smem, a_smem_layout, Float32 if split_a else TFloat32)
    sB = _operand(smem, b_smem_layout)
    if const_expr(glu):
        sBg = _operand(smem, b_smem_layout)
    if const_expr(three_pass):
        sA_lo = _operand(smem, a_smem_layout)
        sB_lo = _operand(smem, b_smem_layout)
        if const_expr(glu):
            sBg_lo = _operand(smem, b_smem_layout)

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier
    )
    tmem.allocate(tmem_cols)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        if const_expr(three_pass and not split_a):
            cpasync.prefetch_descriptor(tma_atom_a_lo)
        if const_expr(three_pass):
            cpasync.prefetch_descriptor(tma_atom_b_lo)

    # bytes the TMA loads of one stage deliver
    a_bytes = cute.size_in_bytes(TFloat32, cute.select(a_smem_layout, mode=[0, 1, 2]))
    b_bytes = cute.size_in_bytes(TFloat32, cute.select(b_smem_layout, mode=[0, 1, 2]))
    if const_expr(glu):
        b_bytes = 2 * b_bytes
    tma_bytes = a_bytes + b_bytes
    if const_expr(three_pass):
        tma_bytes = tma_bytes + b_bytes
    if const_expr(three_pass and not split_a):
        tma_bytes = tma_bytes + a_bytes
    ab_pipe = pipeline.PipelineTmaUmma.create(
        num_stages=stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=tma_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    )
    ab_producer, ab_consumer = ab_pipe.make_participants()
    if const_expr(split_a):
        # each warp role gets its own participants, made here, so no pipeline state
        # crosses from one role's region into another's: the split warps wait for
        # the TMA loads (and never release the stage, the MMA warp does), and a
        # second pipeline carries "stage split" from them to the MMA warp
        _, ab_split_consumer = ab_pipe.make_participants()
        split_producer, split_consumer = pipeline.PipelineAsync.create(
            num_stages=stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, SPLIT_THREADS),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            barrier_storage=storage.split_mbar_ptr.data_ptr(),
        ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    thr_mma = tiled_mma.get_slice(0)

    tAsA, tAgA = _tma_a(tma_atom_a, mA, sA, thr_mma, mma_tiler, coord)
    tBsB, tBgB = _tma_b(tma_atom_b, mB, sB, thr_mma, mma_tiler, coord)
    sA_mma = sA
    if const_expr(split_a):
        # the raw tile seen as TF32 for the UMMA descriptors, keeping its 128B
        # swizzle: the swizzle lives in the pointer type, which recast_tensor (and
        # recast_ptr without swizzle_) rebuilds without it
        sA_mma = cute.make_tensor(
            cute.recast_ptr(sA.iterator, swizzle_=a_smem_layout.inner, dtype=TFloat32),
            sA.layout,
        )
    tCrA = tiled_mma.make_fragment_A(sA_mma)
    tCrB = tiled_mma.make_fragment_B(sB)
    if const_expr(glu):
        tBsBg, tBgBg = _tma_b(tma_atom_b, mB, sBg, thr_mma, mma_tiler, coord_g)
        tCrBg = tiled_mma.make_fragment_B(sBg)
    if const_expr(three_pass and not split_a):
        tAsA_lo, tAgA_lo = _tma_a(tma_atom_a_lo, mA_lo, sA_lo, thr_mma, mma_tiler, coord)
    if const_expr(three_pass):
        tBsB_lo, tBgB_lo = _tma_b(tma_atom_b_lo, mB_lo, sB_lo, thr_mma, mma_tiler, coord)
        tCrA_lo = tiled_mma.make_fragment_A(sA_lo)
        tCrB_lo = tiled_mma.make_fragment_B(sB_lo)
        if const_expr(glu):
            tBsBg_lo, tBgBg_lo = _tma_b(
                tma_atom_b_lo, mB_lo, sBg_lo, thr_mma, mma_tiler, coord_g
            )
            tCrBg_lo = tiled_mma.make_fragment_B(sBg_lo)

    # the accumulators in TMEM: (MMA, MMA_M, MMA_N, value / gate)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
    n_acc = 1
    if const_expr(glu):
        n_acc = 2
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, n_acc))
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    tCtAcc_all = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
    tCtAcc = tCtAcc_all[None, None, None, 0]
    if const_expr(glu):
        tCtAccG = tCtAcc_all[None, None, None, 1]

    n_blocks = cute.size(tCrA, mode=[2])
    # this split's k-tiles, k_tiles, comes from the host as a Python int, so the
    # loop's trip count is a literal for the prefetch_stages pipelining pass
    # whatever cute.size returns; unsplit, the TMA coordinate is the pipeline's count
    if const_expr(splits > 1):
        k_off = bidz * k_tiles
    # registers for the in-kernel split of A, and its flat view geometry: the
    # stage's memory elementwise (A and A_lo share one swizzled layout), four
    # floats per access, the split warps' threads striding through it
    if const_expr(split_a):
        r_in = cute.make_rmem_tensor(4, Float32)
        r_hi_bits = cute.make_rmem_tensor(4, cutlass.Int32)
        r_hi = cute.recast_tensor(r_hi_bits, Float32)
        r_lo = cute.make_rmem_tensor(4, Float32)
        n_a = cute.size(sA[None, None, None, 0])

    if const_expr(not split_a):
        # warp 0 issues the TMA loads (stages - 2 ahead) and the MMAs. All stateful
        # calls (the pipelines, the MMA's ACCUMULATE field) stay inline in this body:
        # the tracer threads them across the staged loop only when it sees them here
        if warp_idx == 0:
            acc_empty = acc_producer.acquire_and_advance()
            for _ in cutlass.range(k_tiles, prefetch_stages=stages - 2):
                ee = ab_producer.acquire_and_advance()
                if const_expr(splits == 1):
                    src, dst, bar = (None, ee.count), (None, ee.index), ee.barrier
                else:
                    src, dst, bar = (None, ee.count + k_off), (None, ee.index), ee.barrier
                cute.copy(tma_atom_a, tAgA[src], tAsA[dst], tma_bar_ptr=bar)
                cute.copy(tma_atom_b, tBgB[src], tBsB[dst], tma_bar_ptr=bar)
                if const_expr(glu):
                    cute.copy(tma_atom_b, tBgBg[src], tBsBg[dst], tma_bar_ptr=bar)
                if const_expr(three_pass):
                    cute.copy(tma_atom_a_lo, tAgA_lo[src], tAsA_lo[dst], tma_bar_ptr=bar)
                    cute.copy(tma_atom_b_lo, tBgB_lo[src], tBsB_lo[dst], tma_bar_ptr=bar)
                    if const_expr(glu):
                        cute.copy(
                            tma_atom_b_lo, tBgBg_lo[src], tBsBg_lo[dst], tma_bar_ptr=bar
                        )

                f = ab_consumer.wait_and_advance()
                for k_block in cutlass.range_constexpr(n_blocks):
                    kc = (None, None, k_block, f.index)
                    # the small products first, the big one last (CUTLASS's order);
                    # the first product of each accumulator overwrites it
                    if const_expr(three_pass):
                        cute.gemm(tiled_mma, tCtAcc, tCrA_lo[kc], tCrB[kc], tCtAcc)
                        if const_expr(glu):
                            cute.gemm(tiled_mma, tCtAccG, tCrA_lo[kc], tCrBg[kc], tCtAccG)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[kc], tCrB_lo[kc], tCtAcc)
                        if const_expr(glu):
                            cute.gemm(tiled_mma, tCtAccG, tCrA[kc], tCrBg_lo[kc], tCtAccG)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[kc], tCrB[kc], tCtAcc)
                    if const_expr(glu):
                        cute.gemm(tiled_mma, tCtAccG, tCrA[kc], tCrBg[kc], tCtAccG)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                f.release()
            acc_empty.commit()
    else:
        # warp-specialised: each role's loop and state live in its own region.
        # Warp 0: the TMA loads (A raw, B and B_lo rounded to TF32 by the tensor map)
        if warp_idx == 0:
            for _ in cutlass.range(k_tiles):
                ee = ab_producer.acquire_and_advance()
                if const_expr(splits == 1):
                    src, dst, bar = (None, ee.count), (None, ee.index), ee.barrier
                else:
                    src, dst, bar = (None, ee.count + k_off), (None, ee.index), ee.barrier
                cute.copy(tma_atom_a, tAgA[src], tAsA[dst], tma_bar_ptr=bar)
                cute.copy(tma_atom_b, tBgB[src], tBsB[dst], tma_bar_ptr=bar)
                if const_expr(glu):
                    cute.copy(tma_atom_b, tBgBg[src], tBsBg[dst], tma_bar_ptr=bar)
                cute.copy(tma_atom_b_lo, tBgB_lo[src], tBsB_lo[dst], tma_bar_ptr=bar)
                if const_expr(glu):
                    cute.copy(tma_atom_b_lo, tBgBg_lo[src], tBsBg_lo[dst], tma_bar_ptr=bar)
        # warps 2 and 3: when a stage lands, A in place becomes hi = cvt.rna.tf32(a)
        # and A_lo gets a - hi (generic-proxy writes, fenced for the tensor core's
        # async proxy), then "stage split"
        if warp_idx >= SPLIT_WARP0:
            lane = tidx - 32 * SPLIT_WARP0
            for _ in cutlass.range(k_tiles):
                f = ab_split_consumer.wait_and_advance()
                p = split_producer.acquire_and_advance()
                flat = cute.make_tensor(
                    sA[None, None, None, f.index].iterator, cute.make_layout((4, n_a // 4))
                )
                flat_lo = cute.make_tensor(
                    sA_lo[None, None, None, f.index].iterator,
                    cute.make_layout((4, n_a // 4)),
                )
                for j in cutlass.range_constexpr(n_a // 4 // SPLIT_THREADS):
                    c = lane + SPLIT_THREADS * j
                    cute.autovec_copy(flat[None, c], r_in)
                    for q in cutlass.range_constexpr(4):
                        r_hi_bits[q] = cutlass.Int32(cute.arch.cvt_f32_tf32(r_in[q]))
                    for q in cutlass.range_constexpr(4):
                        r_lo[q] = r_in[q] - r_hi[q]
                    cute.autovec_copy(r_hi, flat[None, c])
                    cute.autovec_copy(r_lo, flat_lo[None, c])
                cute.arch.fence_proxy("async.shared", space="cta")
                p.commit()
        # warp 1: per stage, wait for the split, issue the MMAs, release the stage
        if warp_idx == 1:
            acc_empty = acc_producer.acquire_and_advance()
            for _ in cutlass.range(k_tiles):
                sp = split_consumer.wait_and_advance()
                f = ab_consumer.wait_and_advance()
                for k_block in cutlass.range_constexpr(n_blocks):
                    kc = (None, None, k_block, f.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrA_lo[kc], tCrB[kc], tCtAcc)
                    if const_expr(glu):
                        cute.gemm(tiled_mma, tCtAccG, tCrA_lo[kc], tCrBg[kc], tCtAccG)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[kc], tCrB_lo[kc], tCtAcc)
                    if const_expr(glu):
                        cute.gemm(tiled_mma, tCtAccG, tCrA[kc], tCrBg_lo[kc], tCtAccG)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[kc], tCrB[kc], tCtAcc)
                    if const_expr(glu):
                        cute.gemm(tiled_mma, tCtAccG, tCrA[kc], tCrBg[kc], tCtAccG)
                f.release()
                sp.release()
            acc_empty.commit()

    # epilogue: each thread one row of the tile, in sub-tiles of columns
    epi_tiler = (
        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtiles),
    )
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(
        tmem_atom, cute.zipped_divide(tCtAcc, epi_tiler)[None, 0]
    )
    thr_copy = tmem_tiled_copy.get_slice(tidx)

    tDtC = thr_copy.partition_S(cute.zipped_divide(tCtAcc, epi_tiler))
    if const_expr(glu):
        tDtG = thr_copy.partition_S(cute.zipped_divide(tCtAccG, epi_tiler))
    E = (mma_tiler, coord, thr_mma, epi_tiler, thr_copy)
    tDcC = _epi(cute.make_identity_tensor((M, N_out)), *E)
    N_acc = N_out  # the width of the [M, N] inputs and saved outputs (2 F for glu)
    if const_expr(glu):
        N_acc = 2 * N_out
    N_c = N_out  # the output's and dact_h's width (2 F for the glu derivative)
    if const_expr(dact == "glu"):
        N_c = 2 * N_out
    if const_expr(splits == 1):
        tDgC = _epi(_rows(mC, M, N_c, N_out), *E)
    if const_expr(dact == "glu"):
        tDgC2 = _epi(_rows(mC, M, N_c, N_out, N_out), *E)
    if const_expr(save_acc and splits == 1):
        # (no offset at all unsplit: a staged offset, even a zero one, loses the
        # pointer's 16-byte alignment and the saved accumulator's stores go scalar,
        # a visible cost in the forward under jax.grad, which saves the accumulator
        # of every row-scaled GEMM)
        tDgAcc = _epi(_rows(mAcc, M, N_acc, N_out), *E)
        if const_expr(glu):
            tDgAccG = _epi(_rows(mAcc, M, N_acc, N_out, N_out), *E)
    if const_expr(save_acc and splits > 1):
        # split z's partials at [z] of the [splits, M, N] accumulator
        z0 = bidz * (M * N_acc)
        tDgAcc = _epi(_rows(mAcc, M, N_acc, N_out, z0), *E)
        if const_expr(glu):
            tDgAccG = _epi(_rows(mAcc, M, N_acc, N_out, z0 + N_out), *E)
    if const_expr(save_pre):
        tDgPre = _epi(_rows(mPre, M, N_acc, N_out), *E)
        if const_expr(glu):
            tDgPreG = _epi(_rows(mPre, M, N_acc, N_out, N_out), *E)
    if const_expr(has_bias):
        tDgBias = _epi(_rows(bias, M, 0, N_out), *E)
        if const_expr(glu):
            tDgBiasG = _epi(_rows(bias, M, 0, N_out, N_out), *E)
    if const_expr(has_pre_add):
        tDgPreAdd = _epi(_rows(pre_add, M, N_acc, N_out), *E)
        if const_expr(glu):
            tDgPreAddG = _epi(_rows(pre_add, M, N_acc, N_out, N_out), *E)
    if const_expr(has_rank1):
        tDgR1 = _epi(_rows(rank1_col, M, 0, N_out), *E)
        if const_expr(glu):
            tDgR1G = _epi(_rows(rank1_col, M, 0, N_out, N_out), *E)
    if const_expr(has_residual):
        tDgRes = _epi(_rows(residual, M, N_out, N_out), *E)
    if const_expr(has_residual2):
        tDgRes2 = _epi(_rows(residual2, M, N_out, N_out), *E)
    if const_expr(dact != "none"):
        tDgH = _epi(_rows(dact_h, M, N_c, N_out), *E)
        if const_expr(dact == "glu"):
            tDgHG = _epi(_rows(dact_h, M, N_c, N_out, N_out), *E)

    shape = tDcC[None, None, 0].shape
    rA = cute.make_rmem_tensor(shape, Float32)
    rG = cute.make_rmem_tensor(shape, Float32)
    rT = cute.make_rmem_tensor(shape, Float32)
    rU = cute.make_rmem_tensor(shape, Float32)

    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()

    row = tDcC[None, None, 0][0][0]
    mc = cutlass.min(row, M - 1)
    inside = row < M
    if const_expr(has_row_scale):
        sc = row_scale[mc]
        if const_expr(row_scale_mode == "rstd"):
            sc = cute.math.rsqrt(sc * Float32(rstd_inv_k) + Float32(rstd_eps))
    if const_expr(has_rank1):
        r1 = rank1_row[mc]
    if const_expr(has_row_mask):
        keep = row_mask[mc]
    row_sum = Float32(0.0)
    row_sumsq = Float32(0.0)

    for i in cutlass.range_constexpr(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], rA)
        a = rA.load()
        if const_expr(glu):
            cute.copy(tmem_tiled_copy, tDtG[None, None, i], rG)
            g = rG.load()
        if const_expr(save_acc):
            if inside:
                cute.autovec_copy(rA, tDgAcc[None, None, i])
        if const_expr(save_acc and glu):
            if inside:
                cute.autovec_copy(rG, tDgAccG[None, None, i])
        if const_expr(has_row_scale):
            a = a * sc
            if const_expr(glu):
                g = g * sc
        if const_expr(has_bias):
            a = a + _load(rT, tDgBias, i)
            if const_expr(glu):
                g = g + _load(rT, tDgBiasG, i)
        if const_expr(has_pre_add):
            if inside:
                cute.autovec_copy(tDgPreAdd[None, None, i], rT)
            a = a + rT.load()
        if const_expr(has_pre_add and glu):
            if inside:
                cute.autovec_copy(tDgPreAddG[None, None, i], rT)
            g = g + rT.load()
        if const_expr(has_rank1):
            a = a + r1 * _load(rT, tDgR1, i)
            if const_expr(glu):
                g = g + r1 * _load(rT, tDgR1G, i)
        if const_expr(save_pre):
            rU.store(a)
            if inside:
                cute.autovec_copy(rU, tDgPre[None, None, i])
        if const_expr(save_pre and glu):
            rU.store(g)
            if inside:
                cute.autovec_copy(rU, tDgPreG[None, None, i])
        if const_expr(activation == "silu"):
            a = a / (1.0 + cute.math.exp(-a))
        if const_expr(glu):
            a = a / (1.0 + cute.math.exp(-g))
        if const_expr(dact == "silu"):
            if inside:
                cute.autovec_copy(tDgH[None, None, i], rT)
            h = rT.load()
            sg = 1.0 / (1.0 + cute.math.exp(-h))
            a = a * sg * (1.0 + h * (1.0 - sg))
        if const_expr(dact == "glu"):
            if inside:
                cute.autovec_copy(tDgH[None, None, i], rT)
                cute.autovec_copy(tDgHG[None, None, i], rU)
            v = rT.load()
            sg = 1.0 / (1.0 + cute.math.exp(-rU.load()))
            d_gate = a * v * sg * (1.0 - sg)
            a = a * sg
        if const_expr(has_residual):
            if inside:
                cute.autovec_copy(tDgRes[None, None, i], rT)
            a = a + rT.load()
        if const_expr(has_residual2):
            if inside:
                cute.autovec_copy(tDgRes2[None, None, i], rT)
            a = a + rT.load()
        if const_expr(has_row_mask):
            a = a * keep
            if const_expr(dact == "glu"):
                d_gate = d_gate * keep
        if const_expr(splits == 1):
            rU.store(a)
            if inside:
                cute.autovec_copy(rU, tDgC[None, None, i])
        if const_expr(dact == "glu"):
            rU.store(d_gate)
            if inside:
                cute.autovec_copy(rU, tDgC2[None, None, i])
        if const_expr(row_stats != "none"):
            row_sum = row_sum + a.reduce(cute.ReductionOp.ADD, 0.0, 0)
            row_sumsq = row_sumsq + (a * a).reduce(cute.ReductionOp.ADD, 0.0, 0)
    acc_full.release()

    # the CTA owns whole rows (bN = N_out): this thread's sums are the row's
    if const_expr(row_stats != "none"):
        if inside:
            mSumsq[row] = row_sumsq
    if const_expr(row_stats == "both"):
        if inside:
            mSum[row] = row_sum

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def launch_dense_tf32(
    stream: cuda.CUstream,
    x: cute.Tensor,
    w: cute.Tensor,
    x_lo: cute.Tensor,
    w_lo: cute.Tensor,
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
    bN: cutlass.Constexpr[int],
    bK: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    b_k_major: cutlass.Constexpr[bool],
    three_pass: cutlass.Constexpr[bool],
    subtiles: cutlass.Constexpr[int],
    has_bias: cutlass.Constexpr[bool],
    has_row_scale: cutlass.Constexpr[bool],
    row_scale_mode: cutlass.Constexpr[str],
    rstd_eps: cutlass.Constexpr[float],
    has_pre_add: cutlass.Constexpr[bool],
    has_rank1: cutlass.Constexpr[bool],
    save_pre: cutlass.Constexpr[bool],
    save_acc: cutlass.Constexpr[bool],
    activation: cutlass.Constexpr[str],
    dact: cutlass.Constexpr[str],
    has_residual: cutlass.Constexpr[bool],
    has_residual2: cutlass.Constexpr[bool],
    has_row_mask: cutlass.Constexpr[bool],
    row_stats: cutlass.Constexpr[str],
    splits: cutlass.Constexpr[int] = 1,
    split_a: cutlass.Constexpr[bool] = False,
):
    """Builds the tiled MMA, the swizzled shared layouts and the TMA atoms (their
    descriptors are encoded on the host from the runtime pointers) and launches a
    CTA per ``(128, bN)`` tile. ``w`` (and ``w_lo``) come in as the ``(N, K)``
    tensor; switched-off inputs and outputs are ``[1]`` placeholders, as in
    ``_gemm_cute.launch_dense``."""
    M, K = x.shape
    N_out = w.shape[0]
    if const_expr(activation == "glu"):
        N_out = N_out // 2
    assert w.shape[1] == K, f"w is {w.shape} as (N, K), x has K={K}"
    assert K % (splits * bK) == 0, (
        f"K={K} does not split {splits} ways into k-tiles of {bK}"
    )
    assert not split_a or three_pass, "the in-kernel split of A is for 3xTF32"
    assert N_out % bN == 0, f"N_out={N_out} is not a multiple of bN={bN}"
    assert bN % 16 == 0 and 16 <= bN <= 256, f"bN={bN} is not a TF32 UMMA width"
    assert bK % 8 == 0, f"bK={bK} is not a multiple of the TF32 UMMA K of 8"
    assert (bN // subtiles) % 32 == 0, "an epilogue sub-tile is a multiple of 32 columns"
    if const_expr(row_stats != "none"):
        assert bN == N_out, "row statistics need the CTA to own whole rows (bN = N_out)"
    b_major = tcgen05.OperandMajorMode.MN
    if const_expr(b_k_major):
        b_major = tcgen05.OperandMajorMode.K
    op = tcgen05.MmaTF32Op(
        (BM, bN, 8),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        b_major,
    )
    tiled_mma = cute.make_tiled_mma(op)
    mma_tiler = (BM, bN, bK)
    a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, TFloat32, stages)
    b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, TFloat32, stages)
    a_one = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_one = cute.select(b_smem_layout, mode=[0, 1, 2])
    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    # TF32 tensor maps round each element to the nearest TF32 on the way in (the
    # one-pass result is cuBLAS's TF32 GEMM bit for bit, and the 3xTF32 error is
    # fp32-class with the low parts split by rounding); for split_a, A comes raw
    if const_expr(split_a):
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, x, a_one, mma_tiler, tiled_mma
        )
    else:
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, x, a_one, mma_tiler, tiled_mma, internal_type=TFloat32
        )
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op, w, b_one, mma_tiler, tiled_mma, internal_type=TFloat32
    )
    # without three_pass the low-part atoms are the main ones again, never used
    tma_atom_a_lo, tma_tensor_a_lo = tma_atom_a, tma_tensor_a
    tma_atom_b_lo, tma_tensor_b_lo = tma_atom_b, tma_tensor_b
    if const_expr(three_pass and not split_a):
        tma_atom_a_lo, tma_tensor_a_lo = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, x_lo, a_one, mma_tiler, tiled_mma, internal_type=TFloat32
        )
    if const_expr(three_pass):
        tma_atom_b_lo, tma_tensor_b_lo = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, w_lo, b_one, mma_tiler, tiled_mma, internal_type=TFloat32
        )
    acc_cols = bN
    if const_expr(activation == "glu"):
        acc_cols = 2 * bN

    grid = (cute.ceil_div(M, BM), N_out // bN, splits)
    dense_tf32_kernel(
        tiled_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_a_lo,
        tma_tensor_a_lo,
        tma_atom_b_lo,
        tma_tensor_b_lo,
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
        a_smem_layout,
        b_smem_layout,
        mma_tiler,
        M,
        N_out,
        stages,
        three_pass,
        tmem_columns(acc_cols),
        K // (bK * splits),
        subtiles,
        has_bias,
        has_row_scale,
        row_scale_mode,
        1.0 / K,
        rstd_eps,
        has_pre_add,
        has_rank1,
        save_pre,
        save_acc,
        activation,
        dact,
        has_residual,
        has_residual2,
        has_row_mask,
        row_stats,
        splits,
        split_a,
    ).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)
