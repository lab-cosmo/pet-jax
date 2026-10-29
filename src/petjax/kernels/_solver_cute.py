"""The CuTe DSL kernels behind ``kernels.solver``: the ``sorted_window`` adaptive
cutoff solve (``select.get_adaptive_cutoffs_window``) with a warp per atom.
Imported lazily, so ``cutlass`` is needed only when a kernel runs.

Forward, per atom: the warp loads the atom's raw pairs (the first ``capacity`` of its
run of the center-sorted list, given by ``starts`` / ``counts``; a masked pair
is a missing one at ``FAR``), sorts them by (distance, raw index)
with a bitonic network over registers and shuffles, takes the bracket
``[a, b]`` and the count ``lo`` below it from the ``k + 2`` nearest, reads the
window of ``WC`` entries from ``lo`` back out of shared memory, runs the
Newton-bisection on it with warp sums, and finishes with the
implicit-function step on the same window. It writes the cutoff, the root
iterate ``r``, the clamped slope ``dn`` (zero where the cutoff was clipped,
which switches the gradient off) and the window's raw pair indices (``-1``
for an empty entry) for the backward, plus an overflow flag (an unmasked pair past
``capacity``, or a pair below ``b`` beyond the window).

Backward, per atom: the warp zeroes the gradient of its run and writes, for
every window pair, ``d_cutoff * (-1 / dn) * d bump(d, r) / d d`` with the
derivative of ``utils.cutoff_bump`` as JAX takes it (zero outside the clip of
its argument). Every raw pair lies in exactly one run, so the writes cover the
whole gradient without atomics or a zero-initialised buffer.
"""

import numpy as np

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr, range_constexpr

WARP = 32
WARPS_PER_CTA = 4
FAR = 1e30  # select._FAR, the distance of a missing pair
PI = float(np.pi)
INF = float("inf")


def _add(a, b):
    return a + b


def _warp_sum(x):
    return cute.arch.warp_reduction(x, _add)


def _warp_max(x):
    return cute.arch.warp_reduction(x, cute.arch.fmax)


def _warp_min(x):
    return cute.arch.warp_reduction(x, cute.arch.fmin)


def _warp_max_int(x):
    return cute.arch.warp_reduction(x, cutlass.max)


def _self(cond, a, b):
    return Float32(cutlass.select_(cond, a, b))


def _seli(cond, a, b):
    return Int32(cutlass.select_(cond, a, b))


def _selb(cond, a, b):
    return cutlass.Boolean(cutlass.select_(cond, a, b))


def _less(ad, ar, bd, br):
    """``(ad, ar) < (bd, br)``: by distance, then by raw index, so the sort is
    the stable one of the stock path."""
    return (ad < bd) | ((ad == bd) & (ar < br))


# -- forward --


@cute.kernel
def forward_kernel(
    r_ij: cute.Tensor,
    pmask: cute.Tensor,
    starts: cute.Tensor,
    counts: cute.Tensor,
    rp: cute.Tensor,
    cutoffs: cute.Tensor,
    root: cute.Tensor,
    slope: cute.Tensor,
    win: cute.Tensor,
    flags: cute.Tensor,
    CAP: cutlass.Constexpr[int],
    WC: cutlass.Constexpr[int],
    KN: cutlass.Constexpr[int],
    NW: cutlass.Constexpr[int],
    NA: cutlass.Constexpr[int],
    k: cutlass.Constexpr[float],
    cutoff: cutlass.Constexpr[float],
    width: cutlass.Constexpr[float],
    iterations: cutlass.Constexpr[int],
):
    R = CAP // WARP
    LOG = CAP.bit_length() - 1
    Q = WC // WARP
    inv_cutoff = 1.0 / cutoff
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * NW + warp

    smem = cutlass.memory.SmemAllocator()
    sD = smem.allocate_tensor(
        Float32, cute.make_layout((NW, CAP), stride=(CAP, 1)), byte_alignment=16
    )
    sR = smem.allocate_tensor(
        Int32, cute.make_layout((NW, CAP), stride=(CAP, 1)), byte_alignment=16
    )

    if i < NA:
        start = starts[i]
        count = counts[i]
        # an unmasked pair past the capacity is one the solve does not see
        over = Int32(0)
        for c in cutlass.range(CAP + lane, count, WARP):
            over = cutlass.max(over, _seli(pmask[start + c] > 0, Int32(1), Int32(0)))
        over = _warp_max_int(over)

        # the atom's pairs, element e = reg * 32 + lane; a masked pair is missing
        d = []
        raw = []
        for r in range_constexpr(R):
            c = r * WARP + lane
            inside = c < count
            idx = _seli(inside, start + c, Int32(0))
            ok = inside & (pmask[idx] > 0)
            d.append(_self(ok, r_ij[idx], Float32(FAR)))
            raw.append(_seli(ok, idx, Int32(-1)))

        # bitonic sort, ascending by (distance, raw index); every compare-exchange is
        # a pair of selects, no branches
        for size_log in range_constexpr(1, LOG + 1):
            size = 1 << size_log
            for stride_log in range_constexpr(size_log - 1, -1, -1):
                stride = 1 << stride_log
                if const_expr(stride >= WARP):
                    # the partner sits in another register of the same lane
                    rs = stride // WARP
                    for r in range_constexpr(R):
                        if const_expr((r & rs) == 0):
                            pr = r | rs
                            if const_expr(((r * WARP) & size) == 0):
                                swap = _less(d[pr], raw[pr], d[r], raw[r])
                            else:
                                swap = _less(d[r], raw[r], d[pr], raw[pr])
                            lo_d = _self(swap, d[pr], d[r])
                            lo_r = _seli(swap, raw[pr], raw[r])
                            hi_d = _self(swap, d[r], d[pr])
                            hi_r = _seli(swap, raw[r], raw[pr])
                            d[r] = lo_d
                            raw[r] = lo_r
                            d[pr] = hi_d
                            raw[pr] = hi_r
                else:
                    # the partner sits in another lane, same register
                    for r in range_constexpr(R):
                        od = cute.arch.shuffle_sync_bfly(d[r], offset=stride)
                        orw = cute.arch.shuffle_sync_bfly(raw[r], offset=stride)
                        if const_expr(size >= WARP):
                            # the block's direction is bit ``size`` of ``e``, a register bit
                            if const_expr(((r * WARP) & size) == 0):
                                want_min = (lane & stride) == 0
                            else:
                                want_min = (lane & stride) != 0
                        else:
                            want_min = (
                                ((lane >> stride_log) ^ (lane >> size_log)) & 1
                            ) == 0
                        take = _selb(
                            want_min,
                            _less(od, orw, d[r], raw[r]),
                            _less(d[r], raw[r], od, orw),
                        )
                        d[r] = _self(take, od, d[r])
                        raw[r] = _seli(take, orw, raw[r])

        for r in range_constexpr(R):
            sD[warp, r * WARP + lane] = d[r]
            sR[warp, r * WARP + lane] = raw[r]
        cute.arch.sync_warp()

        # the bracket from the k + 2 nearest: lane p holds d_(p+1), FAR past the row
        near = d[0]
        prev = cute.arch.shuffle_sync_up(near, offset=1)
        prev = _self(lane == 0, Float32(0.0), prev)
        in_kn = lane < KN
        rpl = rp[cutlass.min(lane, KN - 1)]
        ca = _self(in_kn & (prev < rpl), cute.arch.fmin(near, rpl), Float32(-INF))
        cb = _self(
            in_kn & (rpl < near + width), cute.arch.fmax(prev + width, rpl), Float32(INF)
        )
        a = _warp_max(ca)
        b = cute.arch.fmin(_warp_min(cb), Float32(cutoff))
        below_a = in_kn & (near <= a - width)
        lo_i = _warp_sum(_seli(below_a, Int32(1), Int32(0)))
        lo_f = _warp_sum(_self(below_a, Float32(1.0), Float32(0.0)))

        # the window: entries lo .. lo + WC - 1 of the sorted run, and the one past it
        dw = []
        rw = []
        for q in range_constexpr(Q):
            e = lo_i + q * WARP + lane
            held = e < CAP
            e = cutlass.min(e, CAP - 1)
            dw.append(_self(held, sD[warp, e], Float32(FAR)))
            rw.append(_seli(held, sR[warp, e], Int32(-1)))
        e_last = lo_i + WC
        held = e_last < CAP
        e_last = cutlass.min(e_last, CAP - 1)
        d_last = _self(held, sD[warp, e_last], Float32(FAR))
        r_last = _seli(held, sR[warp, e_last], Int32(-1))
        over = cutlass.max(over, _seli((r_last >= 0) & (d_last < b), Int32(1), Int32(0)))

        # Newton-bisection from the bracket midpoint, as the stock path
        r_lo = a
        r_hi = b
        r = 0.5 * (a + b)
        dn = Float32(0.0)
        for it in range_constexpr(iterations + 1):
            sf = Float32(0.0)
            sdf = Float32(0.0)
            for q in range_constexpr(Q):
                valid = rw[q] >= 0
                scaled = (dw[q] - (r - width)) / width
                active = valid & (scaled > 0.0) & (scaled < 1.0)
                smaller = valid & (scaled <= 0.0)
                safe = cute.arch.fmin(
                    cute.arch.fmax(scaled, Float32(1e-6)), Float32(1.0 - 1e-6)
                )
                s = PI * safe
                sin_s = cute.math.sin(s)
                tanh_cot = cute.math.tanh(cute.math.cos(s) / sin_s)
                f1 = _self(smaller, Float32(1.0), Float32(0.0))
                sf = sf + _self(active, 0.5 * (1.0 + tanh_cot), f1)
                df = (0.5 * PI / width) * (1.0 - tanh_cot * tanh_cot) / (sin_s * sin_s)
                sdf = sdf + _self(active, df, Float32(0.0))
            x = r * inv_cutoff
            n = lo_f + _warp_sum(sf) + k * (x * x * x)
            dn = _warp_sum(sdf) + (3.0 * k * inv_cutoff) * (x * x)
            if const_expr(it < iterations):
                f = n - k
                below = f <= 0.0
                r_lo = _self(below, r, r_lo)
                r_hi = _self(below, r_hi, r)
                r_newton = r - f / cute.arch.fmax(dn, Float32(1e-6))
                inside = (r_newton >= r_lo) & (r_newton <= r_hi)
                r = _self(inside, r_newton, 0.5 * (r_lo + r_hi))

        # the implicit-function step: the residual with the forward's bump
        sb = Float32(0.0)
        for q in range_constexpr(Q):
            xq = (dw[q] - (r - width)) / width
            safe = cute.arch.fmin(cute.arch.fmax(xq, Float32(1e-6)), Float32(1.0 - 1e-6))
            bump = 0.5 * (1.0 + cute.math.tanh(1.0 / cute.math.tan(PI * safe)))
            bump = _self(xq <= 0.0, Float32(1.0), _self(xq >= 1.0, Float32(0.0), bump))
            sb = sb + bump
        x = r * inv_cutoff
        n_res = lo_f + _warp_sum(sb) + k * (x * x * x) - k
        dn_c = cute.arch.fmax(dn, Float32(1e-6))
        val = r - n_res / dn_c
        cut = cute.arch.fmin(cute.arch.fmax(val, Float32(cutoff / 16)), Float32(cutoff))
        clipped = (val < cutoff / 16) | (val > cutoff)
        if lane == 0:
            cutoffs[i] = cut
            root[i] = r
            slope[i] = _self(clipped, Float32(0.0), dn_c)
            flags[i] = _self(over > 0, Float32(1.0), Float32(0.0))
        for q in range_constexpr(Q):
            win[i, q * WARP + lane] = rw[q]


@cute.jit
def launch_forward(
    stream: cuda.CUstream,
    r_ij: cute.Tensor,
    pmask: cute.Tensor,
    starts: cute.Tensor,
    counts: cute.Tensor,
    rp: cute.Tensor,
    cutoffs: cute.Tensor,
    root: cute.Tensor,
    slope: cute.Tensor,
    win: cute.Tensor,
    flags: cute.Tensor,
    *,
    CAP: cutlass.Constexpr[int],
    WC: cutlass.Constexpr[int],
    KN: cutlass.Constexpr[int],
    NA: cutlass.Constexpr[int],
    k: cutlass.Constexpr[float],
    cutoff: cutlass.Constexpr[float],
    width: cutlass.Constexpr[float],
    iterations: cutlass.Constexpr[int],
):
    NW = WARPS_PER_CTA
    forward_kernel(
        r_ij,
        pmask,
        starts,
        counts,
        rp,
        cutoffs,
        root,
        slope,
        win,
        flags,
        CAP,
        WC,
        KN,
        NW,
        NA,
        k,
        cutoff,
        width,
        iterations,
    ).launch(grid=[(NA + NW - 1) // NW, 1, 1], block=[NW * WARP, 1, 1], stream=stream)


# -- backward --


@cute.kernel
def backward_kernel(
    r_ij: cute.Tensor,
    starts: cute.Tensor,
    counts: cute.Tensor,
    root: cute.Tensor,
    slope: cute.Tensor,
    win: cute.Tensor,
    dcut: cute.Tensor,
    d_r: cute.Tensor,
    WC: cutlass.Constexpr[int],
    NW: cutlass.Constexpr[int],
    NA: cutlass.Constexpr[int],
    width: cutlass.Constexpr[float],
):
    Q = WC // WARP
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * NW + warp
    if i < NA:
        start = starts[i]
        count = counts[i]
        for c in cutlass.range(lane, count, WARP):
            d_r[start + c] = Float32(0.0)
        cute.arch.sync_warp()
        dn = slope[i]
        r = root[i]
        gn = -dcut[i] / cute.arch.fmax(dn, Float32(1e-30))
        gn = _self(dn > 0.0, gn, Float32(0.0))
        for q in range_constexpr(Q):
            rawq = win[i, q * WARP + lane]
            dq = r_ij[cutlass.max(rawq, Int32(0))]
            x = (dq - (r - width)) / width
            # d bump / d d as JAX takes it through cutoff_bump: zero outside the clip
            live = (x > 1e-6) & (x < 1.0 - 1e-6)
            xs = cute.arch.fmin(cute.arch.fmax(x, Float32(1e-6)), Float32(1.0 - 1e-6))
            t = cute.math.tan(PI * xs)
            th = cute.math.tanh(1.0 / t)
            g = gn * ((0.5 * (1.0 - th * th)) * (-(1.0 + t * t) / (t * t)) * (PI / width))
            g = _self(live, g, Float32(0.0))
            if rawq >= 0:
                d_r[rawq] = g


@cute.jit
def launch_backward(
    stream: cuda.CUstream,
    r_ij: cute.Tensor,
    starts: cute.Tensor,
    counts: cute.Tensor,
    root: cute.Tensor,
    slope: cute.Tensor,
    win: cute.Tensor,
    dcut: cute.Tensor,
    d_r: cute.Tensor,
    *,
    WC: cutlass.Constexpr[int],
    NA: cutlass.Constexpr[int],
    width: cutlass.Constexpr[float],
):
    NW = WARPS_PER_CTA
    backward_kernel(
        r_ij, starts, counts, root, slope, win, dcut, d_r, WC, NW, NA, width
    ).launch(grid=[(NA + NW - 1) // NW, 1, 1], block=[NW * WARP, 1, 1], stream=stream)
