"""Star coloring of the Hessian pattern, done on the atom graph and lifted.

The coordinate pattern is the atom pattern with every atom replaced by a dense
3x3 block. Since there is no sparsity in these blocks, we don't need to color
them: we can just color the atom-atom connectivity and "lift" the result into
the full Hessian later. This avoids a lot of redundant compute (1/9 edges!).

Besides the colors, asdex needs to know for every off-diagonal entry (i, j)
which of the two coordinates is the *hub*, because the entry is read from the
hub's row of the compressed Hessian: with h the hub and s the other coordinate,
H[i, j] = B[color[h], s], where B[c, s] = sum of H[s, k] over all k with color c.
That read is exact iff h is the only neighbour of s with color color[h]. So a
valid hub can be found by counting, per atom pair, how many neighbours of one
atom carry the other atom's color. All three coordinates of an atom have the
same neighbours and distinct colors, so one count per atom pair settles all nine
entries of the block. asdex expects the hubs in a ``StarSet``; we hand it the
simplest one, every edge its own star with the hub filled in.

The same count doubles as a check of the coloring. Decompression is exact iff
every edge has a valid hub and every atom is the only neighbour of itself with
its own color (for the diagonal), so a coloring that is merely distance-1 valid,
which would decompress a Hessian wrongly, is rejected here.

Like ``sparse.pattern``, this module has no model vocabulary: it is written to
move into ``asdex`` later.
"""

import numpy as np

import asdex

from typing import get_args

from .pattern import sparsity_patterns

__all__ = ["hessian_coloring", "lift_atom_coloring"]


# -- public API --


def hessian_coloring(centers, others, n, hops, *, mode="fwd_over_rev"):
    """Colored ``(3n, 3n)`` Hessian pattern of a pair list.

    The replacement for ``asdex.hessian_coloring_from_sparsity`` on the
    coordinate pattern: same colors, found on the atom graph.

    Args:
        centers, others, n, hops: See ``sparsity_patterns``.
        mode: asdex Hessian AD mode.
    """
    atom, coord = sparsity_patterns(centers, others, n, hops)
    colors, _num_colors, _star_set = asdex.color_symmetric(atom)
    return lift_atom_coloring(colors, atom, coord, mode=mode)


def lift_atom_coloring(atom_colors, atom_pattern, coord_pattern, *, mode="fwd_over_rev"):
    """Build the coordinate-level ``ColoredPattern`` from an atom coloring.

    The coloring is a black box: any dense, non-negative ``int[n]`` array is
    accepted and validated against the pattern, so nothing here depends on
    asdex's coloring algorithm.

    Args:
        atom_colors: Color per atom, dense in ``[0, num_atom_colors)``.
        atom_pattern: The ``(n, n)`` pattern the colors belong to.
        coord_pattern: Its 3x3 block expansion, ``(3n, 3n)``.
        mode: asdex Hessian AD mode.

    Raises:
        ValueError: If the two patterns are not a block-expansion pair, the
            colors are not dense and non-negative, or ``mode`` is not a
            Hessian mode.
        asdex.InvalidColoringError: If the colors are not a star coloring of
            ``atom_pattern``, i.e. some entry could not be decompressed.
    """
    if mode not in get_args(asdex.HessianMode):
        raise ValueError(f"mode must be one of {get_args(asdex.HessianMode)}, got {mode!r}")
    n = atom_pattern.n
    colors = np.asarray(atom_colors, dtype=np.int64)
    _validate(colors, atom_pattern, coord_pattern, n)

    rows, cols = _lexsorted(atom_pattern)
    hub_is_hi = _atom_hub_choice(rows, cols, colors)
    star_set = _coord_star_set(coord_pattern, rows, cols, hub_is_hi, n)

    coord_colors = (3 * colors[:, None] + np.arange(3)).ravel().astype(np.int32)
    return asdex.ColoredPattern(
        sparsity=coord_pattern,
        colors=coord_colors,
        num_colors=3 * (int(colors.max()) + 1),
        symmetric=True,
        mode=mode,
        star_set=star_set,
    )


# -- hub derivation --


def _atom_hub_choice(rows, cols, colors):
    """Per undirected atom pair ``i < j`` (lexsorted), whether ``j`` is the hub.

    ``mult[k]`` counts the neighbours of ``rows[k]`` sharing ``colors[cols[k]]``,
    so ``mult[k] == 1`` says "``cols[k]`` is a valid hub for the edge,
    ``rows[k]`` the spoke". A star coloring guarantees at least one endpoint
    qualifies; anything less is rejected here rather than downstream.
    """
    key = rows * (colors.max() + 1) + colors[cols]
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    mult = counts[inverse]

    # Diagonal entries read their own color row, so they need distance-1: an atom
    # must be the only neighbour of itself carrying its own color.
    diagonal = rows == cols
    if not (mult[diagonal] == 1).all():
        bad = int(rows[diagonal][mult[diagonal] != 1][0])
        raise asdex.InvalidColoringError(
            f"atom {bad} shares its color with a neighbour (distance-1 violated)"
        )

    # rows/cols are lexsorted by (row, col) and the pattern is symmetric, so
    # sorting by (col, row) lands the transpose of entry k at position k. An
    # asymmetric pattern would misalign the two silently, hence the guard. asdex
    # does not check either: it symmetrizes triangular input, which we cannot
    # do without also changing the coordinate pattern.
    transpose = np.lexsort((rows, cols))
    if not (cols[transpose] == rows).all():
        raise ValueError("atom pattern is not symmetric")
    valid_hi = mult  # entry (i, j): is j a valid hub?
    valid_lo = mult[transpose]  # entry (i, j): is i a valid hub?

    upper = np.flatnonzero(rows < cols)
    hi_ok = valid_hi[upper] == 1
    lo_ok = valid_lo[upper] == 1
    if not (hi_ok | lo_ok).all():
        k = int(upper[~(hi_ok | lo_ok)][0])
        raise asdex.InvalidColoringError(
            f"edge ({rows[k]}, {cols[k]}) has no valid hub: "
            "the colors are not a star coloring of this pattern"
        )
    return hi_ok


def _coord_star_set(coord_pattern, rows, cols, hub_is_hi, n):
    """Lift the per-atom-pair hub choice onto every coordinate edge."""
    crows = np.asarray(coord_pattern.rows).astype(np.int64)
    ccols = np.asarray(coord_pattern.cols).astype(np.int64)

    upper = crows < ccols
    edge_lo, edge_hi = crows[upper], ccols[upper]
    # StarSet lookups binary-search `edge_lo * n + edge_hi`, so the arrays must be
    # lexsorted. The block expansion emits rows/cols block by block, not sorted.
    order = np.lexsort((edge_hi, edge_lo))
    edge_lo, edge_hi = edge_lo[order], edge_hi[order]
    # Strictly increasing keys mean no duplicate entries. Without duplicates the
    # `arange` edge numbering below equals asdex's own (consecutive in lexsorted
    # order), so a coloring persisted with asdex's save/load comes back onto
    # the same star set.
    edge_keys = edge_lo * (3 * n) + edge_hi
    if len(edge_keys) > 1 and not (np.diff(edge_keys) > 0).all():
        raise ValueError("coord pattern has duplicate entries")

    # Hub validity depends only on the atom pair, so look it up there. Keys over
    # the lexsorted upper-triangle atom pairs are already ascending.
    upper_pairs = np.flatnonzero(rows < cols)
    keys = rows[upper_pairs] * n + cols[upper_pairs]
    atom_lo, atom_hi = edge_lo // 3, edge_hi // 3
    inter = atom_lo != atom_hi
    wanted = atom_lo[inter] * n + atom_hi[inter]
    idx = np.searchsorted(keys, wanted)
    hit = (idx < len(keys)) & (keys[np.minimum(idx, len(keys) - 1)] == wanted)
    if not hit.all():
        k = int(np.flatnonzero(~hit)[0])
        raise ValueError(
            f"coord entry couples atoms ({atom_lo[inter][k]}, {atom_hi[inter][k]}) "
            "that the atom pattern does not"
        )
    # Edges inside one atom's 3x3 block: both endpoints are valid hubs, since
    # distance-1 makes the atom the only neighbour of itself in its color.
    choose_hi = np.ones(len(edge_lo), dtype=bool)
    choose_hi[inter] = hub_is_hi[idx]

    num_edges = len(edge_lo)
    return asdex.StarSet(
        star=np.arange(num_edges, dtype=np.int32),
        hub=np.where(choose_hi, edge_hi, edge_lo).astype(np.int32),
        edge_lo=edge_lo.astype(np.int32),
        edge_hi=edge_hi.astype(np.int32),
        edge_pos=np.arange(num_edges, dtype=np.int32),
    )


# -- guards --


def _validate(colors, atom_pattern, coord_pattern, n):
    """Reject inputs the hub derivation would silently misread."""
    if atom_pattern.m != n:
        raise ValueError(f"atom pattern must be square, got {atom_pattern.shape}")
    if coord_pattern.shape != (3 * n, 3 * n):
        raise ValueError(
            f"coord pattern must be ({3 * n}, {3 * n}) for {n} atoms, "
            f"got {coord_pattern.shape}"
        )
    # The lift assumes coord is *exactly* the 3x3 block expansion of atom. With
    # both diagonals forced and entries unique, that pins the nnz; the per-entry
    # atom-pair lookup in `_coord_star_set` catches a same-nnz mismatch.
    if coord_pattern.nnz != 9 * atom_pattern.nnz:
        raise ValueError(
            f"coord pattern has {coord_pattern.nnz} nnz, expected "
            f"{9 * atom_pattern.nnz} (9x the atom pattern's {atom_pattern.nnz})"
        )
    if colors.shape != (n,):
        raise ValueError(f"colors must have shape ({n},), got {colors.shape}")
    if colors.min() < 0:
        # Neutral (-1) colors cannot occur here: the atom pattern carries a full
        # diagonal, so postprocessing keeps every color used.
        raise ValueError("neutral (-1) colors are not supported")
    if (np.bincount(colors) == 0).any():
        # A gap in the color range would be an empty seed, a wasted HVP.
        raise ValueError("colors must be dense in [0, max]")


def _lexsorted(pattern):
    """Pattern entries as ``(rows, cols)`` sorted by ``(row, col)``.

    The transpose trick and the atom-pair binary search both depend on this order.
    """
    rows = np.asarray(pattern.rows).astype(np.int64)
    cols = np.asarray(pattern.cols).astype(np.int64)
    order = np.lexsort((cols, rows))
    return rows[order], cols[order]
