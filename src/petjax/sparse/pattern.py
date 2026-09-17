"""Pair list → asdex Hessian sparsity pattern by graph reachability.

Two atoms couple in the positions-Hessian iff a chain of neighbour edges
connects them within the model's reach, so the pattern is ``(I + A)^hops``,
with ``A`` the 1-hop adjacency from the ``(centers, others)`` pair list. The
exact model-dependent ``hops`` is the Hessian hop count (``2L + 1`` for PET;
see ``UPETCalculator.hessian``); in practice one often truncates below it.

Padding needs no masking: every padding pair in a ``to_structure`` dict points
at the single dummy atom slot, so padding edges are self-loops there and never
reach a real atom (``tests/test_sparsity.py::test_padding_isolated``).

The ``N x N`` atom pattern is 3x3-block expanded to the ``3N x 3N`` coordinate
space; the atom pattern is what ``sparse.coloring`` colors.

This module has no model vocabulary on purpose: pair list in, asdex objects
out, so that it can move into ``asdex`` later.
"""

import numpy as np
from jax import ShapeDtypeStruct

import asdex
from scipy import sparse as sp

__all__ = ["sparsity_patterns"]


# -- public API --


def sparsity_patterns(centers, others, n, hops):
    """Atom-level and coordinate-level Hessian sparsity patterns of a pair list.

    Args:
        centers, others: The (directed) pair list; both directions of every
            edge must be present for the pattern to be symmetric.
        n: Number of atom slots the indices refer to (padding included).
        hops: Reachability radius in graph hops, non-negative.

    Returns:
        ``(atom, coord)``: the ``(I + A)^hops`` reachability pattern over ``n``
        atoms as an ``(n, n)`` ``asdex.SparsityPattern``, and its 3x3-block
        expansion into ``(3n, 3n)`` coordinate space with the positions
        ``(n, 3)`` aval attached so asdex differentiates native ``(n, 3)``
        arrays. ``coord`` has exactly ``9 * atom.nnz`` entries.
    """
    centers = np.asarray(centers)
    others = np.asarray(others)
    pairs = _reachable_pairs(centers, others, int(n), hops)
    return _atom_pairs_to_sparsity(pairs, n), _pairs_to_sparsity(pairs, n)


# -- reachability and block expansion --


def _reachable_pairs(centers, others, n, hops):
    """Atom pairs ``(i, j)`` within ``hops`` of each other, diagonal included.

    Returns a unique ``(n_pairs, 2)`` int64 array sorted by ``(i, j)``. The
    diagonal is always present so every atom has a self-coupling entry at any
    ``hops >= 0``.
    """
    if hops < 0:
        raise ValueError(f"hops must be non-negative, got {hops}")

    data = np.ones(len(centers), dtype=bool)
    A = sp.csr_matrix((data, (centers, others)), shape=(n, n))

    base = sp.eye(n, dtype=bool, format="csr") + A
    pattern = sp.eye(n, dtype=bool, format="csr")
    for _ in range(hops):
        pattern = pattern @ base
        pattern.data[:] = True  # boolean reachability; drop accumulated counts

    coo = pattern.tocoo()
    diag = np.arange(n)
    pairs = np.column_stack(
        [np.concatenate([coo.row, diag]), np.concatenate([coo.col, diag])]
    )
    return np.unique(pairs.astype(np.int64), axis=0)


def _atom_pairs_to_sparsity(pairs, n):
    """Wrap atom-level pairs as an ``(n, n)`` ``asdex.SparsityPattern``.

    No input aval: this pattern is only ever colored, never differentiated on.
    """
    return asdex.SparsityPattern(
        rows=pairs[:, 0].astype(np.int32),
        cols=pairs[:, 1].astype(np.int32),
        shape=(n, n),
    )


def _pairs_to_sparsity(pairs, n):
    """Expand atom-level ``(i, j)`` pairs to 3x3 Cartesian blocks and wrap as an
    ``asdex.SparsityPattern`` of shape ``(3n, 3n)``. The positions ``(n, 3)``
    aval is attached (``3*atom + cart`` is the row-major flatten of ``(n, 3)``)
    so asdex differentiates native ``(n, 3)`` positions and returns an
    ``(n, 3, n, 3)`` Hessian. Entries are emitted block by block, not lexsorted;
    a BCOO Hessian and a coloring persisted with asdex are aligned to this
    order, so changing it changes what a caller who stores either gets back."""
    i, j = pairs[:, 0], pairs[:, 1]
    n_blocks = len(i)
    row_off = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    col_off = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    rows = np.repeat(i * 3, 9) + np.tile(row_off, n_blocks)
    cols = np.repeat(j * 3, 9) + np.tile(col_off, n_blocks)

    return asdex.SparsityPattern.from_coo(
        rows=rows,
        cols=cols,
        shape=(3 * n, 3 * n),
        # Only the (n, 3) shape is load-bearing; asdex validates it and returns
        # an (n, 3, n, 3) Hessian. The aval dtype is nominal — the real dtype
        # follows the positions passed at call time.
        input_avals=(ShapeDtypeStruct((n, 3), np.float64),),
    )
