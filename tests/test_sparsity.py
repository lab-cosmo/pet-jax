"""Tests for the graph reachability sparsity pattern (`petjax.sparse.pattern`).

The pattern is `(I + A)^hops` over the `(centers, others)` pair list, 3x3-block
expanded into `3N x 3N` coordinate space. These check the reachability logic at
the atom level and the invariant that pet-jax's padding (every padding pair a
self-loop on a single dummy slot) never couples to real atoms.
"""

import numpy as np

import pytest

pytest.importorskip("asdex")

from petjax.sparse import sparsity_patterns


def atom_pairs(pattern):
    """Atom-level `(i, j)` pairs present in a 3x3-block coordinate pattern."""
    rows, cols = np.asarray(pattern.rows), np.asarray(pattern.cols)
    return set(zip((rows // 3).tolist(), (cols // 3).tolist()))


def coord_pattern(centers, others, n, hops):
    return sparsity_patterns(centers, others, n, hops)[1]


def undirected_edges(centers, others):
    """Symmetrise a pair list so reachability is over the undirected graph."""
    c = np.concatenate([centers, others])
    o = np.concatenate([others, centers])
    return c, o


# -- reachability --


def test_diagonal_only_at_zero_hops():
    # 0 -- 1 -- 2, but hops=0 sees no edges
    centers, others = undirected_edges(np.array([0, 1]), np.array([1, 2]))
    pairs = atom_pairs(coord_pattern(centers, others, n=3, hops=0))
    assert pairs == {(0, 0), (1, 1), (2, 2)}


def test_linear_chain_one_hop():
    # 0 -- 1 -- 2 -- 3 ; one hop couples direct neighbours only
    centers, others = undirected_edges(np.array([0, 1, 2]), np.array([1, 2, 3]))
    pairs = atom_pairs(coord_pattern(centers, others, n=4, hops=1))
    expected = {(i, i) for i in range(4)}
    expected |= {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    assert pairs == expected
    assert (0, 2) not in pairs  # two apart, not yet reachable


def test_linear_chain_reaches_full_density():
    # 0 -- 1 -- 2 -- 3 ; diameter 3, so hops=3 couples everything
    centers, others = undirected_edges(np.array([0, 1, 2]), np.array([1, 2, 3]))
    pairs = atom_pairs(coord_pattern(centers, others, n=4, hops=3))
    assert pairs == {(i, j) for i in range(4) for j in range(4)}


def test_disconnected_clusters_never_couple():
    # {0,1} and {2,3} are separate; no hop count bridges them
    centers, others = undirected_edges(np.array([0, 2]), np.array([1, 3]))
    pairs = atom_pairs(coord_pattern(centers, others, n=4, hops=5))
    assert (0, 2) not in pairs and (1, 3) not in pairs
    assert (0, 1) in pairs and (2, 3) in pairs


def test_star_graph_couples_leaves_at_two_hops():
    # hub 0 with leaves 1,2,3 ; leaves couple to each other only via 2 hops
    centers, others = undirected_edges(np.array([0, 0, 0]), np.array([1, 2, 3]))
    one = atom_pairs(coord_pattern(centers, others, n=4, hops=1))
    two = atom_pairs(coord_pattern(centers, others, n=4, hops=2))
    assert (1, 2) not in one
    assert (1, 2) in two and (2, 3) in two and (1, 3) in two


def test_symmetric_for_undirected_graph():
    centers, others = undirected_edges(np.array([0, 1, 2]), np.array([1, 2, 3]))
    pairs = atom_pairs(coord_pattern(centers, others, n=4, hops=2))
    assert all((j, i) in pairs for (i, j) in pairs)


def test_monotonic_in_hops():
    centers, others = undirected_edges(np.array([0, 1, 2]), np.array([1, 2, 3]))
    prev = atom_pairs(coord_pattern(centers, others, n=4, hops=0))
    for hops in range(1, 5):
        cur = atom_pairs(coord_pattern(centers, others, n=4, hops=hops))
        assert prev <= cur
        prev = cur


def test_negative_hops_rejected():
    with pytest.raises(ValueError):
        sparsity_patterns(np.array([0]), np.array([1]), n=2, hops=-1)


# -- padding invariant --


def _padded_chain():
    """0 -- 1 -- 2 real, atoms 3,4,5 padding as self-loops on dummy slot 3."""
    real_c, real_o = undirected_edges(np.array([0, 1]), np.array([1, 2]))
    centers = np.concatenate([real_c, np.full(4, 3)])
    others = np.concatenate([real_o, np.full(4, 3)])
    return centers, others, 6


def test_padding_isolated():
    """pet-jax pads pairs as self-loops on the dummy slot; padding atoms must
    couple only to themselves, at any hop count, with no spurious real coupling."""
    centers, others, n = _padded_chain()
    pairs = atom_pairs(coord_pattern(centers, others, n, hops=4))
    for p in (3, 4, 5):
        coupled = {j for (i, j) in pairs if i == p} | {i for (i, j) in pairs if j == p}
        assert coupled == {p}, f"padding atom {p} coupled to {coupled}"


# -- atom level --


def test_atom_pattern_is_the_block_structure():
    centers, others, n = _padded_chain()
    for hops in range(3):
        atom, coord = sparsity_patterns(centers, others, n, hops)
        assert atom.m == atom.n == n
        assert coord.m == coord.n == 3 * n
        assert coord.nnz == 9 * atom.nnz
        assert atom_pairs(coord) == set(zip(atom.rows.tolist(), atom.cols.tolist()))


def test_coord_pattern_carries_positions_aval():
    """asdex differentiates native `(n, 3)` positions via the attached aval."""
    centers, others, n = _padded_chain()
    _, coord = sparsity_patterns(centers, others, n, hops=1)
    (aval,) = coord.input_avals
    assert aval.shape == (n, 3)


def test_atom_pattern_padding_isolated():
    centers, others, n = _padded_chain()
    atom, _ = sparsity_patterns(centers, others, n, hops=4)
    pairs = set(zip(atom.rows.tolist(), atom.cols.tolist()))
    for p in (3, 4, 5):
        coupled = {j for (i, j) in pairs if i == p} | {i for (i, j) in pairs if j == p}
        assert coupled == {p}
