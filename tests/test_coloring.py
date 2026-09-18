"""Tests for the lifted star coloring (`petjax.sparse.coloring`).

Pattern level only: no model, no AD. Decompression is a gather over
`(pattern, colors, star set)`, so pushing a random symmetric matrix supported on
the pattern through `asdex.decompress_data` exercises exactly the production
code path, and every correctly decompressed entry is a one-term sum, so
agreement is bit-exact rather than approximate. Random radius graphs span fills
from ~0.1 to ~0.8 across hop counts 1-4.
"""

import numpy as np
import jax.numpy as jnp

import pytest

asdex = pytest.importorskip("asdex")
sp = pytest.importorskip("scipy.sparse")

from scipy.spatial import cKDTree

from petjax.sparse import hessian_coloring, lift_atom_coloring, sparsity_patterns

# (seed, atoms, cutoff, hops), chosen to span a wide range of fill
GRAPHS = [
    (0, 30, 2.0, 1),
    (1, 40, 2.5, 2),
    (2, 50, 1.8, 2),
    (3, 25, 3.0, 1),
    (4, 60, 2.2, 2),
    (5, 35, 2.6, 3),
    (6, 45, 2.0, 3),
    (7, 55, 1.6, 4),
    (8, 30, 3.2, 2),
]


def radius_graph(seed, n, cutoff):
    """A random point cloud's radius graph as a `(centers, others, n)` pair list."""
    rng = np.random.default_rng(seed)
    pairs = np.array(
        cKDTree(rng.normal(size=(n, 3)) * 2.0).query_pairs(cutoff, output_type="ndarray")
    )
    centers = np.concatenate([pairs[:, 0], pairs[:, 1]])
    others = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return centers, others, n


def coord_pattern(graph, hops):
    return sparsity_patterns(*graph, hops)[1]


def check_roundtrip(coloring, seed=0):
    """Compress a random symmetric matrix on the pattern, decompress, compare.

    Returns `(mismatches, nnz)`; an invalid hub folds a second entry into the
    sum and shows up as a mismatch.
    """
    pattern = coloring.sparsity
    rows = np.asarray(pattern.rows).astype(np.int64)
    cols = np.asarray(pattern.cols).astype(np.int64)
    dim = pattern.n

    rng = np.random.default_rng(seed)
    M = sp.csr_matrix(
        (rng.standard_normal(len(rows), dtype=np.float32), (rows, cols)), shape=(dim, dim)
    )
    H = ((M + M.T) * np.float32(0.5)).tocsr()

    seeds = sp.csr_matrix(
        (np.ones(dim, np.float32), (np.arange(dim), np.asarray(coloring.colors))),
        shape=(dim, coloring.num_colors),
    )
    # B[c, r] = sum_{k : color[k] == c} H[r, k]
    compressed = np.asarray((H @ seeds).todense()).T

    got = np.asarray(asdex.decompress_data(jnp.asarray(compressed), coloring))
    want = np.asarray(H[rows, cols]).ravel()
    return int((got != want).sum()), len(want)


def greedy_distance_1(pattern):
    """A coloring that is distance-1 valid but generally not a star coloring:
    what a naive Jacobian-style coloring would give. The negative control."""
    n = pattern.n
    graph = sp.csr_matrix(
        (np.ones(pattern.nnz, np.int8), (pattern.rows, pattern.cols)), shape=(n, n)
    )
    colors = np.full(n, -1, np.int32)
    for v in range(n):
        neighbours = graph.indices[graph.indptr[v] : graph.indptr[v + 1]]
        used = {int(c) for c in colors[neighbours] if c >= 0}
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    return colors


def atom_colors_of(atom):
    return asdex.color_symmetric(atom)[0]


# -- correctness --


@pytest.mark.parametrize("seed,n,cutoff,hops", GRAPHS)
def test_roundtrip_exact(seed, n, cutoff, hops):
    coloring = hessian_coloring(*radius_graph(seed, n, cutoff), hops)
    bad, total = check_roundtrip(coloring)
    assert bad == 0 and total == coloring.sparsity.nnz


@pytest.mark.parametrize("seed,n,cutoff,hops", GRAPHS)
def test_matches_asdex(seed, n, cutoff, hops):
    """Same color count as asdex's coordinate-level star coloring, and the same
    assignment. The elementwise part is a canary for an upstream change in
    vertex ordering, not a correctness requirement of the lift."""
    graph = radius_graph(seed, n, cutoff)
    ours = hessian_coloring(*graph, hops)
    ref = asdex.hessian_coloring_from_sparsity(coord_pattern(graph, hops))
    assert ours.num_colors == ref.num_colors
    assert np.array_equal(np.asarray(ours.colors), np.asarray(ref.colors))


def test_zero_hops():
    """At `hops=0` every edge is inside one atom's 3x3 block."""
    coloring = hessian_coloring(*radius_graph(1, 40, 2.5), hops=0)
    assert coloring.num_colors == 3
    assert check_roundtrip(coloring)[0] == 0


def test_save_load_roundtrip(tmp_path):
    """A coloring persisted with asdex reloads onto the same star set. asdex
    persists only `star` and `hub` and rebuilds the edge arrays in its own
    canonical order, which the lift's numbering has to match. pet-jax does not
    persist colorings itself; this pins that a caller who does gets the same
    star set back."""
    coloring = hessian_coloring(*radius_graph(4, 60, 2.2), hops=2)
    path = tmp_path / "coloring.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    assert loaded.num_colors == coloring.num_colors
    assert loaded.mode == coloring.mode and loaded.symmetric
    assert np.array_equal(np.asarray(loaded.colors), np.asarray(coloring.colors))
    for field in ("star", "hub", "edge_lo", "edge_hi", "edge_pos"):
        assert np.array_equal(
            getattr(loaded.star_set, field), getattr(coloring.star_set, field)
        ), field
    assert check_roundtrip(loaded)[0] == 0


# -- guards --


@pytest.fixture(scope="module")
def pair():
    return sparsity_patterns(*radius_graph(1, 40, 2.5), hops=2)


def test_rejects_distance_1_coloring(pair):
    atom, coord = pair
    with pytest.raises(asdex.InvalidColoringError):
        lift_atom_coloring(greedy_distance_1(atom), atom, coord)


def test_rejects_asymmetric_atom_pattern(pair):
    atom, coord = pair
    colors = atom_colors_of(atom)
    # Transposing one entry breaks symmetry while keeping nnz, so this reaches the
    # symmetry guard instead of tripping the cheaper block-expansion nnz check.
    rows, cols = np.asarray(atom.rows).copy(), np.asarray(atom.cols).copy()
    k = int(np.flatnonzero(rows < cols)[0])
    rows[k], cols[k] = cols[k], rows[k]
    skewed = asdex.SparsityPattern(rows=rows, cols=cols, shape=atom.shape)
    with pytest.raises(ValueError, match="symmetric"):
        lift_atom_coloring(colors, skewed, coord)


def test_rejects_mismatched_pattern_pair(pair):
    atom, _coord = pair
    _, other_coord = sparsity_patterns(*radius_graph(4, 60, 2.2), hops=2)
    with pytest.raises(ValueError, match="coord pattern"):
        lift_atom_coloring(atom_colors_of(atom), atom, other_coord)


def test_rejects_same_nnz_mismatch(pair):
    """Same nnz, different pairs: caught by the per-entry atom-pair lookup."""
    atom, coord = pair
    rows, cols = np.asarray(atom.rows).copy(), np.asarray(atom.cols).copy()
    # Move one off-diagonal pair (and its transpose) to an absent pair.
    present = set(zip(rows.tolist(), cols.tolist()))
    n = atom.n
    absent = next(
        (i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in present
    )
    k = int(np.flatnonzero(rows < cols)[0])
    kt = int(np.flatnonzero((rows == cols[k]) & (cols == rows[k]))[0])
    rows[k], cols[k] = absent
    rows[kt], cols[kt] = absent[1], absent[0]
    moved = asdex.SparsityPattern(rows=rows, cols=cols, shape=atom.shape)
    with pytest.raises(ValueError, match="does not"):
        lift_atom_coloring(atom_colors_of(moved), moved, coord)


def test_rejects_bad_colors(pair):
    atom, coord = pair
    colors = atom_colors_of(atom)
    with pytest.raises(ValueError, match="shape"):
        lift_atom_coloring(colors[:-1], atom, coord)
    negative = colors.copy()
    negative[0] = -1
    with pytest.raises(ValueError, match="neutral"):
        lift_atom_coloring(negative, atom, coord)
    gapped = colors + 1  # color 0 unused
    with pytest.raises(ValueError, match="dense"):
        lift_atom_coloring(gapped, atom, coord)


def test_rejects_bad_mode(pair):
    atom, coord = pair
    with pytest.raises(ValueError, match="mode"):
        lift_atom_coloring(atom_colors_of(atom), atom, coord, mode="fwd")
