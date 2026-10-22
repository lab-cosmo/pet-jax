"""Device-side flat-NL layout helpers against the host builders in structure.py."""

import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.build import bulk, molecule

from petjax import center_order, corresponding_edges
from petjax.structure import lattice_shifts, neighbor_list, to_structure


@pytest.fixture(params=["primitive", "supercell", "molecule"])
def structure(request):
    # The primitive cell at a cutoff past half its perpendicular lengths has
    # several images per pair and atoms that neighbour themselves.
    if request.param == "primitive":
        atoms, cutoff = bulk("NaCl", crystalstructure="rocksalt", a=5.64), 6.0
    elif request.param == "supercell":
        atoms, cutoff = bulk("Cu", cubic=True) * [2, 2, 2], 4.0
    else:
        atoms, cutoff = molecule("CH3CH2OH"), 3.0
    return jax.tree.map(np.asarray, to_structure(atoms, cutoff=cutoff, skin=0.0))


def _fields(structure):
    return tuple(structure[k] for k in ("centers", "others", "cell_shifts", "pair_mask"))


def test_corresponding_edges_matches_host(structure):
    """Bit-equal to the host reverse map on the host layout, padding included."""
    got = jax.jit(corresponding_edges)(*_fields(structure))
    assert np.array_equal(got, structure["reverse"])


def test_corresponding_edges_any_order(structure):
    """Independent of the pair order: on a shuffled list it is the host map
    carried through the shuffle."""
    perm = np.random.default_rng(0).permutation(structure["centers"].shape[0])
    inverse = np.argsort(perm)
    got = corresponding_edges(*(x[perm] for x in _fields(structure)))
    assert np.array_equal(got, inverse[structure["reverse"][perm]])


def test_center_order_then_corresponding_edges(structure):
    """Sorting a shuffled list by center gives the layout truncate_edges needs,
    and the reverse map on it pairs (i, j, S) with (j, i, -S) as an involution
    that leaves padded pairs alone."""
    perm = np.random.default_rng(1).permutation(structure["centers"].shape[0])
    centers, others, shifts, mask = (x[perm] for x in _fields(structure))
    order = np.asarray(jax.jit(center_order)(centers, mask))
    centers, others, shifts, mask = (x[order] for x in (centers, others, shifts, mask))

    n_real = int(mask.sum())
    assert mask[:n_real].all() and not mask[n_real:].any()
    assert np.all(np.diff(centers[:n_real]) >= 0)

    rev = np.asarray(corresponding_edges(centers, others, shifts, mask))
    real = np.arange(n_real)
    assert np.array_equal(rev[rev], np.arange(rev.shape[0]))
    assert np.array_equal(centers[rev[real]], others[real])
    assert np.array_equal(others[rev[real]], centers[real])
    assert np.array_equal(shifts[rev[real]], -shifts[real])
    assert np.array_equal(rev[n_real:], np.arange(n_real, rev.shape[0]))


def test_corresponding_edges_int32_inputs(structure):
    """No x64 needed: int32 indices and float32 shifts in, int32 out."""
    centers, others, shifts, mask = _fields(structure)
    got = corresponding_edges(
        centers.astype(np.int32), others.astype(np.int32), shifts.astype(np.float32), mask
    )
    assert got.dtype == jnp.int32
    assert np.array_equal(got, structure["reverse"])


# -- device-built raw list against the host list --


def _device_list(atoms, cutoff, capacity):
    host = to_structure(atoms, cutoff=cutoff, skin=0.0)
    shifts = lattice_shifts(host["cell"], atoms.pbc, cutoff)
    nl, overflow = jax.jit(neighbor_list, static_argnums=6)(
        host["positions"],
        host["cell"],
        host["atom_mask"],
        atoms.pbc,
        shifts,
        cutoff,
        capacity,
    )
    return host, jax.tree.map(np.asarray, nl), bool(overflow)


def _pair_set(nl):
    mask = nl["pair_mask"]
    shifts = np.rint(nl["cell_shifts"][mask]).astype(int)
    return set(zip(nl["centers"][mask], nl["others"][mask], *shifts.T))


def _device_cases(mini_xyz):
    from ase.io import read

    yield from ((read(str(mini_xyz), index=i), 4.5) for i in range(12))
    # Past half the perpendicular lengths: several images per pair, self-images.
    yield bulk("NaCl", crystalstructure="rocksalt", a=5.64), 6.0
    yield bulk("Si", cubic=False), 7.5  # skewed primitive cell
    slab = bulk("Cu", cubic=True) * [2, 2, 1]
    slab.pbc = [True, True, False]
    slab.positions += np.random.default_rng(0).normal(scale=0.3, size=slab.positions.shape)
    yield slab, 5.0  # mixed pbc, unwrapped positions


def test_neighbor_list_matches_host(mini_xyz):
    """The device list holds the host list's pair set in its layout: sorted by
    center, padding last on the dummy atom, the last slot empty, and a reverse
    map that pairs (i, j, S) with (j, i, -S) as an involution."""
    for atoms, cutoff in _device_cases(mini_xyz):
        n_pairs = int(to_structure(atoms, cutoff=cutoff, skin=0.0)["pair_mask"].sum())
        host, nl, overflow = _device_list(atoms, cutoff, n_pairs + 7)
        assert not overflow
        assert _pair_set(nl) == _pair_set(host)

        mask = nl["pair_mask"]
        n_real = int(mask.sum())
        assert n_real == n_pairs and mask[:n_real].all() and not mask[n_real:].any()
        assert np.all(np.diff(nl["centers"][:n_real]) >= 0)
        dummy = host["positions"].shape[0] - 1
        assert (nl["centers"][n_real:] == dummy).all() and (
            nl["others"][n_real:] == dummy
        ).all()
        assert not nl["cell_shifts"][n_real:].any()

        rev = nl["reverse"]
        real = np.arange(n_real)
        assert np.array_equal(rev[rev], np.arange(rev.shape[0]))
        assert np.array_equal(nl["centers"][rev[real]], nl["others"][real])
        assert np.array_equal(nl["cell_shifts"][rev[real]], -nl["cell_shifts"][real])
        assert np.array_equal(rev[n_real:], np.arange(n_real, rev.shape[0]))


def test_neighbor_list_overflow():
    """One slot short (the last slot stays empty) is an overflow; exact fit is not."""
    atoms, cutoff = bulk("NaCl", crystalstructure="rocksalt", a=5.64), 6.0
    n_pairs = int(to_structure(atoms, cutoff=cutoff, skin=0.0)["pair_mask"].sum())
    assert not _device_list(atoms, cutoff, n_pairs + 1)[2]
    assert _device_list(atoms, cutoff, n_pairs)[2]
