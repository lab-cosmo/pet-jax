"""Device-side flat-NL builders and layout helpers against the host builders
in structure.py."""

import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.build import bulk, molecule

from petjax import center_order, corresponding_edges, get_neighbor_list_fn
from petjax.structure import to_structure
from petjax.utils import edge_displacements


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


# -- device-built raw list (vesin-jax) --


def _x64(enabled):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", enabled)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def enable_x64():
    """vesin-jax only runs with 64-bit mode."""
    yield from _x64(True)


@pytest.fixture
def disable_x64():
    """Other tests (the float64 calculators) leave 64-bit mode enabled."""
    yield from _x64(False)


@pytest.fixture(params=["primitive", "supercell", "slab", "molecule"])
def system(request, enable_x64):
    """``(atoms, cutoff)``, covering several images per pair, mixed periodicity
    and no periodicity."""
    if request.param == "primitive":
        return bulk("NaCl", crystalstructure="rocksalt", a=5.64), 6.0
    elif request.param == "supercell":
        atoms = bulk("Cu", cubic=True) * [2, 2, 2]
        atoms.rattle(0.1, seed=0)
        return atoms, 4.0
    elif request.param == "slab":
        atoms = bulk("Cu", cubic=True) * [2, 2, 2]
        atoms.pbc = [True, True, False]
        return atoms, 4.0
    else:
        return molecule("CH3CH2OH"), 3.0


def _max_neighbors(host):
    """The largest number of pairs of an atom of a host structure."""
    centers = np.asarray(host["centers"])[np.asarray(host["pair_mask"])]
    return max(int(np.bincount(centers).max()), 1) if len(centers) else 1


def _device_list(atoms, host, radius, skin=0.0, max_neighbors=None, n_atoms=None):
    if max_neighbors is None:
        max_neighbors = _max_neighbors(host)
    neighbor_list = get_neighbor_list_fn(radius, max_neighbors, skin=skin)
    fn = jax.jit(lambda positions, cell: neighbor_list(positions, cell, atoms.pbc, n_atoms))
    nl = fn(jnp.asarray(host["positions"]), jnp.asarray(host["cell"]))
    return jax.tree.map(np.asarray, nl)


def _pairs(nl):
    mask = nl["pair_mask"]
    return set(
        zip(
            nl["centers"][mask].tolist(),
            nl["others"][mask].tolist(),
            map(tuple, np.rint(nl["cell_shifts"][mask]).astype(int).tolist()),
            strict=True,
        )
    )


def _check_layout(nl, dummy):
    """The to_structure layout: real pairs first and sorted by center, padding
    on the dummy atom with zero shift and reverse = self, last slot empty, and
    the reverse map pairs (i, j, S) with (j, i, -S)."""
    centers, others, shifts, mask, rev = (
        nl[k] for k in ("centers", "others", "cell_shifts", "pair_mask", "reverse")
    )
    n_real = int(mask.sum())
    assert mask[:n_real].all() and not mask[n_real:].any()
    assert not mask[-1]
    assert np.all(np.diff(centers[:n_real]) >= 0)
    assert np.all(centers[n_real:] == dummy) and np.all(others[n_real:] == dummy)
    assert np.all(shifts[n_real:] == 0)
    assert np.array_equal(rev[n_real:], np.arange(n_real, rev.shape[0]))
    real = np.arange(n_real)
    assert np.array_equal(centers[rev[real]], others[real])
    assert np.array_equal(others[rev[real]], centers[real])
    assert np.array_equal(shifts[rev[real]], -shifts[real])


def test_neighbor_list_matches_host(system):
    atoms, cutoff = system
    host = jax.tree.map(np.asarray, to_structure(atoms, cutoff=cutoff, skin=0.0))
    nl = _device_list(atoms, host, cutoff)

    assert nl["centers"].shape == (len(atoms) * _max_neighbors(host) + 1,)
    assert nl["centers"].dtype == host["centers"].dtype
    assert nl["cell_shifts"].dtype == host["cell_shifts"].dtype
    assert _pairs(nl) == _pairs(host)
    _check_layout(nl, dummy=host["positions"].shape[0] - 1)


def test_neighbor_list_padded_atoms(enable_x64):
    """With several padding atoms, only the first n_atoms are searched and the
    padding still lands on the last (dummy) atom."""
    atoms = bulk("Cu", cubic=True) * [2, 2, 2]
    host = jax.tree.map(
        np.asarray,
        to_structure(atoms, cutoff=4.0, skin=0.0, n_atoms_bucket_strategy="multiples"),
    )
    N_padded = host["positions"].shape[0]
    assert N_padded > len(atoms) + 1

    nl = _device_list(atoms, host, 4.0, n_atoms=len(atoms))
    assert _pairs(nl) == _pairs(host)
    _check_layout(nl, dummy=N_padded - 1)


def test_neighbor_list_overflow(enable_x64):
    """An atom with more neighbors than max_neighbors is an error, never a
    truncated list."""
    atoms, cutoff = bulk("Cu", cubic=True) * [2, 2, 2], 4.0
    host = jax.tree.map(np.asarray, to_structure(atoms, cutoff=cutoff, skin=0.0))
    with pytest.raises(jax.errors.JaxRuntimeError, match="more neighbors"):
        _device_list(atoms, host, cutoff, max_neighbors=_max_neighbors(host) - 1)


def test_neighbor_list_needs_x64(disable_x64):
    atoms, cutoff = bulk("Cu", cubic=True) * [2, 2, 2], 4.0
    host = to_structure(atoms, cutoff=cutoff, skin=0.0, float_dtype=np.float32)
    with pytest.raises(RuntimeError, match="jax_enable_x64"):
        _device_list(atoms, host, cutoff)


def test_neighbor_list_skin(enable_x64):
    """With a skin, the list stays exact along a trajectory, across rebuilds
    and cell changes."""
    atoms, cutoff = bulk("Cu", cubic=True) * [2, 2, 2], 4.0
    host = to_structure(atoms, cutoff=cutoff, skin=0.5)
    # room for the pairs that enter the cutoff along the trajectory
    max_neighbors = _max_neighbors(host)
    exact = get_neighbor_list_fn(cutoff, max_neighbors)
    cached = get_neighbor_list_fn(cutoff, max_neighbors, skin=0.5)
    exact = jax.jit(lambda positions, cell, nl=exact: nl(positions, cell, atoms.pbc))
    cached = jax.jit(lambda positions, cell, nl=cached: nl(positions, cell, atoms.pbc))

    rng = np.random.default_rng(0)
    positions = jnp.asarray(host["positions"])
    cell = jnp.asarray(host["cell"])
    for step in range(8):
        positions = positions.at[:-1].add(rng.normal(scale=0.05, size=(len(atoms), 3)))
        if step == 5:
            cell = cell * 1.01
        nl = cached(positions, cell)
        expected = exact(positions, cell)
        nl, expected = jax.tree.map(np.asarray, (nl, expected))
        assert _pairs(nl) == _pairs(expected)


def test_neighbor_list_gradients(system):
    """A pair energy through the device list built inside the differentiated
    function has the host list's energy, forces and cell derivative."""
    atoms, cutoff = system
    host = to_structure(atoms, cutoff=cutoff, skin=0.0)
    neighbor_list = get_neighbor_list_fn(cutoff, _max_neighbors(host))

    def pair_energy(positions, cell, nl):
        R = edge_displacements(
            positions, nl["centers"], nl["others"], nl["cell_shifts"], cell
        )
        r = jnp.sqrt(jnp.sum(R**2, axis=-1) + 1e-15)
        return jnp.sum(jnp.where(nl["pair_mask"], (cutoff - r) ** 3, 0.0))

    def device_energy(positions, cell):
        nl = neighbor_list(positions, cell, atoms.pbc)
        return pair_energy(positions, cell, nl)

    def host_energy(positions, cell):
        nl = {k: jnp.asarray(host[k]) for k in ("centers", "others", "cell_shifts")}
        nl["pair_mask"] = jnp.asarray(host["pair_mask"])
        return pair_energy(positions, cell, nl)

    positions = jnp.asarray(host["positions"], dtype=jnp.float32)
    cell = jnp.asarray(host["cell"], dtype=jnp.float32)
    grad = jax.value_and_grad(device_energy, argnums=(0, 1))
    energy, (forces, virial) = jax.jit(grad)(positions, cell)
    energy_ref, (forces_ref, virial_ref) = jax.value_and_grad(host_energy, argnums=(0, 1))(
        positions, cell
    )

    np.testing.assert_allclose(energy, energy_ref, rtol=1e-5)
    np.testing.assert_allclose(forces, forces_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(virial, virial_ref, rtol=1e-4, atol=1e-4)
