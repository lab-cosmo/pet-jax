import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.build import bulk, molecule

from petjax.structure import center_order, corresponding_edges, to_structure


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
    centers, others, shifts, mask = _fields(structure)
    got = jax.jit(corresponding_edges)(
        centers.astype(np.int32), others.astype(np.int32), shifts.astype(np.float32), mask
    )
    assert got.dtype == jnp.int32
    assert np.array_equal(got, structure["reverse"])


def test_center_order_then_corresponding_edges(structure):
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
