"""pack_edges: selection-free fixed-width packing of a flat NL."""

import numpy as np
import jax

import pytest
from ase.build import bulk, molecule

from petjax import pack_edges
from petjax.structure import to_structure

CUTOFF = 3.5


def _structure(atoms):
    return jax.tree.map(np.asarray, to_structure(atoms, cutoff=CUTOFF, skin=0.0))


def _pack(structure, k):
    from petjax.utils import edge_displacements

    R_ij = edge_displacements(
        structure["positions"],
        structure["centers"],
        structure["others"],
        structure["cell_shifts"],
        structure["cell"],
    )
    return pack_edges(
        R_ij,
        structure["centers"],
        structure["others"],
        structure["reverse"],
        structure["pair_mask"],
        structure["species"],
        structure["atom_mask"],
        k,
    )


def _max_count(structure):
    real = structure["pair_mask"]
    return int(np.bincount(structure["centers"][real]).max())


@pytest.fixture(params=["crystal", "molecule"])
def structure(request):
    if request.param == "crystal":
        atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64) * [2, 2, 2]
    else:
        atoms = molecule("H2O")
    return _structure(atoms)


def test_pack_edges_round_trip(structure):
    """Every unmasked pair survives packing exactly once; nothing else does."""
    k = _max_count(structure)
    packed, overflow = _pack(structure, k)
    assert not bool(overflow)
    assert packed["pair_cutoffs"] is None

    real = np.asarray(packed["pair_mask"])
    got = {
        (int(c), int(o))
        for c, o in zip(
            np.asarray(packed["centers"])[real], np.asarray(packed["neighbors"])[real]
        )
    }
    mask = structure["pair_mask"]
    want = {
        (int(c), int(o))
        for c, o in zip(structure["centers"][mask], structure["others"][mask])
    }
    assert got == want
    assert int(real.sum()) == int(mask.sum())


def test_pack_edges_reverse_involution(structure):
    k = _max_count(structure)
    packed, _ = _pack(structure, k)
    rev = np.asarray(packed["reverse"])
    real = np.flatnonzero(np.asarray(packed["pair_mask"]))
    assert np.array_equal(rev[rev[real]], real)


def test_pack_edges_overflow_on_undersized_width(structure):
    k = _max_count(structure)
    if k < 2:
        pytest.skip("needs at least 2 neighbors on some center")
    _, overflow = _pack(structure, k - 1)
    assert bool(overflow)


def test_pack_edges_forward_smoke(structure):
    """UPET consumes the packed layout with pair_cutoffs=None (scalar-cutoff bump)."""
    import jax.numpy as jnp

    from petjax import UPET

    model = UPET(
        d_pet=8,
        d_node=8,
        d_head=8,
        d_feedforward=8,
        num_heads=2,
        num_attention_layers=1,
        num_gnn_layers=1,
        cutoff=CUTOFF,
    )
    packed, overflow = _pack(structure, _max_count(structure))
    assert not bool(overflow)
    params = model.init(jax.random.key(0), **packed)
    energy = model.apply(params, **packed)
    assert bool(jnp.all(jnp.isfinite(energy)))
