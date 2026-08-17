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
        structure["atomic_numbers"],
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


def _orphaned(structure):
    """Copy of `structure` with one direction of one pair dropped.

    Mimics an upstream per-center trim (or an overflow drop): the surviving
    direction keeps a `reverse` pointing at the padded sentinel pair, so the
    forward gathers a masked slot.
    """
    structure = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in structure.items()}
    sentinel = structure["pair_mask"].shape[0] - 1
    real = np.flatnonzero(structure["pair_mask"][:sentinel])
    kept = int(real[0])
    dropped = int(structure["reverse"][kept])
    assert dropped != kept

    structure["pair_mask"][dropped] = False
    structure["reverse"][kept] = sentinel
    return structure


def _backbone():
    from petjax.model import Backbone

    return Backbone(
        d_pet=8,
        d_node=8,
        d_feedforward=8,
        num_heads=2,
        num_attention_layers=1,
        num_gnn_layers=2,
        cutoff=CUTOFF,
    )


def _run_backbone(structure, k, shift=0.0):
    """Backbone on `structure` packed to width `k`.

    `shift` offsets every parameter: a fresh init leaves all biases zero, which
    happens to zero the padded sentinel slot and hide any leak through it.
    """
    backbone = _backbone()
    packed, overflow = _pack(structure, k)
    assert not bool(overflow)
    inputs = {key: value for key, value in packed.items() if key != "pair_cutoffs"}
    params = backbone.init(jax.random.key(0), **inputs, pair_cutoffs=None)
    if shift:
        params = jax.tree.map(lambda x: x + shift, params)
    node, messages, _ = backbone.apply(params, **inputs, pair_cutoffs=None)
    return np.asarray(node), np.asarray(messages), np.asarray(packed["pair_mask"])


def test_masked_slots_carry_no_features(structure):
    """Padded pair slots leave the backbone at exactly zero.

    The reverse gather is the one consumer of a masked slot, so anything left
    there leaks into real edges whose reciprocal was trimmed upstream.
    """
    _, messages, pair_mask = _run_backbone(structure, _max_count(structure) + 5)
    assert np.all(messages[~pair_mask] == 0.0)


def test_backbone_invariant_to_packed_width(structure):
    """`k` is a padding choice: widening it must not move real features.

    Runs with an orphaned reverse -- the one case that reads a masked slot --
    and with shifted parameters, so the sentinel slot is not incidentally zero.
    """
    orphaned = _orphaned(structure)
    k = _max_count(orphaned)

    node_tight, msg_tight, mask_tight = _run_backbone(orphaned, k, shift=0.3)
    node_wide, msg_wide, mask_wide = _run_backbone(orphaned, k + 11, shift=0.3)

    np.testing.assert_allclose(node_tight, node_wide, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        msg_tight[mask_tight], msg_wide[mask_wide], rtol=1e-5, atol=1e-6
    )
