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


# -- select_edges: the adaptive selection's mask, host-side --


def test_select_edges_mask(structure):
    """The host-side selection is internally consistent (pair cutoffs are the
    endpoint means of the atomic ones, the mask is ``r_ij <= pair_cutoff`` on
    unmasked pairs, symmetric under ``reverse``) and matches the eager selection
    core on the same displacements."""
    from petjax import select_edges
    from petjax.select import _select_edges
    from petjax.utils import edge_displacements, safe_norm

    hypers = dict(num_neighbors_adaptive=4, cutoff=CUTOFF, cutoff_width_adaptive=0.5)
    sel = select_edges(structure, **hypers, method="grid")
    centers, others, pair_mask = (structure[k] for k in ("centers", "others", "pair_mask"))
    N = structure["positions"].shape[0]

    assert all(isinstance(x, np.ndarray) for x in sel)
    assert sel.atomic_cutoffs.shape == (N,)
    assert sel.pair_cutoffs.shape == centers.shape
    assert sel.selected.dtype == bool and sel.selected.shape == centers.shape
    assert not sel.selected[~pair_mask].any()
    assert np.array_equal(sel.selected[structure["reverse"]], sel.selected)

    R_ij = edge_displacements(
        structure["positions"], centers, others, structure["cell_shifts"], structure["cell"]
    )
    r_ij = np.asarray(safe_norm(R_ij, axis=-1))
    pair_cutoffs = (sel.atomic_cutoffs[centers] + sel.atomic_cutoffs[others]) / 2
    assert np.array_equal(sel.pair_cutoffs, pair_cutoffs)
    assert np.array_equal(sel.selected, (r_ij <= sel.pair_cutoffs) & pair_mask)

    # The jitted kernel and the eager core agree to float32 rounding on the
    # cutoffs (XLA fuses the reductions differently) and exactly on the mask.
    eager = _select_edges(R_ij, centers, others, pair_mask, N, **hypers, method="grid")
    assert np.allclose(sel.atomic_cutoffs, eager[0], rtol=1e-6, atol=0)
    assert np.allclose(sel.pair_cutoffs, eager[1], rtol=1e-6, atol=0)
    assert np.array_equal(sel.selected, np.asarray(eager[2]))
