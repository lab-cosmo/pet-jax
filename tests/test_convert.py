"""Unit tests for the state-dict rename pipeline in `petjax.convert`.

The end-to-end checks live in `test_predictions.py`, which runs real
checkpoints of both PET-MAD readout-head naming schemes. What is worth pinning
separately is the head-scoping rule itself: a readout target that lands on the
wrong head module overwrites that head's weights, which no shape check
necessarily catches.
"""

import numpy as np

import pytest

from petjax.convert import (
    _CHARGE_TABLE,
    _SPIN_TABLE,
    _finalize_key,
    _layout_conditioning_tables,
    _rename_key,
    _scatter_species_embeddings,
    _scope_key,
)


def scope_of(state_dict_key):
    """Run one state-dict key through the pipeline, return its module scope.

    The value only has to survive `_finalize_key`'s transpose, so any 2D array
    stands in for the real tensor.
    """
    renamed, _ = _finalize_key(_rename_key(state_dict_key), np.zeros((2, 3)))
    return _scope_key(renamed, state_dict_key).split(".")[0]


@pytest.mark.parametrize(
    "state_dict_key, expected",
    [
        # PET-MAD v1.5 spells the direct-force target plural, v1.6 singular;
        # both are the same head.
        ("node_heads.non_conservative_forces.0.0.bias", "forces_head"),
        ("node_heads.non_conservative_force.0.0.bias", "forces_head"),
        (
            "node_last_layers.non_conservative_forces.0.non_conservative_forces___0.bias",
            "forces_head",
        ),
        (
            "node_last_layers.non_conservative_force.0.non_conservative_force___0.bias",
            "forces_head",
        ),
        ("edge_heads.non_conservative_stress.0.0.bias", "stress_head"),
        ("edge_heads.energy.0.0.bias", "energy_head"),
        ("node_last_layers.energy.0.energy___0.bias", "energy_head"),
        # Everything that is not a readout head belongs to the feature core.
        ("gnn_layers.0.trans.layers.0.attention.input_linear.bias", "backbone"),
        ("node_embedders.0.weight", "backbone"),
        ("system_conditioning.charge_embedding.weight", "backbone"),
    ],
)
def test_head_scoping(state_dict_key, expected):
    assert scope_of(state_dict_key) == expected


@pytest.mark.parametrize(
    "state_dict_key, expected",
    [
        (
            "system_conditioning.charge_embedding.weight",
            "system_conditioning.charge_embedding.embedding",
        ),
        (
            "system_conditioning.spin_multiplicity_embedding.weight",
            "system_conditioning.spin_multiplicity_embedding.embedding",
        ),
        (
            "system_conditioning.project.0.weight",
            "system_conditioning.project.Dense_0.kernel",
        ),
        ("system_conditioning.project.2.bias", "system_conditioning.project.Dense_1.bias"),
    ],
)
def test_conditioning_key_names(state_dict_key, expected):
    """The conditioning module's Flax names, as `SystemConditioning` declares them."""
    renamed, _ = _finalize_key(_rename_key(state_dict_key), np.zeros((2, 3)))
    assert renamed == expected


def test_conditioning_tables_indexed_by_value():
    """Upstream's offset rows become value-indexed rows: charge c at row c
    (negative c counted from the end), multiplicity m at row m."""
    max_charge = 2
    upstream_charge = np.arange(-max_charge, max_charge + 1, dtype=np.float32)[:, None]
    upstream_spin = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    flat = {_CHARGE_TABLE: upstream_charge.copy(), _SPIN_TABLE: upstream_spin.copy()}
    _layout_conditioning_tables(flat, max_charge)
    charge = np.asarray(flat[_CHARGE_TABLE])
    spin = np.asarray(flat[_SPIN_TABLE])
    for c in range(-max_charge, max_charge + 1):
        assert charge[c, 0] == c
    assert spin[0, 0] == 0.0
    for m in (1, 2, 3):
        assert spin[m, 0] == m


def test_scatter_leaves_conditioning_tables_alone():
    """Species tables are re-indexed by atomic number; the charge / spin
    tables are indexed by value and must keep their rows."""
    species = np.arange(6, dtype=np.float32).reshape(2, 3)
    charge = np.arange(15, dtype=np.float32).reshape(5, 3)
    flat = {
        "backbone.node_embedders_0.embedding": species,
        "backbone.system_conditioning.charge_embedding.embedding": charge,
    }
    _scatter_species_embeddings(flat, atomic_types=[1, 8], n_rows=10)
    scattered = np.asarray(flat["backbone.node_embedders_0.embedding"])
    assert scattered.shape == (10, 3)
    np.testing.assert_array_equal(scattered[[1, 8]], species)
    np.testing.assert_array_equal(
        flat["backbone.system_conditioning.charge_embedding.embedding"], charge
    )


def test_unknown_readout_target_rejected():
    """An unrecognized target must fail loudly, not land in energy_head."""
    with pytest.raises(ValueError, match="unknown readout target"):
        scope_of("node_heads.non_conservative_dipole.0.0.bias")
