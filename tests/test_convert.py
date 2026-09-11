"""Unit tests for the state-dict rename pipeline in `petjax.convert`.

The end-to-end checks live in `test_predictions.py`, which runs real
checkpoints of both PET-MAD readout-head naming schemes. What is worth pinning
separately is the head-scoping rule itself, because getting it wrong is silent:
a readout target that fails to match its head module used to fall through to
`energy_head` and overwrite the energy weights with another head's.
"""

import numpy as np

import pytest

from petjax.convert import _finalize_key, _rename_key, _scope_key


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
    ],
)
def test_head_scoping(state_dict_key, expected):
    assert scope_of(state_dict_key) == expected


def test_unknown_readout_target_rejected():
    """An unrecognized target must fail loudly, not land in energy_head."""
    with pytest.raises(ValueError, match="unknown readout target"):
        scope_of("node_heads.non_conservative_dipole.0.0.bias")
