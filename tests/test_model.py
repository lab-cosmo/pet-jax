"""UPET.__call__ plumbing: return_features exposes the node embedding unchanged."""

import numpy as np
import jax
import jax.numpy as jnp

import pytest

from petjax import UPET

D_NODE = 8


def _model(**kwargs):
    return UPET(
        d_pet=8,
        d_node=D_NODE,
        d_head=8,
        d_feedforward=8,
        num_heads=2,
        num_attention_layers=1,
        num_gnn_layers=1,
        cutoff=3.5,
        **kwargs,
    )


@pytest.fixture
def inputs():
    rng = np.random.default_rng(0)
    n, p = 3, 6
    return {
        "R_ij": jnp.asarray(rng.normal(scale=1.5, size=(p, 3)), dtype=jnp.float32),
        "centers": jnp.asarray([0, 0, 1, 1, 2, 2]),
        "neighbors": jnp.asarray([1, 2, 0, 2, 0, 1]),
        "species": jnp.asarray([1, 8, 1]),
        "reverse": jnp.asarray([2, 4, 0, 5, 1, 3]),
        "pair_mask": jnp.ones(p, dtype=bool),
        "atom_mask": jnp.ones(n, dtype=bool),
    }


def test_return_features_leaves_energy_unchanged(inputs):
    model = _model()
    params = model.init(jax.random.key(0), **inputs)
    energy = model.apply(params, **inputs)
    energy_f, node = model.apply(params, **inputs, return_features=True)
    assert jnp.array_equal(energy, energy_f)
    assert node.shape == (3, D_NODE)


def test_return_features_with_direct_heads(inputs):
    model = _model(direct_forces=True, direct_stress=True)
    params = model.init(jax.random.key(0), **inputs)
    out = model.apply(params, **inputs)
    out_f, node = model.apply(params, **inputs, return_features=True)
    assert set(out) == set(out_f) == {"energy", "forces", "stress"}
    for k in out:
        assert jnp.array_equal(out[k], out_f[k])
    assert node.shape == (3, D_NODE)
