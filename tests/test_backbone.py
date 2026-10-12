"""`Backbone` hooks (`species_embedding` / `node_features`, `cutoff_function`):
checkpoint-free checks on a tiny randomly initialised backbone."""

import numpy as np
import jax
import jax.numpy as jnp

import pytest

from petjax import Backbone, cutoff_cosine

CONFIG = dict(
    d_pet=8,
    d_node=16,
    d_feedforward=8,
    num_heads=2,
    num_gnn_layers=2,
    cutoff=3.0,
    cutoff_width=1.0,
    max_atomic_number=8,
)


def _inputs():
    """Three atoms, all pairs within the cutoff: N = 3, k = 2, P = 6."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.2, 0.0]])
    pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    centers = jnp.array([i for i, _ in pairs])
    neighbors = jnp.array([j for _, j in pairs])
    return {
        "R_ij": positions[neighbors] - positions[centers],
        "centers": centers,
        "neighbors": neighbors,
        "atomic_numbers": jnp.array([8, 1, 1]),
        "reverse": jnp.array([pairs.index((j, i)) for i, j in pairs]),
        "pair_mask": jnp.ones(6, dtype=bool),
        "atom_mask": jnp.ones(3, dtype=bool),
    }


def test_node_features_reproduce_species_embedding():
    inputs = _inputs()
    model = Backbone(**CONFIG)
    params = model.init(jax.random.key(0), **inputs)
    reference = model.apply(params, **inputs)

    # The external-features backbone owns everything but the species table,
    # so the default model's params minus that table must fit it exactly.
    table = params["params"].pop("node_embedders_0")["embedding"]
    external = Backbone(**CONFIG, species_embedding=False)
    features = table[inputs["atomic_numbers"]]
    assert (
        "node_embedders_0"
        not in external.init(jax.random.key(0), **inputs, node_features=features)["params"]
    )
    injected = external.apply(params, **inputs, node_features=features)
    for a, b in zip(reference, injected):
        np.testing.assert_array_equal(a, b)


def test_node_features_must_match_species_embedding():
    inputs = _inputs()
    features = jnp.zeros((3, CONFIG["d_node"]))
    with pytest.raises(ValueError, match="species_embedding=False"):
        Backbone(**CONFIG).init(jax.random.key(0), **inputs, node_features=features)
    with pytest.raises(ValueError, match="needs node_features"):
        Backbone(**CONFIG, species_embedding=False).init(jax.random.key(0), **inputs)


def test_cutoff_function_is_used():
    inputs = _inputs()
    model = Backbone(**CONFIG, cutoff_function=cutoff_cosine)
    params = model.init(jax.random.key(0), **inputs)
    _, _, cutoffs = model.apply(params, **inputs)

    r = jnp.linalg.norm(inputs["R_ij"], axis=-1)
    np.testing.assert_allclose(cutoffs, cutoff_cosine(r, 3.0, 1.0), atol=1e-6)
    assert 0.0 < float(cutoffs.min()) < 1.0  # the 2.2 / 2.4 Å pairs sit in the taper


def test_cutoff_cosine():
    r = jnp.array([0.5, 2.0, 2.5, 3.0, 4.0])
    np.testing.assert_allclose(
        cutoff_cosine(r, 3.0, 1.0), [1.0, 1.0, 0.5, 0.0, 0.0], atol=1e-7
    )
