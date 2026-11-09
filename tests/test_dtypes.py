"""A float32 model stays in float32 when 64-bit mode is enabled: neither the
forward nor its derivatives (forces and stress) may promote to float64."""

import jax
import jax.numpy as jnp

import re

import pytest
from ase.io import read

from petjax import UPET, get_predict_fn, to_structure
from petjax.select import truncate
from petjax.utils import cast_floats

CONFIG = dict(
    d_pet=8,
    d_node=16,
    d_head=8,
    d_feedforward=8,
    num_heads=1,
    num_attention_layers=1,
    num_gnn_layers=2,
    cutoff=4.0,
    cutoff_width=0.5,
    cutoff_width_adaptive=0.5,
    num_neighbors_adaptive=8,
    max_atomic_number=118,
)


@pytest.fixture
def enable_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("method", ["grid", "solver"])
def test_float32_with_x64(mini_xyz, enable_x64, method):
    atoms = read(str(mini_xyz), index="0")
    model = UPET(**CONFIG, adaptive_cutoff_method=method)

    structure = cast_floats(to_structure(atoms, CONFIG["cutoff"]), jnp.float32)
    structure["k_sel_sizer"] = jnp.zeros(16, dtype=bool)
    truncated, _ = truncate(
        structure, CONFIG["num_neighbors_adaptive"], CONFIG["cutoff"], 0.5, method=method
    )
    params = cast_floats(model.init(jax.random.key(0), **truncated), jnp.float32)

    # Python scalars (e.g. `jnp.clip(x, 1e-6, ...)`) are float64 constants in
    # the compiled code and converted on use, so only arrays are checked
    predict_fn = get_predict_fn(model, stress=True)
    hlo = predict_fn.lower(params, structure).as_text()
    assert re.findall(r"tensor<[0-9x]+xf64>", hlo) == []

    results = predict_fn(params, structure)
    for key in ("energy", "forces", "stress"):
        assert results[key].dtype == jnp.float32
