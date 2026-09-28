"""Charge / spin-multiplicity conditioning: model, structure inputs, calculator.

Parity with upstream is checked in `test_predictions.py` style against a real
conditioned checkpoint; what is pinned here is the plumbing — that the inputs
reach the model, that an unconditioned model is untouched, and that bad inputs
fail host-side rather than clamp silently inside JIT.
"""

import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.io import read

from petjax import UPET, UPETCalculator, to_structure
from petjax.select import truncate
from petjax.structure import conditioning_inputs

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
    adaptive_cutoff_method="grid",
    num_neighbors_adaptive=8,
    max_atomic_number=118,
)
CONDITIONED = dict(system_conditioning=True, max_charge=3, max_spin_multiplicity=3)


@pytest.fixture(scope="module")
def atoms(mini_xyz):
    return read(str(mini_xyz), index="0")


def make_calc(atoms, **overrides):
    """Randomly initialised calculator; the random gate makes the conditioning
    branch active, unlike upstream's zero-init at the start of training."""
    config = {**CONFIG, **overrides}
    model = UPET(**config)
    structure = to_structure(atoms, config["cutoff"])
    structure["k_sel_sizer"] = jnp.zeros(16, dtype=bool)
    truncated, _ = truncate(structure, 8, config["cutoff"], 0.5)
    params = model.init(jax.random.PRNGKey(0), **truncated)
    metadata = {
        "config": config,
        "shifts": {int(z): 0.0 for z in set(atoms.get_atomic_numbers())},
    }
    return UPETCalculator(model, params, metadata, stress=False)


def energy(calc, atoms, **info):
    atoms = atoms.copy()
    atoms.info.update(info)
    atoms.calc = calc
    return atoms.get_potential_energy()


def test_conditioning_params_present(atoms):
    calc = make_calc(atoms, **CONDITIONED)
    cond = calc._params["params"]["backbone"]["system_conditioning"]
    assert cond["charge_embedding"]["embedding"].shape == (7, 16)
    assert cond["spin_multiplicity_embedding"]["embedding"].shape == (4, 16)
    assert cond["project"]["Dense_0"]["kernel"].shape == (32, 16)
    assert cond["project"]["Dense_1"]["kernel"].shape == (16, 16)


def test_charge_and_spin_change_the_energy(atoms):
    calc = make_calc(atoms, **CONDITIONED)
    neutral = energy(calc, atoms)
    assert energy(calc, atoms, charge=0, spin_multiplicity=1) == neutral
    assert energy(calc, atoms, charge=2) != neutral
    assert energy(calc, atoms, spin_multiplicity=3) != neutral
    # Back to the defaults: the cached-result invalidation goes both ways.
    assert energy(calc, atoms, charge=0) == neutral


def test_info_change_recomputes_without_rebuild(atoms):
    """A new charge on the same geometry goes through the position-only
    update, not an NL rebuild."""
    calc = make_calc(atoms, **CONDITIONED)
    energy(calc, atoms)
    n_pair = calc.debug_stats["n_pair_raw"]
    calc.debug_stats = None
    energy(calc, atoms, charge=1)
    assert calc.debug_stats is None
    assert calc._structure["charge"][0] == 1
    assert int(calc._structure["pair_mask"].sum()) == n_pair


def test_unconditioned_model_ignores_info(atoms):
    calc = make_calc(atoms)
    neutral = energy(calc, atoms)
    assert "charge" not in calc._structure
    assert energy(calc, atoms, charge=2, spin_multiplicity=3) == neutral


def test_unconditioned_model_accepts_inputs(atoms):
    """The extra keyword arguments are part of the model signature whether or
    not the model reads them, so batched callers can always pass them."""
    config = {**CONFIG}
    model = UPET(**config)
    structure = to_structure(atoms, config["cutoff"])
    structure["k_sel_sizer"] = jnp.zeros(16, dtype=bool)
    truncated, _ = truncate(structure, 8, config["cutoff"], 0.5)
    params = model.init(jax.random.PRNGKey(0), **truncated)
    N = truncated["atomic_numbers"].shape[0]
    with_inputs = model.apply(
        params,
        **truncated,
        charge=jnp.full(N, 2),
        spin_multiplicity=jnp.full(N, 3),
    )
    without = model.apply(params, **truncated)
    np.testing.assert_array_equal(with_inputs, without)


@pytest.mark.parametrize(
    "info, match",
    [
        ({"charge": 4}, r"charge=4 outside"),
        ({"charge": -4}, r"charge=-4 outside"),
        ({"spin_multiplicity": 0}, r"spin_multiplicity=0 outside"),
        ({"spin_multiplicity": 4}, r"spin_multiplicity=4 outside"),
        ({"charge": 0.5}, r"single integer value"),
        ({"charge": [1, 0]}, r"single integer value"),
    ],
)
def test_invalid_inputs_rejected(atoms, info, match):
    with pytest.raises(ValueError, match=match):
        conditioning_inputs(_with_info(atoms, info), 8, 3, 3)


def test_integer_valued_floats_accepted(atoms):
    inputs = conditioning_inputs(_with_info(atoms, {"charge": -1.0}), 8, 3, 3)
    assert inputs["charge"].dtype == np.int64
    assert (inputs["charge"] == -1).all()
    assert (inputs["spin_multiplicity"] == 1).all()


def test_spin_key_fallback(atoms):
    """OMol data spells the multiplicity `spin`; `spin_multiplicity` wins when
    both are present, and a change of `spin` alone invalidates the cache."""
    inputs = conditioning_inputs(_with_info(atoms, {"spin": 3}), 8, 3, 3)
    assert (inputs["spin_multiplicity"] == 3).all()
    inputs = conditioning_inputs(
        _with_info(atoms, {"spin": 3, "spin_multiplicity": 2}), 8, 3, 3
    )
    assert (inputs["spin_multiplicity"] == 2).all()
    calc = make_calc(atoms, **CONDITIONED)
    assert energy(calc, atoms, spin=3) == energy(calc, atoms, spin_multiplicity=3)
    assert energy(calc, atoms, spin=3) != energy(calc, atoms)


def _with_info(atoms, info):
    atoms = atoms.copy()
    atoms.info.update(info)
    return atoms
