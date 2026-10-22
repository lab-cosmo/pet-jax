"""The flat token layout (``Express.tokens``) beyond the equivalence in
``test_express.py``: the capacity's overflow flag and the calculator's retry."""

import numpy as np
import jax.numpy as jnp

import pytest
from ase.io import read

from petjax import UPETCalculator
from petjax.express import Express
from petjax.select import truncate

FLAT = Express(tokens="flat")


@pytest.fixture(scope="module")
def checkpoint(assets_dir):
    path = assets_dir / "checkpoints" / "pet-mad-xs-v1.6"  # 16 neighbours
    if not (path / "model.msgpack").exists():
        pytest.skip("pet-mad-xs-v1.6 checkpoint missing")
    return path


@pytest.fixture(scope="module")
def atoms(mini_xyz):
    return read(str(mini_xyz), index=0)  # periodic, 9 atoms


def calculator(checkpoint, atoms, express):
    calc = UPETCalculator.from_checkpoint(
        str(checkpoint), default_dtype="float64", express=express
    )
    atoms = atoms.copy()
    atoms.calc = calc
    atoms.get_potential_energy()
    return calc, atoms


def _truncate(calc, structure):
    model = calc._model
    return truncate(
        structure,
        calc._num_neighbors_adaptive,
        model.cutoff,
        model.cutoff_width_adaptive,
        method=model.adaptive_cutoff_method,
        express=calc.express,
    )


def test_capacity_overflow(checkpoint, atoms):
    calc, _ = calculator(checkpoint, atoms, FLAT)
    sizes = calc.debug_stats["tokens"]
    structure = calc._structure
    k_max = int(_truncate(calc, structure)[1].counts.max())

    def overflow(**sizers):
        resized = {**structure, **{k: jnp.zeros(n, bool) for k, n in sizers.items()}}
        return bool(_truncate(calc, resized)[1].overflow)

    assert not overflow()
    assert not overflow(token_sizer=sizes["n_tokens"] + 1)
    assert overflow(token_sizer=sizes["n_tokens"])  # the sentinel slot must stay free
    assert overflow(k_sel_sizer=k_max - 1)


def test_overflow_retry_resizes(checkpoint, atoms):
    """A capacity that falls short inside the skin is widened by the retry."""
    stock, moved = calculator(checkpoint, atoms, None)
    calc, flat = calculator(checkpoint, atoms, FLAT)
    n_tokens = calc.debug_stats["tokens"]["n_tokens"]
    calc._structure["token_sizer"] = jnp.zeros(n_tokens, bool)

    shift = np.zeros_like(atoms.positions)
    shift[0, 0] = 0.01
    for a in (moved, flat):
        a.positions = atoms.positions + shift
    assert abs(flat.get_potential_energy() - moved.get_potential_energy()) < 1e-10
    assert calc.debug_stats["overflow_retry"]
    assert calc._structure["token_sizer"].shape[0] > n_tokens
    np.testing.assert_allclose(flat.get_forces(), moved.get_forces(), atol=1e-10)
