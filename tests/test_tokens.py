"""The flat token layout and the attention tiles (``Express.tokens`` /
``attention_tile``) beyond the equivalence in ``test_express.py``: the
capacities' overflow flags, the calculator's retry and sizing, and higher-order
autodiff through the tiles."""

import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.io import read

from petjax import UPETCalculator
from petjax.express import Express
from petjax.predict import _select_and_predict
from petjax.select import truncate

TILED = Express(tokens="flat", attention_tile=12)


@pytest.fixture(scope="module")
def checkpoint(assets_dir):
    path = assets_dir / "checkpoints" / "pet-mad-xs-v1.6"  # 16 neighbours
    if not (path / "model.msgpack").exists():
        pytest.skip("pet-mad-xs-v1.6 checkpoint missing")
    return path


@pytest.fixture(scope="module")
def atoms(mini_xyz):
    return read(str(mini_xyz), index=0)  # periodic, 9 atoms, all spill past 11


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
    calc, _ = calculator(checkpoint, atoms, TILED)
    sizes = calc.debug_stats["tokens"]
    assert sizes["n_extra"] == len(atoms)  # every atom has an extra tile
    structure = calc._structure
    k_max = int(_truncate(calc, structure)[1].counts.max())

    def overflow(**sizers):
        resized = {**structure, **{k: jnp.zeros(n, bool) for k, n in sizers.items()}}
        return bool(_truncate(calc, resized)[1].overflow)

    assert not overflow()
    assert not overflow(token_sizer=sizes["n_tokens"] + 1, tile_sizer=sizes["n_extra"])
    assert overflow(token_sizer=sizes["n_tokens"])  # the sentinel slot must stay free
    assert overflow(tile_sizer=sizes["n_extra"] - 1)
    assert overflow(k_sel_sizer=k_max - 1)


def test_overflow_retry_resizes(checkpoint, atoms):
    """A capacity that falls short inside the skin is widened by the retry."""
    stock, moved = calculator(checkpoint, atoms, None)
    calc, tiled = calculator(checkpoint, atoms, TILED)
    n_tokens = calc.debug_stats["tokens"]["n_tokens"]
    calc._structure["token_sizer"] = jnp.zeros(n_tokens, bool)
    calc._structure["tile_sizer"] = jnp.zeros(0, bool)

    shift = np.zeros_like(atoms.positions)
    shift[0, 0] = 0.01
    for a in (moved, tiled):
        a.positions = atoms.positions + shift
    assert abs(tiled.get_potential_energy() - moved.get_potential_energy()) < 1e-10
    assert calc.debug_stats["overflow_retry"]
    assert calc._structure["token_sizer"].shape[0] > n_tokens
    np.testing.assert_allclose(tiled.get_forces(), moved.get_forces(), atol=1e-10)


def test_tile_too_narrow(checkpoint, atoms):
    with pytest.raises(ValueError, match="attention_tile >="):
        calculator(checkpoint, atoms, Express(tokens="flat", attention_tile=4))


def test_extra_path_absent(checkpoint, atoms, monkeypatch):
    """With k_sel inside one tile there is no extra tile capacity and the
    cross-tile softmax is never traced."""
    import petjax.model

    def fail(*args, **kwargs):
        raise AssertionError("extra tiles traced")

    monkeypatch.setattr(petjax.model, "_attention_tiles", fail)
    calc, _ = calculator(checkpoint, atoms, Express(tokens="flat", attention_tile=32))
    assert calc._structure["tile_sizer"].shape == (0,)


def test_forward_over_reverse(checkpoint, atoms):
    """``jax.jvp`` of ``jax.grad`` (Hessian-vector products) goes through the
    tiles and the cross-tile softmax and matches the padded layout's."""
    hvps = []
    for ex in (Express(), TILED):
        calc, _ = calculator(checkpoint, atoms, ex)
        model, structure = calc._model, calc._structure

        def energy(positions):
            return _select_and_predict(
                model,
                calc._params,
                {**structure, "positions": positions},
                calc._num_neighbors_adaptive,
                model.cutoff,
                model.cutoff_width_adaptive,
                model.adaptive_cutoff_method,
                express=ex,
            )[0]

        positions = structure["positions"]
        tangent = jax.random.normal(jax.random.key(0), positions.shape, positions.dtype)
        tangent = tangent * structure["atom_mask"][:, None]
        _, hvp = jax.jvp(jax.grad(energy), (positions,), (tangent,))
        assert np.all(np.isfinite(hvp))
        hvps.append(np.asarray(hvp))
    assert np.abs(hvps[0]).max() > 1e-3
    np.testing.assert_allclose(hvps[1], hvps[0], rtol=0, atol=1e-9)
