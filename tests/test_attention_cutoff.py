"""The attention's cutoff modes (`Attention.cutoff`).

"smooth" follows `metatrain` PR #1356 (5025edb). Its weight function is
compared, with its gradients, against `smooth_cutoff_attention` of that commit
on stored random inputs, to rounding. The whole-model comparison on the mini
set (PET-MAD-xs v1.6 weights with `smooth_attention_cutoff` switched on) still
holds the reference of the earlier #1342 and is skipped until it is
regenerated.

The other tests are properties every mode has to keep, checked on the
`Attention` module with random parameters.
"""

import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path

import pytest
from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress

from petjax import UPET, UPETCalculator, load_checkpoint
from petjax.model import Attention, smooth_cutoff_weights

ASSETS = Path(__file__).parent / "assets"
MODES = ("bias", "self_exempt", "smooth", "sigmoid", "sigmoid_mean")
SMOOTH_TOL = {"energy": 2e-6, "forces": 2e-4, "stress": 5e-6}


@pytest.fixture
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


# -- "smooth" against metatrain --


def test_smooth_cutoff_weights_match_metatrain(x64):
    data = np.load(ASSETS / "predictions" / "smooth_cutoff_weights.npz")
    logits, factors = jnp.asarray(data["logits"]), jnp.asarray(data["factors"])

    def probed(logits, factors):
        return jnp.sum(smooth_cutoff_weights(logits, factors) * data["probe"])

    d_logits, d_factors = jax.grad(probed, argnums=(0, 1))(logits, factors)
    np.testing.assert_allclose(
        smooth_cutoff_weights(logits, factors), data["weights"], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(d_logits, data["d_logits"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(d_factors, data["d_factors"], rtol=1e-12, atol=1e-14)


@pytest.mark.skip(reason="reference is from metatrain #1342; not yet regenerated for #1356")
def test_smooth_model_matches_metatrain(x64, assets_dir, mini_xyz):
    checkpoint = assets_dir / "checkpoints" / "pet-mad-xs-v1.6"
    if not (checkpoint / "model.msgpack").exists():
        pytest.skip("pet-mad-xs-v1.6 checkpoint missing; run tox -e fetch-checkpoints")
    reference = np.load(assets_dir / "predictions" / "test_mini_pet-mad-xs-v1.6_smooth.npz")
    frames = read(str(mini_xyz), index=":")
    assert np.array_equal(reference["natoms"], [len(atoms) for atoms in frames])
    forces = np.split(reference["forces"], np.cumsum(reference["natoms"])[:-1])

    params, metadata = load_checkpoint(checkpoint)
    model = UPET(**{**metadata["config"], "attention_cutoff": "smooth"})
    calculator = UPETCalculator(
        model,
        params,
        metadata,
        default_dtype="float64",
        matmul_precision="highest",
    )
    for i, atoms in enumerate(frames):
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        assert abs(energy - reference["energy"][i]) / len(atoms) < SMOOTH_TOL["energy"]
        assert np.abs(atoms.get_forces() - forces[i]).max() < SMOOTH_TOL["forces"]
        if atoms.pbc.all():
            stress = voigt_6_to_full_3x3_stress(atoms.get_stress())
            assert np.abs(stress - reference["stress"][i]).max() < SMOOTH_TOL["stress"]


# -- properties of the weight function --


def _capped(scores, cap=5.0):
    return cap * jnp.tanh(scores / cap)


def test_single_tail_key_receives_its_factor_times_the_full_share(x64):
    rng = np.random.default_rng(0)
    scores = jnp.asarray(5.0 * rng.normal(size=(4, 1, 7)))
    for factor in (1e-6, 1e-2, 0.5):
        factors = jnp.ones((4, 1, 7)).at[:, :, 3].set(factor)
        full = jax.nn.softmax(_capped(scores), axis=-1)
        weights = smooth_cutoff_weights(scores, factors)
        np.testing.assert_allclose(weights[..., 3], factor * full[..., 3], rtol=1e-12)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, rtol=1e-12)


def test_keys_inside_the_cutoff_keep_their_softmax_weight(x64):
    scores = jnp.asarray(np.random.default_rng(1).normal(size=(3, 2, 1, 6)))
    np.testing.assert_allclose(
        smooth_cutoff_weights(scores, jnp.ones((3, 1, 1, 6))),
        jax.nn.softmax(_capped(scores), axis=-1),
        rtol=1e-12,
    )


def test_row_without_keys_is_zero_with_finite_gradient(x64):
    scores = jnp.asarray(np.random.default_rng(2).normal(size=(1, 5)))
    factors = jnp.zeros((1, 5))
    assert np.all(smooth_cutoff_weights(scores, factors) == 0.0)
    gradients = jax.grad(lambda s, f: smooth_cutoff_weights(s, f).sum(), argnums=(0, 1))(
        scores, factors
    )
    assert all(np.all(np.isfinite(g)) for g in gradients)


# -- properties of every mode --


def _attention(mode, tokens=6, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(3, tokens, 16)))
    module = Attention(num_heads=2, cutoff=mode, gate_bias=-1.0)
    ones = jnp.ones((3, tokens))
    params = module.init(jax.random.PRNGKey(seed), x, ones, ones > 0)
    scale = jax.tree.map(lambda p: 3.0 * p, params)  # scores far from uniform
    return lambda cutoffs, mask: module.apply(scale, x, cutoffs, mask), rng


@pytest.mark.parametrize("mode", MODES)
def test_key_at_the_cutoff_equals_absent_key(x64, mode):
    """A key whose factor reaches zero leaves the other tokens as if masked."""
    apply, rng = _attention(mode)
    cutoffs = jnp.asarray(rng.uniform(0.1, 1.0, size=(3, 6))).at[:, 0].set(1.0)
    mask = jnp.ones((3, 6), dtype=bool)
    others = np.array([0, 1, 2, 3, 5])

    absent = apply(cutoffs.at[:, 4].set(0.0), mask.at[:, 4].set(False))
    present = apply(cutoffs.at[:, 4].set(0.0), mask)
    # the softmax modes floor the factor at 1e-15 rather than at zero, which a
    # strongly preferred key multiplies up
    np.testing.assert_allclose(present[:, others], absent[:, others], rtol=0, atol=1e-8)


@pytest.mark.parametrize("mode", MODES)
def test_fully_masked_rows_have_finite_gradients(x64, mode):
    """Padded atoms: every token masked, all factors zero."""
    apply, _ = _attention(mode)
    mask = jnp.ones((3, 6), dtype=bool).at[1].set(False)
    cutoffs = jnp.ones((3, 6)).at[1].set(0.0)
    gradient = jax.grad(lambda c: apply(c, mask)[jnp.array([0, 2])].sum())(cutoffs)
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("mode", ("self_exempt", "smooth", "sigmoid", "sigmoid_mean"))
def test_token_does_not_depend_on_its_own_factor_through_itself(x64, mode):
    """With only itself and the central token as keys, a neighbor token's
    output is independent of its own factor; under "bias" it is not."""
    mask = jnp.ones((3, 2), dtype=bool)

    def neighbor_row(apply, factor):
        return apply(jnp.ones((3, 2)).at[:, 1].set(factor), mask)[:, 1]

    apply, _ = _attention(mode, tokens=2)
    np.testing.assert_allclose(
        neighbor_row(apply, 0.01), neighbor_row(apply, 1.0), rtol=1e-12
    )
    apply, _ = _attention("bias", tokens=2)
    assert np.abs(neighbor_row(apply, 0.01) - neighbor_row(apply, 1.0)).max() > 1e-3


def test_sigmoid_is_linear_in_each_factor(x64):
    apply, rng = _attention("sigmoid")
    cutoffs = jnp.asarray(rng.uniform(0.1, 1.0, size=(3, 6))).at[:, 0].set(1.0)
    mask = jnp.ones((3, 6), dtype=bool)
    at = lambda factor: apply(cutoffs.at[:, 4].set(factor), mask)
    others = np.array([0, 1, 2, 3, 5])
    np.testing.assert_allclose(
        at(0.3)[:, others],
        (0.7 * at(0.0) + 0.3 * at(1.0))[:, others],
        rtol=0,
        atol=1e-13,
    )


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown attention cutoff"):
        _attention("softmax")
