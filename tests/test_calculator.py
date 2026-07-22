"""Tests for UPETCalculator: shift plumbing, no_shadow, relaxation."""

import numpy as np
import jax.numpy as jnp

import copy
import warnings

import pytest
from ase.io import read

from petjax import UPETCalculator
from petjax.convert import load_checkpoint
from petjax.model import UPET
from petjax.select import (
    _n_total,
    determine_k_sel,
    get_adaptive_cutoffs,
    get_adaptive_cutoffs_solver,
    truncate_edges,
)
from petjax.structure import to_structure
from petjax.utils import edge_displacements


@pytest.fixture(scope="module")
def model_data(pet_mad_xs_checkpoint):
    params, metadata = load_checkpoint(pet_mad_xs_checkpoint)
    config = metadata["config"]
    model = UPET(**config)
    return model, params, metadata


def test_shift_applied_once(model_data, mini_xyz):
    """Shifts must be added exactly once, calculator-side, and never leak into forces."""
    model, params, metadata = model_data
    atoms = read(str(mini_xyz), index=0)

    calc_full = UPETCalculator(model, params, metadata, skin=0.5, stress=True)
    atoms.calc = calc_full
    e_full = atoms.get_potential_energy()
    f_full = atoms.get_forces().copy()

    meta_no_shifts = copy.deepcopy(metadata)
    meta_no_shifts["shifts"] = {int(z): 0.0 for z in metadata["shifts"]}
    calc_raw = UPETCalculator(model, params, meta_no_shifts, skin=0.5, stress=True)
    atoms.calc = calc_raw
    e_raw = atoms.get_potential_energy()
    f_raw = atoms.get_forces().copy()

    expected_offset = sum(
        float(metadata["shifts"][int(z)]) for z in atoms.get_atomic_numbers()
    )
    assert abs((e_full - e_raw) - expected_offset) < 1e-10, (
        f"shift applied wrong: diff={e_full - e_raw:.6f} expected={expected_offset:.6f}"
    )
    # Shifts are position-constants; forces must be bit-identical.
    np.testing.assert_array_equal(f_full, f_raw)


def test_shift_fp64_precision(model_data, mini_xyz):
    """Calculator-side shift sum must be fp64-accurate vs naive Python sum."""
    model, params, metadata = model_data
    atoms = read(str(mini_xyz), index=0)

    calc = UPETCalculator(model, params, metadata, skin=0.5, stress=True)
    atoms.calc = calc
    atoms.get_potential_energy()  # triggers _build_structure → _shift_offset

    expected = sum(float(metadata["shifts"][int(z)]) for z in atoms.get_atomic_numbers())
    assert calc._shift_offset == expected, (
        f"shift_offset={calc._shift_offset} expected={expected}"
    )


def test_fp64_forces_finite(pet_mad_xs_checkpoint, mini_xyz):
    """Regression for #4: fp64 forces were NaN due to the JAX-internal fp32
    softmax overflow when masked attention rows filled logits with
    ``-0.7 * finfo(fp64).max = -1.26e308`` — casts to ``-inf`` in fp32, which
    softmaxes to NaN. Fp32 was unaffected because ``-0.7 * finfo(fp32).max``
    stays in fp32 range. Any full-mask row (padded atom) would trip it; must
    stay finite across stress × no_shadow combinations.
    """
    atoms = read(str(mini_xyz), index=0)
    for stress in (False, True):
        for no_shadow in (False, True):
            calc = UPETCalculator.from_checkpoint(
                str(pet_mad_xs_checkpoint),
                default_dtype="float64",
                stress=stress,
                no_shadow=no_shadow,
            )
            atoms.calc = calc
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            assert np.isfinite(e), f"energy NaN: stress={stress} no_shadow={no_shadow}"
            assert np.all(np.isfinite(f)), (
                f"forces contain NaN: stress={stress} no_shadow={no_shadow}"
            )


def test_no_shadow_changes_forces_slightly(model_data, mini_xyz):
    """no_shadow=True must give finite, slightly different forces.

    Cutting the gradient through the adaptive cutoff drops "shadow forces"
    that contribute weakly to the total force. The values must remain finite
    and close to (but not identical to) the full-mode result. The energy is
    bit-identical because no_shadow only affects gradient propagation.
    """
    model, params, metadata = model_data
    atoms = read(str(mini_xyz), index=0)

    calc_full = UPETCalculator(model, params, metadata, skin=0.5, stress=True)
    atoms.calc = calc_full
    e_full = atoms.get_potential_energy()
    f_full = atoms.get_forces().copy()

    calc_noshd = UPETCalculator(
        model,
        params,
        metadata,
        skin=0.5,
        stress=True,
        no_shadow=True,
    )
    atoms.calc = calc_noshd
    e_noshd = atoms.get_potential_energy()
    f_noshd = atoms.get_forces().copy()

    # Energy is bit-identical (no_shadow only affects gradients)
    assert e_full == e_noshd, f"energy changed: {e_full} -> {e_noshd}"

    # Forces are finite
    assert np.all(np.isfinite(f_noshd))

    # Forces differ by a small but nonzero amount on at least one atom
    diff = float(np.max(np.abs(f_full - f_noshd)))
    assert diff > 0.0, "no_shadow had no effect on forces"
    # Sanity bound: shadow contribution should be small in absolute terms.
    # 0.5 eV/Å is generous but catches anything pathological.
    assert diff < 0.5, f"no_shadow force diff suspiciously large: {diff:.4f}"
    print(f"\n  max |f_full - f_no_shadow| = {diff:.2e} eV/Å")


def test_calculator_position_only_update(model_data, mini_xyz):
    """Small position changes reuse the NL, but energy/forces must still update."""
    model, params, metadata = model_data
    atoms = read(str(mini_xyz), index=0)

    calc = UPETCalculator(model, params, metadata, skin=0.5, stress=True)
    atoms.calc = calc

    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()
    pos = atoms.get_positions()
    rebuild_marker_1 = id(calc._predict_fn)

    pos2 = pos + np.random.RandomState(0).randn(*pos.shape) * 0.01
    atoms.set_positions(pos2)
    e2 = atoms.get_potential_energy()
    f2 = atoms.get_forces()
    rebuild_marker_2 = id(calc._predict_fn)

    assert rebuild_marker_1 == rebuild_marker_2, (
        "predict_fn was rebuilt on small position update"
    )
    assert abs(e1 - e2) < 5.0
    # Guard against the "cache never invalidates" regression: if the calculator
    # reuses the prior positions, e1/e2 and f1/f2 come back bit-identical.
    assert e1 != e2, "energy did not change with positions — stale cache?"
    assert not np.array_equal(f1, f2), "forces did not change with positions — stale cache?"


def test_calculator_cell_relaxation(model_data, mini_xyz):
    """End-to-end: BFGS + FrechetCellFilter over many steps on a small periodic
    structure. Validates that (i) energy decreases monotonically (well, doesn't
    increase past a floor), (ii) forces stay finite throughout, and (iii) NL
    rebuild count stays well below the step count — the whole point of the
    Verlet-skin + bucketed-shapes machinery.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS

    # Frame 8 in the mini fixture: 13 atoms, C/Eu/H/O, periodic. Small enough
    # to relax in a few seconds; the H sublattice moves enough to exercise the
    # adaptive-cutoff path without overflowing buckets.
    atoms = read(str(mini_xyz), index=8)

    model, params, metadata = model_data
    calc = UPETCalculator(model, params, metadata, skin=0.5, stress=True)
    atoms.calc = calc

    rebuild_markers = [id(calc._predict_fn)]

    def track_rebuild():
        if id(calc._predict_fn) != rebuild_markers[-1]:
            rebuild_markers.append(id(calc._predict_fn))

    e0 = atoms.get_potential_energy()
    track_rebuild()

    opt = BFGS(FrechetCellFilter(atoms), logfile=None)
    max_steps = 30
    for _ in range(max_steps):
        opt.step()
        track_rebuild()

    e_final = atoms.get_potential_energy()
    f_final = atoms.get_forces()

    assert e_final < e0 + 1e-3, f"energy did not decrease: {e0:.4f} -> {e_final:.4f}"
    assert np.all(np.isfinite(f_final))
    n_rebuilds = len(rebuild_markers) - 1
    assert n_rebuilds < max_steps // 2, (
        f"too many rebuilds: {n_rebuilds} in {max_steps} steps"
    )


def test_pair_cutoffs_none_uses_static_cutoff(model_data, mini_xyz):
    """The model's ``pair_cutoffs=None`` fallback must keep working.

    ``UPETCalculator`` always passes adaptive per-pair cutoffs, so this branch
    is dead in the inference pipeline — but it is the path a model trained
    without the adaptive-cutoff mechanism relies on, falling back to the static
    ``self.cutoff``. Verify it runs, yields finite per-atom energies, and is
    bit-identical to passing a constant ``pair_cutoffs`` array == ``self.cutoff``.
    """
    model, params, metadata = model_data
    config = metadata["config"]
    atoms = read(str(mini_xyz), index=0)

    # Build the flat NL and the packed [N_padded, k_sel] selected layout via
    # the public truncate_edges seam, to get valid model inputs.
    structure = to_structure(atoms, model.cutoff, skin=0.5)
    N_padded = structure["positions"].shape[0]
    k_sel, _ = determine_k_sel(
        structure,
        config["num_neighbors_adaptive"],
        model.cutoff,
        model.cutoff_width_adaptive,
    )
    R_ij = edge_displacements(
        structure["positions"],
        structure["centers"],
        structure["others"],
        structure["cell_shifts"],
        structure["cell"],
    )
    truncated, _ = truncate_edges(
        R_ij,
        structure["centers"],
        structure["others"],
        structure["reverse"],
        structure["pair_mask"],
        structure["species"],
        structure["atom_mask"],
        k_sel,
        config["num_neighbors_adaptive"],
        model.cutoff,
        model.cutoff_width_adaptive,
    )
    pair_cutoffs_sel = truncated.pop("pair_cutoffs")

    out_none = model.apply(params, **truncated)  # pair_cutoffs defaults to None
    out_static = model.apply(
        params, **truncated, pair_cutoffs=jnp.full_like(pair_cutoffs_sel, model.cutoff)
    )
    out_adaptive = model.apply(params, **truncated, pair_cutoffs=pair_cutoffs_sel)

    # Runs, right shape, finite.
    assert out_none.shape == (N_padded,)
    assert np.all(np.isfinite(out_none))

    # pair_cutoffs=None is exactly the static-cutoff path.
    np.testing.assert_array_equal(np.asarray(out_none), np.asarray(out_static))

    # Sanity: pair_cutoffs is not ignored — adaptive cutoffs change the output.
    assert not np.array_equal(np.asarray(out_none), np.asarray(out_adaptive)), (
        "adaptive pair_cutoffs had no effect — test would be vacuous"
    )


def test_load_checkpoint_upgrades_legacy_config(pet_mad_xs_checkpoint):
    """The test asset predates the adaptive-selection hypers; ``load_checkpoint``
    must fill them following metatrain's own migration rules (grid method,
    adaptive width := shared ``cutoff_width``)."""
    _, metadata = load_checkpoint(pet_mad_xs_checkpoint)
    config = metadata["config"]
    assert config["adaptive_cutoff_method"] == "grid"
    assert config["cutoff_width_adaptive"] == config["cutoff_width"]


def test_cutoff_width_adaptive_threads_through(model_data, mini_xyz):
    """Regression for #14: the adaptive selection must run at
    ``cutoff_width_adaptive``, not the final-taper ``cutoff_width``.

    A distinct adaptive width must shift the per-atom cutoffs (and hence the
    prediction) end-to-end through the calculator (k_sel sizing and the in-JIT
    selection), while restating the loaded config's value changes nothing.
    """
    model, params, metadata = model_data
    atoms = read(str(mini_xyz), index=0)

    def energy_with(width_adaptive):
        config = {**metadata["config"], "cutoff_width_adaptive": width_adaptive}
        calc = UPETCalculator(UPET(**config), params, metadata, skin=0.5, stress=False)
        atoms.calc = calc
        return atoms.get_potential_energy()

    loaded_width = metadata["config"]["cutoff_width_adaptive"]

    calc = UPETCalculator(model, params, metadata, skin=0.5, stress=False)
    atoms.calc = calc
    e_loaded = atoms.get_potential_energy()

    assert energy_with(loaded_width) == e_loaded, "restated width must be bit-identical"
    assert energy_with(2.0 * loaded_width) != e_loaded, (
        "adaptive width had no effect — not threaded through?"
    )


def test_grid_selection_reproduces_issue_14_table():
    """Pin the numbers from issue #14's evidence table: single-atom environment,
    40 neighbors evenly spaced 2.0–7.5 Å, target 16. The selection width shifts
    the per-atom cutoff from 4.299 Å (width 1) to 4.684 Å (width 2)."""
    r_ij = jnp.linspace(2.0, 7.5, 40)
    centers = jnp.zeros(40, dtype=int)
    mask = jnp.ones(40, dtype=bool)

    cut_w1 = get_adaptive_cutoffs(centers, r_ij, mask, 16, 1, 7.5, 1.0)
    cut_w2 = get_adaptive_cutoffs(centers, r_ij, mask, 16, 1, 7.5, 2.0)
    np.testing.assert_allclose(float(cut_w1[0]), 4.299, atol=2e-3)
    np.testing.assert_allclose(float(cut_w2[0]), 4.684, atol=2e-3)


def test_solver_finds_root_and_attaches_gradients():
    """The solver's per-atom cutoffs must be roots of the smoothed neighbor
    count (``n_total(r) = target``) wherever unclamped, with masked padded
    pairs excluded, and gradients must flow to the distances through the
    implicit-function-theorem step (finite, nonzero)."""
    import jax

    rng = np.random.RandomState(7)
    counts = [40, 25, 8, 60]
    num_atoms = len(counts)
    cutoff, width, target = 7.5, 1.0, 16
    centers = jnp.array(np.concatenate([np.full(c, i) for i, c in enumerate(counts)]))
    r_ij = jnp.array(np.sort(rng.uniform(0.8, cutoff, size=int(np.sum(counts)))))
    # Append masked padded pairs — they must not shift anything.
    pad = 17
    centers_p = jnp.concatenate([centers, jnp.full(pad, num_atoms - 1)])
    r_p = jnp.concatenate([r_ij, jnp.full(pad, 1e-7)])
    mask_p = jnp.concatenate([jnp.ones(r_ij.shape[0], bool), jnp.zeros(pad, bool)])

    def solve(c, d, m):
        return get_adaptive_cutoffs_solver(c, d, m, target, num_atoms, cutoff, width)

    cuts = solve(centers, r_ij, jnp.ones(r_ij.shape[0], bool))
    cuts_padded = solve(centers_p, r_p, mask_p)
    np.testing.assert_allclose(np.asarray(cuts), np.asarray(cuts_padded), atol=1e-6)

    residual = (
        _n_total(
            cuts,
            r_ij,
            centers,
            jnp.ones(r_ij.shape[0], bool),
            num_atoms,
            width,
            1 / cutoff,
            target,
        )
        - target
    )
    unclamped = (np.asarray(cuts) > cutoff / 16 + 1e-6) & (np.asarray(cuts) < cutoff - 1e-6)
    assert unclamped.any(), "test env produced no unclamped atoms — vacuous"
    np.testing.assert_allclose(np.asarray(residual)[unclamped], 0.0, atol=1e-4)

    grad = jax.grad(lambda d: jnp.sum(solve(centers, d, jnp.ones(r_ij.shape[0], bool))))(
        r_ij
    )
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(jnp.abs(grad).max()) > 0.0, "IFT step attached no gradient"


def test_adaptive_cutoff_solver_end_to_end(model_data, mini_xyz):
    """A solver-method model must run through the whole calculator pipeline
    (k_sel sizing, in-JIT selection, autodiff forces) and give results close
    to — but not identical with — the grid method it supersedes."""
    model, params, metadata = model_data
    atoms = read(str(mini_xyz), index=0)

    solver_model = UPET(**{**metadata["config"], "adaptive_cutoff_method": "solver"})
    calc = UPETCalculator(solver_model, params, metadata, skin=0.5, stress=True)
    atoms.calc = calc
    e_solver = atoms.get_potential_energy()
    f_solver = atoms.get_forces().copy()
    assert np.isfinite(e_solver)
    assert np.all(np.isfinite(f_solver))

    calc_grid = UPETCalculator(model, params, metadata, skin=0.5, stress=True)
    atoms.calc = calc_grid
    e_grid = atoms.get_potential_energy()

    assert e_solver != e_grid, "solver had no effect — not threaded through?"
    # The two methods approximate the same root; a large gap means a broken port.
    assert abs(e_solver - e_grid) < 0.5, f"solver vs grid gap: {e_solver - e_grid:.4f} eV"


def test_debug_stats_and_cutoff_override_warning(model_data, mini_xyz):
    """debug_stats is populated on a rebuild; a too-small cutoff_override warns."""
    model, params, metadata = model_data
    atoms = read(str(mini_xyz), index=0)
    trained = metadata["config"]["cutoff"]

    # No override: stats populated, no cutoff warning.
    calc = UPETCalculator(model, params, metadata, skin=0.5)
    atoms.calc = calc
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        atoms.get_potential_energy()
    s = calc.debug_stats
    assert s["n_atoms"] == len(atoms)
    assert s["N_padded"] >= s["n_atoms"]
    assert s["n_pair_padded"] >= s["n_pair_raw"] > 0
    assert s["k_sel_padded"] >= s["k_sel_actual"] >= 1
    assert 0 < s["max_selected_cutoff"] <= trained
    assert not [w for w in rec if "adaptive selection reaches" in str(w.message)]

    # Override well below the measured reach: warning fires.
    too_small = s["max_selected_cutoff"] / 2
    calc2 = UPETCalculator(model, params, metadata, skin=0.5, cutoff_override=too_small)
    atoms.calc = calc2
    with pytest.warns(UserWarning, match="adaptive selection reaches"):
        atoms.get_potential_energy()
