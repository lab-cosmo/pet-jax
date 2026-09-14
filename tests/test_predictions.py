"""Cross-check petjax predictions against reference predictions.

Default (CI) mode compares against `tests/assets/predictions/test_mini_*.npz`
for every release in `conftest.MINI_RELEASES` — the PET-MAD v1.5 and v1.6
pet-mad-xs checkpoints, whose readout heads are named differently. Extended
mode additionally runs the full test_s / test_m / test_l datasets for both
pet-mad-xs and pet-mad-s; those assets are local-only and skipped when
missing.
"""

import numpy as np

from pathlib import Path

import pytest
from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress

from petjax import UPETCalculator

ASSETS = Path(__file__).parent / "assets"


# -- reference parsing --


def load_reference(path, structures):
    """Reference predictions for `structures`, one dict per structure.

    References hold numbers only — energies, forces, stresses — never geometry:
    the structures come from the dataset file. `natoms` is what ties the two
    together, so a reference built for a different dataset fails here instead
    of silently lining up against the wrong frames.
    """
    data = np.load(path)
    natoms = data["natoms"]
    expected = np.array([len(atoms) for atoms in structures])
    if not np.array_equal(natoms, expected):
        raise AssertionError(
            f"{Path(path).name} does not match the dataset it is compared "
            f"against: atom counts {natoms.tolist()} vs {expected.tolist()}"
        )

    forces = np.split(data["forces"], np.cumsum(natoms)[:-1])
    return [
        {"energy": float(energy), "forces": per_structure, "stress": stress}
        for energy, per_structure, stress in zip(data["energy"], forces, data["stress"])
    ]


# -- calculator loading --

_CALC_CACHE = {}


def get_calc(ckpt_dir, direct=False):
    key = (str(ckpt_dir), direct)
    if key not in _CALC_CACHE:
        extra = {"direct_forces": True, "direct_stress": True} if direct else {}
        _CALC_CACHE[key] = UPETCalculator.from_checkpoint(
            str(ckpt_dir), stress=True, **extra
        )
    return _CALC_CACHE[key]


# -- inference --


def run_single(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    if atoms.pbc.any():
        stress = voigt_6_to_full_3x3_stress(atoms.get_stress())
    else:
        stress = None
    return energy, forces, stress


def _assert_energies(calc, structures, ref):
    max_diff_per_atom = 0.0
    for atoms, ref_data in zip(structures, ref):
        energy, _, _ = run_single(calc, atoms)
        dpa = abs(energy - ref_data["energy"]) / len(atoms)
        max_diff_per_atom = max(max_diff_per_atom, dpa)
    assert max_diff_per_atom < 1e-3, f"max diff/atom = {max_diff_per_atom:.2e}"


def _assert_forces(calc, structures, ref):
    worst_maxae = 0.0
    for atoms, ref_data in zip(structures, ref):
        _, forces, _ = run_single(calc, atoms)
        worst_maxae = max(worst_maxae, float(np.max(np.abs(forces - ref_data["forces"]))))
    assert worst_maxae < 0.01, f"worst force maxAE = {worst_maxae:.2e}"


def _assert_stress(calc, structures, ref):
    worst_maxae = 0.0
    n_tested = 0
    for atoms, ref_data in zip(structures, ref):
        if not atoms.pbc.any():
            continue
        _, _, stress = run_single(calc, atoms)
        worst_maxae = max(worst_maxae, float(np.max(np.abs(stress - ref_data["stress"]))))
        n_tested += 1
    if n_tested > 0:
        assert worst_maxae < 5e-3, f"worst stress maxAE = {worst_maxae:.2e}"


# -- CI (mini) tests --


def test_mini_energies(mini_release, mini_xyz):
    checkpoint, conservative, _ = mini_release
    structures = read(str(mini_xyz), index=":")
    ref = load_reference(conservative, structures)
    _assert_energies(get_calc(checkpoint), structures, ref)


def test_mini_forces(mini_release, mini_xyz):
    checkpoint, conservative, _ = mini_release
    structures = read(str(mini_xyz), index=":")
    ref = load_reference(conservative, structures)
    _assert_forces(get_calc(checkpoint), structures, ref)


def test_mini_stress(mini_release, mini_xyz):
    checkpoint, conservative, _ = mini_release
    structures = read(str(mini_xyz), index=":")
    ref = load_reference(conservative, structures)
    _assert_stress(get_calc(checkpoint), structures, ref)


def test_reference_dataset_mismatch_detected(tmp_path, mini_xyz):
    """The atom counts are the only tie between a reference and its dataset."""
    structures = read(str(mini_xyz), index=":")
    natoms = np.array([len(atoms) for atoms in structures])
    natoms[0] += 1

    path = tmp_path / "mismatched.npz"
    np.savez(
        path,
        energy=np.zeros(len(structures)),
        forces=np.zeros((natoms.sum(), 3)),
        stress=np.zeros((len(structures), 3, 3)),
        natoms=natoms,
    )
    with pytest.raises(AssertionError, match="does not match the dataset"):
        load_reference(path, structures)


# -- extended (local) tests: full test_{s,m,l} × pet-mad-{xs,s} matrix --

EXTENDED_COMBOS = [
    ("pet-mad-xs", "test_s"),
    ("pet-mad-s", "test_s"),
    ("pet-mad-xs", "test_m"),
    ("pet-mad-s", "test_m"),
    ("pet-mad-xs", "test_l"),
    ("pet-mad-s", "test_l"),
]


def _extended_combo_available(model_name, dataset):
    ckpt = ASSETS / "checkpoints" / model_name / "model.msgpack"
    pred = ASSETS / "predictions" / f"{dataset}_{model_name}.npz"
    ds = ASSETS / f"{dataset}.xyz"
    return ckpt.exists() and pred.exists() and ds.exists()


@pytest.fixture(
    params=[
        pytest.param(combo, id=f"{combo[0]}/{combo[1]}", marks=pytest.mark.extended)
        for combo in EXTENDED_COMBOS
    ]
)
def extended_combo(request):
    model_name, dataset = request.param
    if not _extended_combo_available(model_name, dataset):
        pytest.skip(f"Missing extended files for {model_name}/{dataset}")
    ckpt_dir = ASSETS / "checkpoints" / model_name
    calc = get_calc(ckpt_dir)
    structures = read(str(ASSETS / f"{dataset}.xyz"), index=":")
    ref = load_reference(ASSETS / "predictions" / f"{dataset}_{model_name}.npz", structures)
    return calc, ref, structures


def test_extended_energies(extended_combo):
    calc, ref, structures = extended_combo
    _assert_energies(calc, structures, ref)


def test_extended_forces(extended_combo):
    calc, ref, structures = extended_combo
    _assert_forces(calc, structures, ref)


def test_extended_stress(extended_combo):
    calc, ref, structures = extended_combo
    _assert_stress(calc, structures, ref)


# -- non-conservative heads --


def test_mini_direct_energy(mini_release, mini_xyz):
    checkpoint, _, direct = mini_release
    structures = read(str(mini_xyz), index=":")
    ref = load_reference(direct, structures)
    _assert_energies(get_calc(checkpoint, direct=True), structures, ref)


def test_mini_direct_forces(mini_release, mini_xyz):
    checkpoint, _, direct = mini_release
    structures = read(str(mini_xyz), index=":")
    ref = load_reference(direct, structures)
    _assert_forces(get_calc(checkpoint, direct=True), structures, ref)


def test_mini_direct_stress(mini_release, mini_xyz):
    checkpoint, _, direct = mini_release
    structures = read(str(mini_xyz), index=":")
    ref = load_reference(direct, structures)
    _assert_stress(get_calc(checkpoint, direct=True), structures, ref)
