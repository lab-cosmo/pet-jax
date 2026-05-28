"""Cross-check petjax predictions against reference predictions.

Default (CI) mode compares against `tests/assets/predictions/test_mini_*.xyz`
using the LFS-tracked pet-mad-xs checkpoint. Extended mode additionally runs
the full test_s / test_m / test_l datasets for both pet-mad-xs and
pet-mad-s; those assets are local-only and skipped when missing.
"""

import numpy as np

import re
from pathlib import Path

import pytest
from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress

from petjax import UPETCalculator

ASSETS = Path(__file__).parent / "assets"


# -- reference parsing --


def _parse_properties(prop_str):
    fields = prop_str.split(":")
    columns = []
    i = 0
    while i < len(fields):
        columns.append((fields[i], fields[i + 1], int(fields[i + 2])))
        i += 3
    return columns


def _forces_column_offset(prop_str):
    offset = 0
    for name, _dtype, count in _parse_properties(prop_str):
        if name == "forces":
            return offset
        offset += count
    raise ValueError("No 'forces' in Properties")


def load_reference(path):
    with open(path) as f:
        lines = f.readlines()

    results = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        natoms = int(line)
        comment = lines[i + 1]

        energy = float(re.search(r"energy=([^\s]+)", comment).group(1))

        stress_match = re.search(r'stress="([^"]+)"', comment)
        stress = np.array(stress_match.group(1).split(), dtype=np.float64).reshape(3, 3)

        prop_match = re.search(r"Properties=([^\s]+)", comment)
        col_offset = _forces_column_offset(prop_match.group(1))

        forces = np.zeros((natoms, 3))
        for j in range(natoms):
            parts = lines[i + 2 + j].split()
            forces[j] = [
                float(parts[col_offset]),
                float(parts[col_offset + 1]),
                float(parts[col_offset + 2]),
            ]

        results.append({"energy": energy, "forces": forces, "stress": stress})
        i += 2 + natoms

    return results


# -- calculator loading --

_CALC_CACHE = {}


def get_calc(ckpt_dir):
    key = str(ckpt_dir)
    if key not in _CALC_CACHE:
        _CALC_CACHE[key] = UPETCalculator.from_checkpoint(str(ckpt_dir), stress=True)
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


def test_mini_energies(pet_mad_xs_checkpoint, mini_xyz, mini_predictions_xs):
    calc = get_calc(pet_mad_xs_checkpoint)
    structures = read(str(mini_xyz), index=":")
    ref = load_reference(mini_predictions_xs)
    _assert_energies(calc, structures, ref)


def test_mini_forces(pet_mad_xs_checkpoint, mini_xyz, mini_predictions_xs):
    calc = get_calc(pet_mad_xs_checkpoint)
    structures = read(str(mini_xyz), index=":")
    ref = load_reference(mini_predictions_xs)
    _assert_forces(calc, structures, ref)


def test_mini_stress(pet_mad_xs_checkpoint, mini_xyz, mini_predictions_xs):
    calc = get_calc(pet_mad_xs_checkpoint)
    structures = read(str(mini_xyz), index=":")
    ref = load_reference(mini_predictions_xs)
    _assert_stress(calc, structures, ref)


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
    pred = ASSETS / "predictions" / f"{dataset}_{model_name}.xyz"
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
    ref = load_reference(ASSETS / "predictions" / f"{dataset}_{model_name}.xyz")
    structures = read(str(ASSETS / f"{dataset}.xyz"), index=":")
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
