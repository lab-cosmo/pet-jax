"""Regenerate the `metatrain` reference predictions the test suite compares against.

Runs upstream `metatrain` (never `pet-jax`) on a dataset and writes an `.npz` of
`energy` (n_structures,), `forces` (n_atoms_total, 3), `stress`
(n_structures, 3, 3), and `natoms` (n_structures,). Numbers only — the geometry
stays in the dataset file, and `natoms` is what lets
`test_predictions.load_reference` split the forces back up and check that the
reference belongs to the dataset it is compared against. Non-periodic frames
get a zero stress placeholder; the stress assertions skip them anyway.

Two files per checkpoint, one per force/stress mode:

    test_mini_<name>.npz          conservative (energy gradients)
    test_mini_<name>_direct.npz   non-conservative (direct readout heads)

Needs the upstream stack, not the inference one, and `metatrain >= 2026.4` to
read a PET checkpoint v16 through `metatrain`'s own model classes (2026.3 tops
out at v14). Run it from the repo root against the `.ckpt`, not the converted
`pet-jax` checkpoint:

    uv run --with 'metatrain>=2026.4' --with metatomic-torch --with ase \\
        python tests/generate_references.py <source.ckpt> pet-mad-xs-v1.6
"""

import numpy as np

import argparse
from pathlib import Path

from ase.io import read

ASSETS = Path(__file__).parent / "assets"


def predict(calculator, frames):
    """Run metatrain over `frames`, collected into the stored array layout."""
    energy, forces, stress = [], [], []
    for atoms in frames:
        driven = atoms.copy()
        driven.calc = calculator
        energy.append(driven.get_potential_energy())
        forces.append(driven.get_forces())
        # ASE refuses stress without periodicity; the reference format wants a
        # value on every frame, and the stress assertions skip aperiodic ones.
        stress.append(
            driven.get_stress(voigt=False) if driven.pbc.any() else np.zeros((3, 3))
        )

    return {
        "energy": np.asarray(energy),
        "forces": np.concatenate(forces),
        "stress": np.asarray(stress),
        "natoms": np.asarray([len(atoms) for atoms in frames]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ckpt", type=Path, help="upstream metatrain .ckpt")
    parser.add_argument(
        "name", help="checkpoint name used in the output filenames, e.g. pet-mad-xs-v1.6"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ASSETS / "test_mini.xyz",
        help="input structures (default: the mini CI dataset)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ASSETS / "predictions", help="output directory"
    )
    args = parser.parse_args(argv)

    from metatomic.torch.ase_calculator import MetatomicCalculator
    from metatrain.utils.io import load_model

    exported = load_model(str(args.ckpt)).export()
    frames = read(str(args.dataset), index=":")
    stem = args.dataset.stem

    for suffix, non_conservative in (("", False), ("_direct", True)):
        calculator = MetatomicCalculator(exported, non_conservative=non_conservative)
        path = args.out_dir / f"{stem}_{args.name}{suffix}.npz"
        np.savez_compressed(path, **predict(calculator, frames))
        print(f"wrote {len(frames)} frames to {path}")


if __name__ == "__main__":
    main()
