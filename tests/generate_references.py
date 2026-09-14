"""Regenerate the `metatrain` reference predictions the test suite compares against.

Runs upstream `metatrain` (never `pet-jax`) on a dataset and writes extxyz with
`energy`, `forces`, and `stress` — the format `test_predictions.load_reference`
parses. Non-periodic frames get a zero stress placeholder; the stress assertions
skip them anyway.

Two files per checkpoint, one per force/stress mode:

    test_mini_<name>.xyz          conservative (energy gradients)
    test_mini_<name>_direct.xyz   non-conservative (direct readout heads)

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

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

ASSETS = Path(__file__).parent / "assets"


def predict(calculator, frames):
    """Re-attach metatrain's predictions to copies of the input frames."""
    out = []
    for atoms in frames:
        driven = atoms.copy()
        driven.calc = calculator
        energy = driven.get_potential_energy()
        forces = driven.get_forces()
        # ASE refuses stress without periodicity; the reference format wants a
        # value on every frame, and the stress assertions skip aperiodic ones.
        stress = driven.get_stress(voigt=False) if driven.pbc.any() else np.zeros((3, 3))

        frame = atoms.copy()
        frame.calc = SinglePointCalculator(
            frame, energy=energy, forces=forces, stress=stress
        )
        out.append(frame)
    return out


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
        path = args.out_dir / f"{stem}_{args.name}{suffix}.xyz"
        write(str(path), predict(calculator, frames), format="extxyz")
        print(f"wrote {len(frames)} frames to {path}")


if __name__ == "__main__":
    main()
