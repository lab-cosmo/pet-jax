"""Regenerate the `metatrain` reference predictions the test suite compares against.

Runs upstream `metatrain` (never `pet-jax`) on a dataset and writes an `.npz` of
`energy` (n_structures,), `forces` (n_atoms_total, 3), `stress`
(n_structures, 3, 3), and `natoms` (n_structures,). Numbers only — the geometry
stays in the dataset file, and `natoms` is what lets
`test_predictions.load_reference` split the forces back up and check that the
reference belongs to the dataset it is compared against. The model runs in
fp64 so the references are exact to ~1e-14 and the fp64 tests can pin
`pet-jax` to them bit for bit; fp32 tests compare to the same files. Frames
that are not periodic in all three directions get a zero stress placeholder,
because `metatrain` zeroes its direct stress head there; the stress assertions
skip such frames in both modes.

Two files per checkpoint, one per force/stress mode:

    test_mini_<name>.npz          conservative (energy gradients)
    test_mini_<name>_direct.npz   non-conservative (direct readout heads)

Needs the upstream stack, not the inference one, with a `metatrain` that has
the bisection "solver" (metatensor/metatrain#1331, not yet in a release; pin
its merge commit). Run it from the repo root against the `.ckpt`, not the
converted `pet-jax` checkpoint:

    uv run --with 'metatrain @ git+https://github.com/metatensor/metatrain@1fc02b10cd4914e4c99487a45ff802c722ecd20c' \\
        --with metatomic-torch --with ase \\
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
        stress.append(
            driven.get_stress(voigt=False) if driven.pbc.all() else np.zeros((3, 3))
        )

    return {
        "energy": np.asarray(energy),
        "forces": np.concatenate(forces),
        "stress": np.asarray(stress),
        "natoms": np.asarray([len(atoms) for atoms in frames]),
    }


def restore_fp64_scales_and_shifts(model, checkpoint):
    """Put the checkpoint's fp64 scales and composition shifts back into `model`.

    `metatrain` fits both in fp64 and stores them as serialized TensorMaps, but
    loading casts them to the dtype the model was trained in, and for an fp32
    checkpoint `.to(torch.float64)` cannot bring the lost digits back (a global
    factor of ~1e-8 and a per-species energy offset of ~1e-7 eV). `pet-jax`
    keeps the fp64 values, so the references must as well.
    """
    import torch
    from metatensor.torch import load_buffer

    checkpoint = checkpoint.get("wrapped_model_checkpoint", checkpoint)
    state = checkpoint["best_model_state_dict"] or checkpoint["model_state_dict"]

    def restore(tensor_map, key):
        stored = load_buffer(state[key])
        with torch.no_grad():
            for labels, block in tensor_map.items():
                block.values[:] = stored.block(labels).values

    # Checkpoints before the per-target / per-property split (ckpt v13) carry
    # only the combined scale; upstream's upgrade then sets the per-target
    # scale to it and the per-property one to 1, which needs no restoring.
    scaler = model.scaler.model
    for target in scaler.scales:
        combined = f"scaler.{target}_scaler_buffer"
        restore(scaler.scales[target], combined)
        per_target = f"scaler.{target}_per_target_scaler_buffer"
        restore(
            scaler.per_target_scales[target],
            per_target if per_target in state else combined,
        )
        per_property = f"scaler.{target}_per_property_scaler_buffer"
        if per_property in state:
            restore(scaler.per_property_scales[target], per_property)
    for index, additive in enumerate(model.additive_models):
        for target in additive.model.weights:
            restore(
                additive.model.weights[target],
                f"additive_models.{index}.{target}_composition_buffer",
            )


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

    import torch
    from metatomic.torch.ase_calculator import MetatomicCalculator
    from metatrain.utils.io import load_model

    model = load_model(str(args.ckpt))
    # PET-MAD ships inside an LLPR wrapper whose export is fp32-only; the
    # wrapped PET itself exports at any dtype.
    if type(model.core).__name__ == "LLPRUncertaintyModel":
        model = model.core.model
    model = model.to(torch.float64)
    restore_fp64_scales_and_shifts(
        model, torch.load(args.ckpt, weights_only=False, map_location="cpu")
    )
    exported = model.export()
    frames = read(str(args.dataset), index=":")
    stem = args.dataset.stem

    # Direct mode reads whichever non-conservative heads the checkpoint has;
    # PET-OMol ships a force head only, so its stress stays the strain derivative.
    outputs = exported.capabilities().outputs
    direct = True if "non_conservative_stress" in outputs else "forces"

    for suffix, non_conservative in (("", False), ("_direct", direct)):
        calculator = MetatomicCalculator(exported, non_conservative=non_conservative)
        path = args.out_dir / f"{stem}_{args.name}{suffix}.npz"
        np.savez_compressed(path, **predict(calculator, frames))
        print(f"wrote {len(frames)} frames to {path}")


if __name__ == "__main__":
    main()
