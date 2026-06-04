# `pet-jax`

A clean JAX/Flax reimplementation of the **uPET (Point-Edge Transformer)** family of interatomic potentials, targeting numerical equivalence with the upstream [metatrain](https://github.com/metatensor/metatrain) PyTorch implementation while being efficient enough for production inference and geometry optimisation.

`pet-jax` loads `metatrain` checkpoints (e.g. PET-MAD), runs them on JAX, and exposes an ASE calculator (`UPETCalculator`) suitable as a drop-in replacement for the PyTorch reference in molecular dynamics, relaxations, and downstream property calculations.

## What's in the box

```
src/petjax/
  __init__.py     # public API
  model.py        # UPET (Flax nn.Module)
  calculator.py   # UPETCalculator (ASE)
  convert.py      # metatrain .ckpt → Flax msgpack
  cli.py          # petjax-convert entrypoint
  structure.py    # host-side neighborlist build
  select.py       # in-JIT adaptive selection
  predict.py      # in-JIT forward + autodiff
  utils.py        # shared helpers
```

## Installation

`pet-jax` needs Python ≥ 3.10. It is not on PyPI yet, so install it from the repository:

```bash
pip install "git+https://github.com/lab-cosmo/pet-jax"
```

This pulls in the inference stack (`jax`, `flax`, `numpy`, `ase`, `vesin`, `marathon-train`) — everything needed to load a checkpoint and run the calculator.

Converting upstream `metatrain` `.ckpt` files needs the optional `convert` extra (`torch`, `metatomic-torch`, `metatrain`), which is **not** required for inference:

```bash
pip install "pet-jax[convert] @ git+https://github.com/lab-cosmo/pet-jax"
```

Or from a checkout, which is the easiest way to also get the examples and tests:

```bash
git clone https://github.com/lab-cosmo/pet-jax
cd pet-jax
pip install -e ".[convert]"   # drop [convert] for an inference-only install
```

Any environment manager works (`pip`, `uv`, `conda`, ...); the examples below use plain console scripts, and the `uv` + `tox` development workflow is in [Development](#development) (`uv sync --extra convert` for the editable install).

## Quick start

### Fetch a PET-MAD checkpoint

```bash
petjax-convert pet-mad-xs --out checkpoints/pet-mad-xs
```

This downloads the PET-MAD `.ckpt` from Hugging Face (`lab-cosmo/upet`) and converts it directly to `pet-jax`'s Flax msgpack layout — no TorchScript intermediate. The `convert` extra pulls in `torch`, `metatomic-torch`, and `metatrain` (needed only for conversion; not for inference).

`petjax-convert` also accepts arbitrary URLs or local `.ckpt` paths:

```bash
petjax-convert https://example.com/my-pet.ckpt --out checkpoints/my-pet
petjax-convert ~/runs/best.ckpt              --out checkpoints/my-pet
```

### Inference on a single structure

```python
from ase.io import read
from petjax import UPETCalculator

calc = UPETCalculator.from_checkpoint("checkpoints/pet-mad-xs", stress=True)

atoms = read("my_structure.xyz")
atoms.calc = calc

energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Å
stress = atoms.get_stress()            # voigt, eV/Å³
```

Relaxations and so on can be done with the same calculator!

## Architecture: how the calculator handles adaptive cutoffs

The PET model itself operates on a "rectangular"-style neighborlist, i.e., shaped as `[n_atoms, n_neighbors]`, since its main operation is edge-to-edge attention which (naively) requires an intermediate `[n_atoms, n_neighbors, n_neighbors]` attention matrix.[^flat-nl]

[^flat-nl]: Internally, the neighborlist is stored in flattened `[n_atoms*n_neighbors]` form, but this is not relevant here.

The universal PET models wrap an *adaptive cutoff* procedure around the model, which selects a per-atom cutoff such that only an approximately fixed number of neighbors are considered for each atom. This avoids spikes in memory usage for dense systems, improves batching efficiency for diverse dataset during training, and also lets the model predict dimer curves better. At a high level, the procedure works as follows:

1. Compute an initial large neighborlist with a big cutoff,
2. Determine a cutoff for each atom that fits approximately the target number of neighbors,
3. Compute per-pair cutoffs as average of both per-atom cutoffs,
4. Apply a per-pair boolean mask — keep pair `(i, j)` iff `r_ij` is at most `pair_cutoff_ij`.

The subtlety here is that since the cutoffs depend on the position of atoms *outside* the cutoff, gradients have to flow through the procedure to some extent and it therefore has to be done inside the model's forward pass. So we need to make the procedure compatible with `jax`, so we can `jit` it and transform it with `grad`. The constraints: No data-dependent shapes, and few shape changes to avoid costly recompilations. This is made more complicated by the fact that smoothness dictates that the procedure is not exact: We can only *target* a certain number of neighbors, but we can't guarantee it.

The solution to this is a two-phase design: Outside of `jax`, we compute the initial big neighborlist and determine the final number, `k_sel`, of neighbors. We can round `k_sel` to something larger to avoid recompiles. We also have to pad the initial big neighborlist to a fixed shape, but this is easy to achieve. Inside `jax` (i.e., the `jax.jit` boundary), we then re-run the procedure and pack into `k_sel`, which we already know ahead of time due to step one. If we exceed `k_sel`, we return an `overflow` signal to tell the calculator that `k_sel` needs to be recomputed.

For the design rationale and more details, see [`src/petjax/README.md`](src/petjax/README.md).

## Performance knobs

`UPETCalculator` exposes a few keyword arguments that trade off accuracy, memory, JIT recompile frequency, and per-step cost. Maximum efficiency requires tuning them for your particular problem.

- `default_dtype` (default `"float32"`): set to `"float64"` for demanding relaxations or precision-sensitive comparisons. fp32 is ~2× faster; PET-MAD checkpoints ship fp32, so fp64 just promotes the cached params. For most inference fp32 is fine.
- `skin` (default `0.5` Å): Verlet-skin radius. Larger skin → fewer raw-NL rebuilds (good for long MD with small step sizes) at the cost of more padded pairs per step. Shrink if `vesin`/raw-NL build dominates; grow if you see frequent skin-triggered rebuilds.
- `stress` (default `True`): set to `False` to skip the strain-derivative virial. Drops one `value_and_grad` argument; very slightly faster, no `stress` key in `results`.
- `no_shadow` (default `False`): cut gradients through the adaptive-cutoff function. Slightly faster, slightly different forces (drops the "shadow" contribution from the cutoff procedure). Energy is unchanged.
- `bucket_strategy` (default `"multiples"`) and axis-specific overrides `n_atoms_bucket_strategy`, `n_pair_bucket_strategy`, `k_sel_bucket_strategy`: how shapes are rounded up to bucket sizes. Coarser bucketing → fewer JIT recompiles but more padded compute; finer → tighter shapes but more recompiles when shapes drift. The default is a reasonable compromise.
- `extra_neighbors` (default `4`): slack added to `k_sel_actual` so a step or two of neighbour growth doesn't trigger an overflow rebuild. Increase if you see frequent overflow retries during MD; decrease to save padded edge work.
- `num_neighbors_adaptive` (default `None`): override how many neighbours per atom the adaptive cutoff aims to select; `None` uses the value the model was trained with. Raising it lets the model see more neighbours per atom — accuracy tends to improve (models are robust to this) for a modest increase in per-step cost. If you also set a tight `cutoff_override`, the wider reach makes its warning below more likely to fire.
- `cutoff_override` (default `None`): narrow the `vesin` raw-NL query radius below the trained `config["cutoff"]`. ⚠ **Correctness risk**: must stay above the largest per-atom adaptive cutoff that selection ever reaches, otherwise surviving pairs are silently dropped. Only set this when you know your system's adaptive cutoffs don't approach `config["cutoff"]`. The calculator measures the actual reach each rebuild (`debug_stats["max_selected_cutoff"]`) and emits a `warnings.warn` if your override falls below it — use that to find a safe value.
- `debug` (default `False`): emit a per-rebuild summary to `stderr` (padding waste on each axis, `k_sel_actual → k_sel_padded`, the measured `max_selected_cutoff` vs the raw-NL/trained cutoff, and whether `predict_fn` will retrace). Keyed to NL rebuilds and overflow events, not per MD step, so long trajectories stay quiet. The same numbers are always available on `calc.debug_stats` regardless of this flag; the too-small-`cutoff_override` warning above also fires independently of it.


## Conventions

- **Energy** is total energy (eV), not per-atom.
- **Stress** returned by the calculator is in ASE's Voigt convention (eV/Å³); internally the virial is the strain derivative `dU/dε` in eV.
- **Scaling/shifting**: raw model output is multiplied by `energy_scale` inside JIT (at params dtype; forces scale too). Per-element composition shifts are added post-JIT on the calculator side in Python fp64; they contribute zero to forces. Both live in checkpoint metadata.
- **Adaptive cutoff**: per-atom, recomputed inside the autograd graph each step (required for force correctness).

## Checkpoint format

`pet-jax` checkpoints are a directory with two files:

```
<ckpt_dir>/
  model.msgpack    # Flax parameter tree (nested dict of arrays)
  metadata.yaml    # config (architecture hypers), energy_scale, shifts,
                   # species_to_index
```

Use `UPETCalculator.from_checkpoint("<ckpt_dir>")` to load. Conversion from the upstream `metatrain` `.ckpt` format goes through `petjax-convert`, which reads the LLPR-wrapped PET-MAD checkpoint directly (no TorchScript intermediate). Checkpoint format is pinned to PET-MAD v1.5.0 (outer `llpr` v3 / inner `pet` v11); other versions are rejected — run `mtt upgrade` on the source to migrate.

## Validation

`tests/test_predictions.py` compares the calculator's output against saved `metatrain` reference `.xyz` files on the mini CI dataset (and the larger `test_s/m/l` datasets under `--run-extended`).

`tests/test_calculator.py` additionally covers:

- **Shift plumbing**: composition shifts are added exactly once, in fp64, with zero leakage into forces.
- **no_shadow**: optional cut of adaptive-cutoff gradients preserves energy and produces finite, slightly shifted forces.
- **Position-only update**: small geometry changes do not trigger a re-JIT.
- **Cell relaxation**: BFGS + FrechetCellFilter over 30 steps on a small periodic structure — energy doesn't increase, forces stay finite, NL rebuilds stay below half the step count (the whole point of the Verlet-skin + bucketed-shapes machinery).

## Working with the upstream PyTorch implementation

To run the original `metatrain`/`metatomic-torch` code without polluting the `pet-jax` environment:

```bash
uv run --with metatrain --with metatomic-torch python ...
```

Useful for cross-checks against ground truth.

## Development

The canonical task runner is `tox` (with the `tox-uv` plugin so environments are created via `uv`). Envs declared in `tox.ini`:

- `tox -e lint` — `ruff check` + `ruff format --check`
- `tox -e tests` — `pytest` mini suite (pass `--run-extended` after `--` for the extended local suite)
- `tox -e fetch-checkpoints` — `petjax-convert pet-mad-xs --out tests/assets/checkpoints/pet-mad-xs` (one-off; uses the `convert` extra)

Bootstrap:

```bash
uv tool install --with tox-uv tox
tox -e fetch-checkpoints   # first time only
tox -e lint
tox -e tests

# extended local suite (test_s, test_m, test_l × pet-mad-xs, pet-mad-s; needs
# extra assets + the pet-mad-s checkpoint, all gitignored)
tox -e tests -- --run-extended
```

Extended tests skip individually when their input files are missing.

For tight inner-loop work you can bypass `tox`:

```bash
uv sync                                  # install core + dev
uv run pytest tests/ -x
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Dependencies

Core (`pyproject.toml`):

- `jax`, `jaxlib`, `flax` — neural net + autodiff
- `numpy`, `ase` — data structures
- `vesin` — fast neighbourlist construction
- `marathon-train` — checkpoint I/O (msgpack/yaml) and bucket-size helpers

Optional (the `convert` extra — see [Installation](#installation)):

- `torch`, `metatomic-torch`, `metatrain` — only needed when converting `metatrain` `.ckpt` files. Inference itself runs on the JAX stack alone.

## Contributing

Code conventions (`ruff` config, naming patterns, file layout, JIT placement rule, Markdown soft-wrap, internal NL/attention conventions) and the architecture deep-dive live in [`src/petjax/README.md`](src/petjax/README.md). Read it before submitting non-trivial PRs.

## Status

Working: inference, forces, stress, BFGS/FIRE relaxations, cell optimization, `metatrain` checkpoint conversion, parity with upstream PET.

Not yet: training (use `metatrain` directly), batched multi-structure inference in the calculator (single-structure only), GPU performance tuning.

## License

MIT. See [LICENSE](LICENSE).
