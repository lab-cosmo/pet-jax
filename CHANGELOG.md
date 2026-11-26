# Changelog

All notable changes to `pet-jax` are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [effort-based versioning](https://jacobtomlinson.dev/effver/) (EffVer, as used by JAX): the bumped component says how much effort upgrading takes, from none (micro) through some (meso) to significant (macro). While on `0.x`, the second number plays the macro role and the third the meso role.

Every PR that changes something a user can see adds a line under **Unreleased**, in the matching group (Added, Changed, Fixed, Removed). At release time, **Unreleased** becomes the new version's section.

## Unreleased

## [0.1.1] - 2026-11-26

### Changed

- `truncate`, `truncate_edges` and `pack_edges` return a `Truncation` named tuple in place of the bare `overflow` flag; read `.overflow` where you unpacked the flag. It also carries the cutoffs, the selected mask and the count per center, so a driver can size `k_sel` without leaving jit.

### Added

- `petjax.structure.get_neighbor_list_fn` builds the neighbour list inside jit with `vesin-jax`, for drivers that never leave it. Needs the new `vesin-jax` extra.
- `petjax.structure.center_order` and `corresponding_edges`: jit-safe versions of the sort by center and the reverse-edge map, for neighbour lists built on the device.
- `pack_edges` takes optional per-pair `pair_cutoffs`. Without them the model's scalar cutoff applies, as before.
- `Backbone` takes `final_message_passing` (default `True`); switching it off skips the reverse-edge message passing after the last GNN layer. `metatrain`'s PET has no such option, so this is only for models trained in JAX.
- Two `Backbone` hooks for architectures built on it, such as FlashMD: `species_embedding=False` replaces the species embedding table with a `node_features` input, and `cutoff_function` picks the cutoff (default `cutoff_bump`; `cutoff_cosine` is new).
- `petjax-convert --force` overwrites existing output.

### Fixed

- `cutoff_bump` no longer rounds to zero in fp32 near the cutoff, which made the attention bias jump there. It is now evaluated as `sigmoid(2 cot(πx))`, as in `metatrain`.
- A float32 model no longer runs in float64 when JAX's 64-bit mode is on (as with i-PI's jax backend).
- The `solver` adaptive cutoff finds the root by plain bisection, as `metatrain` does since metatensor/metatrain#1331. The previous Newton-bisection hybrid could return a cutoff that is not a root (lab-cosmo/pet-jax#48). Predictions of `solver` checkpoints such as PET-MAD v1.6 move by up to ~1e-5 eV/Å; `grid` checkpoints are unchanged.
- The RMSNorm and LayerNorm epsilons match `torch`'s defaults (`finfo(dtype).eps` and `1e-5`) instead of `flax`'s (`1e-6`). Predictions move by ~1e-7 relative; in fp64, agreement with `metatrain` improves by an order of magnitude.
- `petjax-convert` keeps the energy, force and stress scales in fp64 instead of rounding them to fp32. Reconvert older checkpoints with `--force` to get fp64 predictions that match `metatrain` to ~1e-14 eV/atom instead of ~1e-7. fp32 inference is unchanged.
- The package metadata requires Python ≥ 3.11, as `jax>=0.10` does. It claimed 3.10, so installing there failed with a resolver error.

## [0.1.0] - 2026-10-01

### Added

- First release. Includes the JAX/Flax reimplementation of uPET, the `UPETCalculator` ASE calculator (energy, forces, stress, adaptive cutoffs), and `petjax-convert` for `metatrain` checkpoints (PET-MAD, PET-OMAT and other bare or LLPR-wrapped PET checkpoints, from format v10 onwards). Also includes charge/spin system conditioning (e.g. PET-OMol).
