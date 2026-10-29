# Changelog

All notable changes to `pet-jax` are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [effort-based versioning](https://jacobtomlinson.dev/effver/) (EffVer, as used by JAX): the bumped component says how much effort upgrading takes, from none (micro) through some (meso) to significant (macro). While on `0.x`, the second number plays the macro role and the third the meso role.

Every PR that changes something a user can see adds a line under **Unreleased**, in the matching group (Added, Changed, Fixed, Removed). At release time, **Unreleased** becomes the new version's section.

## Unreleased

### Changed

- `truncate`, `truncate_edges` and `pack_edges` return a `Truncation` named tuple in place of the bare `overflow` flag: the per-atom and per-pair cutoffs, the selected mask, the selected count per center, and `overflow`, mirroring what `select_edges` reports host-side. Callers that unpacked the flag read `.overflow`; the counts let a driver size `k_sel` without leaving jit.

### Added

- `center_order` and `corresponding_edges`: jit-safe twins of the two layout steps `to_structure` does on the host (sort a padded flat neighbour list by center with padding last, and build the reverse map by stable multi-key sorts on int32 keys), for raw neighbour lists built on the device.
- `Backbone` takes `final_message_passing` (default `True`). With it off, the reverse-edge message passing after the last GNN layer is skipped. `metatrain` has no equivalent, so such a model is not exportable.
- `Backbone` gains two hooks for architectures built on the PET backbone, such as FlashMD: `species_embedding=False` drops the species embedding table and takes the initial node features as a `node_features` input instead, and the `cutoff_function` attribute (default `cutoff_bump`; `cutoff_cosine` is new in `petjax.utils`) picks the cutoff. Both default to the current behaviour.
- `pack_edges` takes optional per-pair `pair_cutoffs` and packs them with the pairs. Without them the model's scalar cutoff applies, as before.

### Fixed

- The package metadata now requires Python ≥ 3.11. It claimed 3.10, but `jax>=0.10` needs 3.11, so installing on 3.10 failed with a resolver error instead of a clear message.
- The RMSNorm and LayerNorm epsilons now match `torch`'s defaults (`finfo(dtype).eps` and `1e-5`) instead of `flax`'s (`1e-6`). Predictions move by ~1e-7 relative; in fp64, agreement with `metatrain` improves by an order of magnitude.
- `petjax-convert` keeps the energy, force and stress scales in fp64 instead of rounding them to fp32. Checkpoints converted before this carry the rounded scales; reconvert with the new `--force` flag (the converter otherwise leaves existing output alone) to get fp64 predictions that match `metatrain` to ~1e-14 eV/atom instead of ~1e-7. fp32 inference is unchanged.

## [0.1.0] - 2026-10-01

### Added

- First release. Includes the JAX/Flax reimplementation of uPET, the `UPETCalculator` ASE calculator (energy, forces, stress, adaptive cutoffs), and `petjax-convert` for `metatrain` checkpoints (PET-MAD, PET-OMAT and other bare or LLPR-wrapped PET checkpoints, from format v10 onwards). Also includes charge/spin system conditioning (e.g. PET-OMol).
