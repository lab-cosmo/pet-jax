# Changelog

All notable changes to `pet-jax` are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [effort-based versioning](https://jacobtomlinson.dev/effver/) (EffVer, as used by JAX): the bumped component says how much effort upgrading takes, from none (micro) through some (meso) to significant (macro). While on `0.x`, the second number plays the macro role and the third the meso role.

Every PR that changes something a user can see adds a line under **Unreleased**, in the matching group (Added, Changed, Fixed, Removed). At release time, **Unreleased** becomes the new version's section.

## Unreleased

### Fixed

- The package metadata now requires Python ≥ 3.11. It claimed 3.10, but `jax>=0.10` needs 3.11, so installing on 3.10 failed with a resolver error instead of a clear message.

## [0.1.0] - 2026-10-01

### Added

- First release. Includes the JAX/Flax reimplementation of uPET, the `UPETCalculator` ASE calculator (energy, forces, stress, adaptive cutoffs), and `petjax-convert` for `metatrain` checkpoints (PET-MAD, PET-OMAT and other bare or LLPR-wrapped PET checkpoints, from format v10 onwards). Also includes charge/spin system conditioning (e.g. PET-OMol).
