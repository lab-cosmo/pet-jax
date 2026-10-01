# Changelog

All notable changes to `pet-jax` are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [semantic versioning](https://semver.org/).

Every PR that changes something a user can see adds a line under **Unreleased**, in the matching group (Added, Changed, Fixed, Removed). At release time, **Unreleased** becomes the new version's section.

## Unreleased

### Added

- First release. Includes the JAX/Flax reimplementation of uPET, the `UPETCalculator` ASE calculator (energy, forces, stress, adaptive cutoffs), and `petjax-convert` for `metatrain` checkpoints (PET-MAD, PET-OMAT and other bare or LLPR-wrapped PET checkpoints, from format v10 onwards). Also includes charge/spin system conditioning (e.g. PET-OMol).
