"""ASE calculator for UPET (JAX). Verlet-skin NL cache, JIT'd adaptive
selection inside the forward, overflow-triggered re-JIT."""

import numpy as np
import jax
import jax.numpy as jnp

from ase.calculators.calculator import BaseCalculator
from ase.stress import full_3x3_to_voigt_6_stress

from .predict import get_predict_fn
from .select import determine_k_sel
from .structure import _bucket_or, to_structure
from .utils import cast_floats


class UPETCalculator(BaseCalculator):
    """ASE calculator for UPET (JAX) with a two-phase NL: raw NL at
    ``cutoff + skin`` (Verlet-cached), adaptive selection inside JIT every
    step. Overflow triggers a rebuild with larger ``k_sel`` and re-JIT."""

    name = "upet-jax"
    parameters = {}
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model,
        params,
        metadata,
        skin=0.5,
        stress=True,
        default_dtype="float32",
        no_shadow=False,
        bucket_strategy="multiples",
        n_atoms_bucket_strategy=None,
        n_pair_bucket_strategy=None,
        k_sel_bucket_strategy=None,
        extra_neighbors=4,
        k_sel_override=None,
        cutoff_override=None,
        add_offset=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = model
        self._metadata = metadata
        self._skin = skin
        self._stress = stress
        self._no_shadow = no_shadow
        self._bucket_strategy = bucket_strategy
        self._n_atoms_bucket_strategy = n_atoms_bucket_strategy
        self._n_pair_bucket_strategy = n_pair_bucket_strategy
        self._k_sel_bucket_strategy = (
            k_sel_bucket_strategy if k_sel_bucket_strategy is not None else bucket_strategy
        )
        self._extra_neighbors = extra_neighbors
        self._k_sel_override = k_sel_override
        self._add_offset = add_offset

        # cutoff_override narrows ONLY the vesin raw-NL query radius (to
        # cutoff_override + skin). The UPET model is untouched — probes, the
        # adaptive-cutoff layer and the cutoff bump all keep using the trained
        # config["cutoff"]. A perf knob for when the trained cutoff is wider
        # than the radius adaptive selection actually reaches.
        #
        # CORRECTNESS RISK: the raw NL must still hold every pair the model
        # uses. Adaptive per-atom cutoffs range up to config["cutoff"], so if
        # cutoff_override is below the largest one selected, surviving pairs are
        # silently dropped — and the adaptive-cutoff counts themselves are
        # starved. Both corrupt energy/forces with no error raised.
        self._cutoff = (
            cutoff_override if cutoff_override is not None else metadata["config"]["cutoff"]
        )
        self._species_to_index = metadata["species_to_index"]

        if default_dtype == "float64":
            jax.config.update("jax_enable_x64", True)
        self._dtype = jnp.float64 if default_dtype == "float64" else jnp.float32

        # Cast params to the requested dtype so fp64 callers don't silently
        # run mixed-precision (fp32 weights × fp64 activations) with a PET-MAD
        # checkpoint that ships as fp32.
        self._params = cast_floats(params, self._dtype)

        self._nl_cache = NeighborListCache(skin=skin)
        # predict_fn is shape-agnostic — built once; it reads N_padded / k_sel
        # off the input arrays and the JIT retraces per shape.
        self._predict_fn = get_predict_fn(
            self._model,
            stress=self._stress,
            no_shadow=self._no_shadow,
        )
        self._N_padded = None
        self._n_pair_padded = None
        self._k_sel = None
        self._structure = None
        self._shift_offset = 0.0

        self._shifts = {int(z): float(v) for z, v in metadata["shifts"].items()}

        if not stress:
            self.implemented_properties = ["energy", "forces"]

    @classmethod
    def from_checkpoint(cls, folder, **kwargs):
        from .convert import load_checkpoint
        from .model import UPET

        params, metadata = load_checkpoint(folder)
        model = UPET(**metadata["config"], energy_scale=metadata["energy_scale"])
        return cls(model, params, metadata, **kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=None, **kwargs):
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("No atoms provided")

        if self._nl_cache.needs_update(atoms):
            self._build_structure(atoms)
        else:
            self._update_geometry(atoms)

        results = self._predict_fn(self._params, self._structure)

        if bool(results["overflow"]):
            self._build_structure(atoms, force_recompute_k_sel=True)
            results = self._predict_fn(self._params, self._structure)
            if bool(results["overflow"]):
                raise RuntimeError(
                    "UPETCalculator: cannot recover from overflow "
                    f"after retry (k_sel={self._k_sel})"
                )

        n_real = len(atoms)
        energy = float(results["energy"])
        if self._add_offset:
            energy += self._shift_offset
        forces = np.array(results["forces"][:n_real], dtype=np.float64)

        self.results = {"energy": energy, "forces": forces}

        if self._stress and "stress" in results and atoms.pbc.any():
            virial = np.array(results["stress"], dtype=np.float64)
            volume = atoms.get_volume()
            self.results["stress"] = full_3x3_to_voigt_6_stress(virial / volume)

        return self.results

    # -- internals --

    def _to_jax_structure(self, structure):
        return cast_floats(structure, self._dtype)

    def _build_structure(self, atoms, force_recompute_k_sel=False):
        """Rebuild the raw NL and size k_sel; stamp the structure pytree.

        Called by ``calculate`` only on a Verlet-cache miss (atom count /
        species / pbc change, or displacement past the skin) or an overflow
        retry — a plain displacement step goes through ``_update_geometry``
        and rebuilds nothing.

        Two things, each behind its own gate:

        * raw NL — always (``to_structure``, vesin at ``cutoff + skin``).
        * k_sel — recomputed via ``determine_k_sel`` (expensive: a CPU
          selection kernel at ``[n_pair_padded]``) only on the first call, a
          shape change, or ``force_recompute_k_sel``; otherwise the cached
          bucket is reused, since k_sel rarely shifts between displacement-only
          rebuilds.

        ``self._predict_fn`` is shape-agnostic (built once in ``__init__``): it
        reads N_padded / k_sel off the input arrays, so a shape change just
        retraces it. k_sel reaches the JIT via the ``k_sel_sizer`` carrier
        stamped into the structure here.

        ``extra_neighbors`` pads k_sel above ``k_sel_actual`` so a few new
        neighbours don't overflow every step (bucket promotion only kicks in at
        a boundary). If k_sel still grows past its bucket, the forward returns
        ``overflow=True`` and ``calculate`` retries here with
        ``force_recompute_k_sel=True``.
        """
        structure = to_structure(
            atoms,
            self._cutoff,
            self._species_to_index,
            skin=self._skin,
            bucket_strategy=self._bucket_strategy,
            n_atoms_bucket_strategy=self._n_atoms_bucket_strategy,
            n_pair_bucket_strategy=self._n_pair_bucket_strategy,
        )

        N_padded = structure["positions"].shape[0]
        n_pair_padded = structure["centers"].shape[0]

        shape_changed = (
            self._N_padded is None
            or N_padded != self._N_padded
            or n_pair_padded != self._n_pair_padded
        )

        if self._k_sel_override is not None:
            # Caller forced k_sel — skip determine_k_sel entirely. Overflow retry
            # still protects against genuinely-too-small overrides.
            k_sel_padded = self._k_sel_override
        elif force_recompute_k_sel or shape_changed:
            k_sel_actual = determine_k_sel(
                structure,
                self._model.get_probes(),
                self._metadata["config"]["num_neighbors_adaptive"],
                self._metadata["config"]["cutoff_width"],
            )
            # T = k_sel edge tokens + 1 central-atom token. Bucket T (an even
            # T keeps attention on XLA's fused fast path); k_sel = T - 1.
            token_padded = _bucket_or(
                k_sel_actual + 1 + self._extra_neighbors, self._k_sel_bucket_strategy
            )
            k_sel_padded = token_padded - 1
        else:
            k_sel_padded = self._k_sel

        self._N_padded = N_padded
        self._n_pair_padded = n_pair_padded
        self._k_sel = k_sel_padded

        # k_sel_sizer carries k_sel into the shape-agnostic predict_fn via its
        # shape; the JIT retraces when k_sel (or any other shape) changes.
        self._structure = {
            **self._to_jax_structure(structure),
            "k_sel_sizer": jnp.zeros(k_sel_padded, dtype=bool),
        }
        max_shift = (
            int(np.max(np.abs(structure["cell_shifts"])))
            if len(structure["cell_shifts"]) > 0
            else 0
        )
        self._nl_cache.save_reference(atoms, max_cell_shift=max_shift)

        # Composition shifts are constant w.r.t. positions; Python fp64 sum
        # added post-JIT. Species can only change via a rebuild, so here.
        self._shift_offset = sum(self._shifts[int(z)] for z in atoms.get_atomic_numbers())

    def _update_geometry(self, atoms):
        """Update positions and cell without rebuilding NL."""
        n_real = len(atoms)
        N_padded = self._N_padded
        positions = np.zeros((N_padded, 3), dtype=np.float64)
        positions[:n_real] = atoms.get_positions()
        cell = np.array(atoms.get_cell()[:], dtype=np.float64)
        self._structure = {
            **self._structure,
            "positions": jnp.array(positions, dtype=self._dtype),
            "cell": jnp.array(cell, dtype=self._dtype),
        }


class NeighborListCache:
    """Verlet-skin neighbor list cache.

    Tracks position displacement and cell deformation between rebuilds.
    """

    def __init__(self, skin=0.5):
        self.skin = skin
        self._ref_positions = None
        self._ref_cell = None
        self._ref_pbc = None
        self._ref_numbers = None
        self._max_cell_shift = None

    def needs_update(self, atoms):
        if self._ref_positions is None:
            return True
        if len(atoms) != len(self._ref_positions):
            return True
        if (atoms.get_atomic_numbers() != self._ref_numbers).any():
            return True
        if (atoms.get_pbc() != self._ref_pbc).any():
            return True

        if self._max_cell_shift is None:
            if (atoms.get_cell()[:] != self._ref_cell).any():
                return True

        displacements = atoms.get_positions() - self._ref_positions
        max_disp = float(np.linalg.norm(displacements, axis=1).max())

        if self._max_cell_shift is not None and self._max_cell_shift > 0:
            cell_change = np.array(atoms.get_cell()[:]) - self._ref_cell
            cell_vector_norms = np.linalg.norm(cell_change, axis=1)
            max_cell_contrib = float(self._max_cell_shift * cell_vector_norms.sum())
        else:
            max_cell_contrib = 0.0

        return bool((2 * max_disp + max_cell_contrib) > self.skin)

    def save_reference(self, atoms, max_cell_shift=None):
        self._ref_positions = atoms.get_positions().copy()
        self._ref_cell = np.array(atoms.get_cell()[:]).copy()
        self._ref_pbc = atoms.get_pbc().copy()
        self._ref_numbers = atoms.get_atomic_numbers().copy()
        self._max_cell_shift = max_cell_shift
