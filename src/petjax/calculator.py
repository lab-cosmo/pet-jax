"""ASE calculator for UPET (JAX). Verlet-skin NL cache, JIT'd adaptive
selection inside the forward, overflow-triggered re-JIT."""

import numpy as np
import jax
import jax.numpy as jnp

import sys
import warnings

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
        cutoff_override=None,
        add_offset=True,
        debug=False,
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
        self._add_offset = add_offset
        self._debug = debug
        # Perf-tuning numbers, refreshed each NL rebuild (None until first
        # `calculate`). Always populated; `debug` only gates the stderr summary.
        self.debug_stats = None
        # Cached max selected adaptive cutoff from the last `determine_k_sel`;
        # carried across rebuilds that reuse k_sel (which skip the sizing run).
        self._max_selected_cutoff = None

        # cutoff_override narrows ONLY the vesin raw-NL query radius (to
        # cutoff_override + skin). The UPET model is untouched — probes, the
        # adaptive-cutoff layer and the cutoff bump all keep using the trained
        # config["cutoff"]. A performance knob for when the trained cutoff is wider
        # than the radius adaptive selection actually reaches.
        #
        # CORRECTNESS RISK: the raw NL must still hold every pair the model
        # uses. Adaptive per-atom cutoffs range up to config["cutoff"], so if
        # cutoff_override is below the largest one selected, surviving pairs are
        # silently dropped — and the adaptive-cutoff counts themselves are
        # starved. Both corrupt energy/forces with no error raised.
        #
        # `_check_cutoff_override` warns (per rebuild) once the measured reach
        # exceeds cutoff_override — but it is a warning, not a guard.
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
        """Rebuild the raw NL, size k_sel, and assemble the structure pytree
        (with the ``k_sel_sizer`` carrier).

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
        written into the structure here.

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

        k_sel_actual = None  # set only when determine_k_sel runs (debug stat)
        if force_recompute_k_sel or shape_changed:
            k_sel_actual, self._max_selected_cutoff = determine_k_sel(
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

        self._record_debug(atoms, structure, k_sel_actual, shape_changed, force_recompute_k_sel)
        self._check_cutoff_override()

    def _record_debug(self, atoms, structure, k_sel_actual, shape_changed, force_recompute_k_sel):
        """Refresh and maybe print ``self.debug_stats`` after a rebuild. The
        padded sizes are read off ``self`` (just stamped by ``_build_structure``);
        the rest are passed in as they aren't kept on ``self``."""
        self.debug_stats = {
            "n_atoms": len(atoms),
            "N_padded": self._N_padded,
            "n_pair_raw": int(structure["pair_mask"].sum()),
            "n_pair_padded": self._n_pair_padded,
            "k_sel_actual": k_sel_actual,  # None if reused (no sizing run)
            "k_sel_padded": self._k_sel,
            "extra_neighbors": self._extra_neighbors,
            "recomputed_k_sel": k_sel_actual is not None,
            "max_selected_cutoff": self._max_selected_cutoff,  # None until first sizing
            "cutoff_raw_nl": self._cutoff,
            "cutoff_trained": self._metadata["config"]["cutoff"],
            "retrace": shape_changed,  # predict_fn retraces iff a shape changed
            "overflow_retry": force_recompute_k_sel,
        }
        if self._debug:
            self._print_debug(self.debug_stats)

    def _check_cutoff_override(self):
        """Warn if ``cutoff_override`` is smaller than the reach the adaptive
        selection asks for.

        ``max_selected_cutoff`` is the largest per-atom adaptive cutoff the
        selection assigns (up to the trained cutoff). The raw NL only supplies
        neighbours to ``cutoff_override + skin``; once that cutoff passes
        ``cutoff_override`` the skin is spent as cutoff slack and real neighbours
        are silently dropped, corrupting energy/forces. Only fires when an
        override is active; the reach is read from the last sizing run."""
        co = self._max_selected_cutoff
        override_active = self._cutoff < self._metadata["config"]["cutoff"]
        if co is None or not override_active or co <= self._cutoff:
            return
        raw_nl_radius = self._cutoff + self._skin
        warnings.warn(
            f"UPETCalculator: adaptive selection reaches {co:.2f} Å, exceeding "
            f"cutoff_override={self._cutoff:.2f} Å (raw-NL radius "
            f"{raw_nl_radius:.2f} Å incl. {self._skin:.2f} Å skin). The override "
            f"is too small — it eats the Verlet skin and risks silently dropping "
            f"pairs the model needs (corrupting energy/forces). Raise "
            f"cutoff_override toward the trained "
            f"{self._metadata['config']['cutoff']:.2f} Å, or unset it.",
            stacklevel=2,
        )

    @staticmethod
    def _print_debug(s):
        def pct(used, total):
            return f"{100.0 * (total - used) / total:.1f}% pad" if total else "n/a"

        tag = "overflow-rebuild" if s["overflow_retry"] else "rebuild"
        lines = [
            f"[upet] {tag}: atoms {s['n_atoms']}/{s['N_padded']} "
            f"({pct(s['n_atoms'], s['N_padded'])})",
            f"[upet]   pairs {s['n_pair_raw']}/{s['n_pair_padded']} "
            f"({pct(s['n_pair_raw'], s['n_pair_padded'])})",
        ]
        if s["recomputed_k_sel"]:
            lines.append(
                f"[upet]   k_sel {s['k_sel_actual']}->{s['k_sel_padded']} "
                f"(recompute, +{s['extra_neighbors']} extra)"
            )
        else:
            lines.append(f"[upet]   k_sel {s['k_sel_padded']} (reused)")
        if s["max_selected_cutoff"] is not None:
            override = "" if s["cutoff_raw_nl"] == s["cutoff_trained"] else " [override]"
            lines.append(
                f"[upet]   max_cutoff {s['max_selected_cutoff']:.2f} selected / "
                f"{s['cutoff_raw_nl']:.2f} raw-NL{override} / "
                f"{s['cutoff_trained']:.2f} trained"
            )
        if s["retrace"]:
            lines.append("[upet]   shape changed -> predict_fn retrace")
        print("\n".join(lines), file=sys.stderr)

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
