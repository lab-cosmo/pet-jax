"""ASE calculator for UPET (JAX). Verlet-skin NL cache (or a raw NL built
inside the forward), JIT'd adaptive selection inside the forward,
overflow-triggered re-JIT, optional cutoff guard."""

import numpy as np
import jax
import jax.numpy as jnp

import math
import sys
import warnings

from ase.calculators.calculator import BaseCalculator
from ase.stress import full_3x3_to_voigt_6_stress

from .express import Express
from .predict import get_predict_fn
from .select import select_edges
from .structure import (
    _bucket_or,
    conditioning_inputs,
    lattice_shifts,
    read_conditioning,
    to_structure,
)
from .utils import cast_floats


class UPETCalculator(BaseCalculator):
    """ASE calculator for UPET (JAX) with a two-phase NL: raw NL at
    ``cutoff + skin`` (Verlet-cached), adaptive selection inside JIT every
    step. Overflow triggers a rebuild with larger ``k_sel`` and re-JIT.

    With ``express={"neighbor_list": "device"}`` the raw NL is instead built
    inside the forward at ``cutoff`` every step (no skin, no Verlet cache, no
    vesin per step). Its static pair capacity is the padded pair count of a
    host list at ``cutoff + skin``, built with ``to_structure`` only when the
    calculator (re)sizes: on the first call, when atom count, species or pbc
    change, and on overflow. The same host list feeds ``select_edges``, which
    sizes ``k_sel``. A pair overflow on the device rebuilds that host list from
    the current geometry, which holds every current pair, so one retry fits.

    With ``express={"cutoff_guard": True}`` and ``cutoff_override`` active, a
    step whose adaptive cutoffs reach the raw list's radius (the list may have
    dropped pairs) is discarded: the raw-NL cutoff widens by ``cutoff_widen``
    (capped at the trained cutoff), the structure is rebuilt and the forward
    rerun, counted in ``debug_stats["cutoff_guard_events"]``. The condition is
    exact for the solver method: the truncated selection equals the full one
    iff every adaptive cutoff lies below the radius the list is complete to,
    and a violation always shows as a cutoff at or above it. That radius is
    ``cutoff_override`` for the device list and ``cutoff_override + skin -
    drift`` for the host list, ``drift`` the Verlet bound on how far any pair
    distance moved since the build."""

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
        matmul_precision="high",
        no_shadow=False,
        bucket_strategy="multiples",
        n_atoms_bucket_strategy=None,
        n_pair_bucket_strategy=None,
        k_sel_bucket_strategy=None,
        extra_neighbors=4,
        num_neighbors_adaptive=None,
        cutoff_override=None,
        cutoff_widen=0.5,
        add_offset=True,
        debug=False,
        express=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Performance switches (see express.py); the model-level ones become
        # module attributes, so the stock model is untouched when all are off.
        self.express = Express.from_any(express)
        updates = self.express.model_updates()
        self._model = model.clone(**updates) if updates else model
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
        # Per-atom adaptive-selection target; None keeps the trained value. Fed
        # to both consumers (k_sel sizing and the forward), which must agree.
        self._num_neighbors_adaptive = (
            num_neighbors_adaptive
            if num_neighbors_adaptive is not None
            else metadata["config"]["num_neighbors_adaptive"]
        )
        self._add_offset = add_offset
        self._debug = debug
        # Perf-tuning numbers, refreshed each NL rebuild (None until first
        # `calculate`). Always populated; `debug` only gates the stderr summary.
        self.debug_stats = None
        # Cached max selected adaptive cutoff from the last sizing pass;
        # carried across rebuilds that reuse k_sel (which skip the sizing run).
        self._max_selected_cutoff = None
        # Per-call reach of the selection, read off the forward every step:
        # the largest adaptive cutoff and the largest selected count. What the
        # cutoff guard and the benchmark driver watch.
        self.call_stats = None
        self._device_nl = self.express.neighbor_list == "device"
        self._cutoff_widen = cutoff_widen
        self._cutoff_guard_events = 0
        # (cutoffs, positions, cell) of the last call, for express.seed_cutoffs.
        self._seed = None

        # cutoff_override narrows ONLY the raw-NL radius (to cutoff_override +
        # skin for the host list, cutoff_override for the device list). The UPET
        # model is untouched — the adaptive-cutoff selection and the cutoff bump
        # all keep using the trained config["cutoff"]. A performance knob for when the trained cutoff is wider
        # than the radius adaptive selection actually reaches.
        #
        # CORRECTNESS RISK: the raw NL must still hold every pair the model
        # uses. Adaptive per-atom cutoffs range up to config["cutoff"], so if
        # cutoff_override is below the largest one selected, surviving pairs are
        # silently dropped — and the adaptive-cutoff counts themselves are
        # starved. Both corrupt energy/forces with no error raised.
        #
        # `_check_cutoff_override` warns (per rebuild) once the measured reach
        # exceeds cutoff_override — a warning; the cutoff guard (express) is the
        # guard.
        self._cutoff = (
            cutoff_override if cutoff_override is not None else metadata["config"]["cutoff"]
        )

        if default_dtype == "float64":
            jax.config.update("jax_enable_x64", True)
        self._dtype = jnp.float64 if default_dtype == "float64" else jnp.float32

        # fp32 matmul accumulation on accelerators; "high" for NVE energy
        # conservation (no-op on CPU). None leaves the global config alone.
        self._matmul_precision = matmul_precision
        if matmul_precision is not None:
            jax.config.update("jax_default_matmul_precision", matmul_precision)

        # Cast params to the requested dtype so fp64 callers don't silently
        # run mixed-precision (fp32 weights × fp64 activations) with a PET-MAD
        # checkpoint that ships as fp32.
        self._params = cast_floats(params, self._dtype)

        # The device list is rebuilt every step: the cache only notices atom
        # count, species and pbc changes.
        self._nl_cache = NeighborListCache(skin=np.inf if self._device_nl else skin)
        self._predict_fn = get_predict_fn(
            self._model,
            stress=self._stress,
            no_shadow=self._no_shadow,
            num_neighbors_adaptive=self._num_neighbors_adaptive,
            express=self.express,
        )
        self._N_padded = None
        self._n_pair_padded = None
        self._k_sel = None
        # Static capacities of the flat token list and the extra attention
        # tiles (empty for the padded layout), sized next to k_sel.
        self._token_sizes = {}
        self._structure = None
        self._shift_offset = 0.0

        self._shifts = {int(z): float(v) for z, v in metadata["shifts"].items()}

        if not stress:
            self.implemented_properties = ["energy", "forces"]

    @classmethod
    def from_checkpoint(cls, folder, direct_forces=False, direct_stress=False, **kwargs):
        from .convert import load_checkpoint
        from .model import UPET

        params, metadata = load_checkpoint(folder)
        model = UPET(
            **metadata["config"],
            direct_forces=direct_forces,
            direct_stress=direct_stress,
        )
        return cls(model, params, metadata, **kwargs)

    def check_state(self, atoms, tol=1e-15):
        # ASE only compares geometry, so a new charge or spin multiplicity in
        # atoms.info would otherwise return the cached results.
        changes = super().check_state(atoms, tol=tol)
        if self._model.system_conditioning and self.atoms is not None:
            if read_conditioning(atoms) != read_conditioning(self.atoms):
                changes.append("conditioning")
        return changes

    def calculate(self, atoms=None, properties=None, system_changes=None, **kwargs):
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("No atoms provided")

        if self._nl_cache.needs_update(atoms):
            self._build_structure(atoms)
        else:
            self._update_geometry(atoms)

        results = self._predict(atoms)
        if self.express.cutoff_guard:
            while self._cutoff_violated(atoms, results):
                self._cutoff = min(
                    self._cutoff + self._cutoff_widen, self._metadata["config"]["cutoff"]
                )
                self._cutoff_guard_events += 1
                self._build_structure(atoms, force_recompute_k_sel=True)
                results = self._predict(atoms)

        self.call_stats = {
            "max_cutoff": float(results["max_cutoff"]),
            "k_max": int(results["k_max"]),
        }
        if self.express.seed_cutoffs:
            cell = np.array(atoms.get_cell()[:])
            self._seed = (results["atomic_cutoffs"], atoms.get_positions().copy(), cell)

        n_real = len(atoms)
        energy = float(results["energy"])
        if self._add_offset:
            energy += self._shift_offset
        forces = np.array(results["forces"][:n_real], dtype=np.float64)
        if self._model.direct_forces:
            # Direct forces: remove the spurious net force (per-structure mean).
            forces = forces - forces.mean(axis=0)

        self.results = {"energy": energy, "forces": forces}

        if self._stress and "stress" in results and atoms.pbc.any():
            virial = np.array(results["stress"], dtype=np.float64)
            volume = atoms.get_volume()
            self.results["stress"] = full_3x3_to_voigt_6_stress(virial / volume)

        return self.results

    # -- internals --

    def _predict(self, atoms):
        """The forward, with one rebuild-and-retry on overflow."""
        results = self._predict_fn(self._params, self._structure)
        if bool(results["overflow"]):
            self._build_structure(atoms, force_recompute_k_sel=True)
            results = self._predict_fn(self._params, self._structure)
            if bool(results["overflow"]):
                raise RuntimeError(
                    "UPETCalculator: cannot recover from overflow "
                    f"after retry (k_sel={self._k_sel}, "
                    f"pair capacity={self._n_pair_padded})"
                )
        return results

    def _cutoff_violated(self, atoms, results):
        """Whether the raw list may have dropped a pair the selection needs:
        the largest adaptive cutoff at or above the radius the list is
        complete to. Never with the raw radius at the trained cutoff."""
        if self._cutoff >= self._metadata["config"]["cutoff"]:
            return False
        if self._device_nl:
            complete_to = self._cutoff
        else:
            complete_to = self._cutoff + self._skin - self._nl_cache.pair_drift(atoms)
        return float(results["max_cutoff"]) >= complete_to

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
        * k_sel — recomputed via ``select_edges`` (expensive: a CPU
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
            skin=self._skin,
            bucket_strategy=self._bucket_strategy,
            n_atoms_bucket_strategy=self._n_atoms_bucket_strategy,
            n_pair_bucket_strategy=self._n_pair_bucket_strategy,
            atom_block=self.express.atom_block,
        )

        N_padded = structure["positions"].shape[0]
        n_pair_padded = structure["centers"].shape[0]

        shape_changed = (
            self._N_padded is None
            or N_padded != self._N_padded
            or n_pair_padded != self._n_pair_padded
        )

        k_sel_actual = None  # set only when the sizing pass runs (debug stat)
        if force_recompute_k_sel or shape_changed:
            # The trained self._model.cutoff, NOT self._cutoff: cutoff_override
            # narrows only the raw-NL radius, never the selection's reach.
            selection = select_edges(
                structure,
                self._num_neighbors_adaptive,
                self._model.cutoff,
                self._model.cutoff_width_adaptive,
                method=self._model.adaptive_cutoff_method,
                express=self.express,
            )
            k_sel_actual = max(int(selection.counts.max()), 1)
            # The selection's real reach, for tuning cutoff_override.
            self._max_selected_cutoff = float(
                selection.pair_cutoffs[selection.selected].max(initial=0.0)
            )
            # T = k_sel edge tokens + 1 central-atom token. Bucket T (an even
            # T keeps attention on XLA's fused fast path); k_sel = T - 1.
            token_padded = _bucket_or(
                k_sel_actual + 1 + self._extra_neighbors, self._k_sel_bucket_strategy
            )
            k_sel_padded = token_padded - 1
            self._token_sizes = self._size_tokens(selection.counts, k_sel_padded)
        else:
            k_sel_padded = self._k_sel

        self._N_padded = N_padded
        self._n_pair_padded = n_pair_padded
        self._k_sel = k_sel_padded

        # k_sel_sizer carries k_sel into the shape-agnostic predict_fn via its
        # shape; the JIT retraces when k_sel (or any other shape) changes.
        # The largest image shift of the host list bounds how far a cell change
        # moves any pair distance (the seed's Verlet bound and the cache's).
        max_shift = (
            int(np.max(np.abs(structure["cell_shifts"])))
            if len(structure["cell_shifts"]) > 0
            else 0
        )
        self._max_shift = max_shift
        geometry = self._device_inputs(structure, atoms) if self._device_nl else structure
        self._structure = {
            **self._to_jax_structure(geometry),
            **self._conditioning(atoms, N_padded),
            **self._seed_inputs(atoms, N_padded, max_shift),
            "k_sel_sizer": jnp.zeros(k_sel_padded, dtype=bool),
        }
        # token_sizer / tile_sizer carry the flat layout's capacities likewise.
        for key, size in (("token_sizer", "p_cap"), ("tile_sizer", "e_cap")):
            if size in self._token_sizes:
                self._structure[key] = jnp.zeros(self._token_sizes[size], dtype=bool)
        # The device list is rebuilt with its shifts every step, so a cell
        # change never stales it: the cache only watches the atom count,
        # species and pbc.
        self._nl_cache.save_reference(
            atoms, max_cell_shift=0 if self._device_nl else max_shift
        )

        # Composition shifts are constant w.r.t. positions; Python fp64 sum
        # added post-JIT. Species can only change via a rebuild, so here. The
        # KeyError doubles as the guard for untrained elements — this runs before
        # the forward, where an out-of-range Z embedding would clamp silently.
        try:
            self._shift_offset = sum(
                self._shifts[int(z)] for z in atoms.get_atomic_numbers()
            )
        except KeyError as exc:
            raise ValueError(
                f"UPETCalculator: structure contains atomic number {exc.args[0]}, "
                f"which the model was not trained on."
            ) from exc

        self._record_debug(
            atoms, structure, k_sel_actual, shape_changed, force_recompute_k_sel
        )
        self._check_cutoff_override()

    def _size_tokens(self, counts, k_sel):
        """Capacities of the flat token list (``p_cap``) and the extra attention
        tiles (``e_cap``) from the host-side selected counts, sized and widened
        together with ``k_sel``.

        ``p_cap`` is ``token_margin`` times the selected pairs, bucketed like
        the pair axis, plus the always-empty sentinel slot: the total is a sum
        over all atoms and drifts far less than the largest count. ``e_cap``
        is ``tile_margin`` times the atoms spilling past the tile, at least
        ``tile_floor``, so one atom crossing the tile width does not overflow
        every step; it is 0 (no extra path in the graph) when ``k_sel`` fits
        in one tile. An atom past two tiles cannot be fixed by resizing, so it
        raises here.
        """
        express = self.express
        if express.tokens != "flat":
            return {}
        n_tokens = int(counts.sum())
        strategy = (
            self._n_pair_bucket_strategy
            if self._n_pair_bucket_strategy is not None
            else self._bucket_strategy
        )
        p_cap = _bucket_or(math.ceil(express.token_margin * n_tokens) + 1, strategy)
        sizes = {"n_tokens": n_tokens, "p_cap": p_cap}
        width = express.attention_tile
        if width is not None:
            k_max = int(counts.max())
            if k_max > 2 * (width - 1):
                raise ValueError(
                    f"UPETCalculator: an atom selects {k_max} neighbours, more than "
                    f"the {2 * (width - 1)} that two attention tiles of width "
                    f"{width} hold; use attention_tile >= {math.ceil(k_max / 2) + 1}."
                )
            n_extra = int((counts > width - 1).sum())
            e_cap = 0
            if k_sel > width - 1:
                e_cap = max(express.tile_floor, math.ceil(express.tile_margin * n_extra))
            sizes.update(n_extra=n_extra, e_cap=e_cap)
        return sizes

    def _record_debug(
        self, atoms, structure, k_sel_actual, shape_changed, force_recompute_k_sel
    ):
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
            "cutoff_guard_events": self._cutoff_guard_events,
            "express": self.express.as_dict(),
            "tokens": dict(self._token_sizes),
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
        if self.express.cutoff_guard or co is None or not override_active:
            return
        if co <= self._cutoff:
            return
        skin = 0.0 if self._device_nl else self._skin
        warnings.warn(
            f"UPETCalculator: adaptive selection reaches {co:.2f} Å, exceeding "
            f"cutoff_override={self._cutoff:.2f} Å (raw-NL radius "
            f"{self._cutoff + skin:.2f} Å incl. {skin:.2f} Å skin). The override "
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
        """Update positions, cell, and the conditioning inputs without
        rebuilding the NL (for the device list, also its shift window)."""
        n_real = len(atoms)
        N_padded = self._N_padded
        positions = np.zeros((N_padded, 3), dtype=np.float64)
        positions[:n_real] = atoms.get_positions()
        cell = np.array(atoms.get_cell()[:], dtype=np.float64)
        window = {}
        max_shift = self._max_shift
        if self._device_nl and atoms.pbc.any():
            window["lattice_shifts"] = lattice_shifts(cell, atoms.pbc, self._cutoff)
            # The current window bounds every pair's image shift at this cell.
            max_shift = int(np.abs(window["lattice_shifts"]).max())
        self._structure = {
            **self._structure,
            "positions": jnp.array(positions, dtype=self._dtype),
            "cell": jnp.array(cell, dtype=self._dtype),
            **window,
            **self._conditioning(atoms, N_padded),
            **self._seed_inputs(atoms, N_padded, max_shift),
        }

    def _seed_inputs(self, atoms, N_padded, max_cell_shift):
        """``cutoff_seed`` and ``seed_displacement`` for ``seed_cutoffs``: the
        last call's cutoffs and half the largest change of any pair distance
        since (atom displacements plus the cell change through the largest
        image shift). Always present, so the forward is traced once; ``inf``
        when there is no seed (first call, other atoms)."""
        if not self.express.seed_cutoffs:
            return {}
        cutoffs, delta = jnp.zeros(N_padded, self._dtype), np.inf
        if self._seed is not None and self._seed[0].shape[0] == N_padded:
            cutoffs, positions, cell = self._seed
            if len(positions) == len(atoms):
                moved = np.linalg.norm(atoms.get_positions() - positions, axis=1).max()
                strained = np.linalg.norm(atoms.get_cell()[:] - cell, axis=1).sum()
                delta = moved + 0.5 * (max_cell_shift or 0) * strained
        return {
            "cutoff_seed": cutoffs,
            "seed_displacement": jnp.asarray(delta, self._dtype),
        }

    def _device_inputs(self, structure, atoms):
        """The structure for the device list: the host list's geometry, with
        its pair arrays replaced by what ``select.truncate`` builds them from.
        ``pair_sizer`` carries the pair capacity (the host list's padded pair
        count) the way ``k_sel_sizer`` carries ``k_sel``."""
        keys = ("positions", "cell", "atomic_numbers", "atom_mask")
        return {
            **{k: structure[k] for k in keys},
            "pbc": np.asarray(atoms.pbc, dtype=bool),
            "lattice_shifts": lattice_shifts(structure["cell"], atoms.pbc, self._cutoff),
            "neighbor_radius": np.asarray(self._cutoff, dtype=np.float64),
            "pair_sizer": np.zeros(structure["centers"].shape[0], dtype=bool),
        }

    def _conditioning(self, atoms, N_padded):
        """Charge / spin-multiplicity inputs for a conditioned model, read off
        ``atoms.info`` on every step (values only, never a shape, so no
        retrace); nothing for an unconditioned one, so its JIT inputs are
        unchanged."""
        if not self._model.system_conditioning:
            return {}
        return conditioning_inputs(
            atoms, N_padded, self._model.max_charge, self._model.max_spin_multiplicity
        )


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

    def pair_drift(self, atoms):
        """Bound on how far any pair distance (periodic images included) moved
        since the reference: twice the largest displacement plus the cell
        change times the largest cell shift."""
        displacements = atoms.get_positions() - self._ref_positions
        max_disp = float(np.linalg.norm(displacements, axis=1).max())
        if self._max_cell_shift is not None and self._max_cell_shift > 0:
            cell_change = np.array(atoms.get_cell()[:]) - self._ref_cell
            cell_vector_norms = np.linalg.norm(cell_change, axis=1)
            max_cell_contrib = float(self._max_cell_shift * cell_vector_norms.sum())
        else:
            max_cell_contrib = 0.0
        return 2 * max_disp + max_cell_contrib

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

        return bool(self.pair_drift(atoms) > self.skin)

    def save_reference(self, atoms, max_cell_shift=None):
        self._ref_positions = atoms.get_positions().copy()
        self._ref_cell = np.array(atoms.get_cell()[:]).copy()
        self._ref_pbc = atoms.get_pbc().copy()
        self._ref_numbers = atoms.get_atomic_numbers().copy()
        self._max_cell_shift = max_cell_shift
