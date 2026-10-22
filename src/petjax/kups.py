"""Run UPET as a kUPS potential.

kUPS (https://github.com/cusp-ai-oss/kups) rebuilds the raw neighbour list on
the device every step, batches independent systems in one program, and
differentiates an energy function over positions and cell. This module is
that energy function plus the two factories kUPS's own model bridges
(``tojax``, ``torch``) offer. Optional: ``pip install "pet-jax[kups]"``,
Python >= 3.12.

Per call, all inside kUPS's jit:

1. pad the particle table by one dummy row. kUPS's out-of-bounds sentinel
   index is the atom count, which is exactly pet-jax's dummy atom;
2. take per-edge displacements off the ``Edges`` (per-system cells), sort by
   center and build the reverse map on the device;
3. run ``truncate_edges`` at the static ``k_sel`` held on the state. Its
   ``overflow`` becomes a kUPS runtime assertion whose fix grows ``k_sel``
   (``UPETParameters``) and reruns the step, the way kUPS grows its own
   neighbour-list capacities;
4. sum the per-atom energies per system.

The calculator's two-phase design (host-side sizing, in-JIT selection) is
not needed here: kUPS's assertion-and-fix loop is the sizing.
"""

import numpy as np
import jax
import jax.numpy as jnp

from kups.core.capacity import Capacity, LensCapacity
from kups.core.cell import AnyPeriodicity
from kups.core.data import Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    IsNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborList,
    NeighborListFactory,
    NeighborListPoints,
)
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import EMPTY_LENS, Energy, Potential, PotentialOut
from kups.core.typing import HasAtomicNumbers, HasCell, IsState, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field
from kups.core.utils.kahan import KahanSummand
from kups.potential.common.energy import FullSumComposer, PotentialFromEnergy
from kups.potential.common.geometry import Geometry, PositionsAndCell
from kups.potential.common.graph import (
    GRAPH_GEOMETRY,
    GraphConstructor,
    GraphInputConstructor,
    GraphPotentialInput,
)

from typing import Any, Literal, NamedTuple, Protocol

from .convert import load_checkpoint
from .model import UPET
from .select import truncate_edges
from .structure import center_order, corresponding_edges
from .utils import cast_floats

# -- the model bundle and the sizing state --


class IsUPETParticles(NeighborListPoints, HasAtomicNumbers, Protocol):
    """Particle protocol: positions, atomic numbers, system, inclusion, exclusion."""


class Conditioning(Protocol):
    """Per-system total charge and spin multiplicity for a conditioned
    checkpoint, read off the systems table. Integer arrays of shape
    ``(n_systems,)``, validated by the caller: an out-of-range value indexes
    an embedding table inside jit and yields NaN rather than an error."""

    def __call__(self, systems: Table[SystemId, Any]) -> tuple[jax.Array, jax.Array]: ...


@dataclass
class UPETParameters:
    """Sizing state for the adapter, held on the simulation state next to
    kUPS's ``neighborlist_params``.

    Attributes:
        k_sel: Neighbours per atom the selected list is packed to. Grown by
            the overflow fix; start it near the checkpoint's
            ``num_neighbors_adaptive`` times two.
        growth_base: Growth factor of the fix: ``k_sel`` becomes the next
            power of this base at or above the largest selected count.
    """

    k_sel: int = field(static=True)
    growth_base: float = field(static=True, default=1.25)


@dataclass
class UPETMliap:
    """A converted checkpoint as kUPS sees it.

    Attributes:
        model: The ``UPET`` module, with the adaptive-selection hypers.
        params: Flax parameter tree, in the working dtype.
        cutoff: Per-system raw neighbour-list radius. The trained cutoff
            unless narrowed by ``cutoff_override`` (same risk as on the
            calculator: the raw list must hold every pair selection reaches).
        shifts: Composition shift per atomic number, ``(max_atomic_number + 1,)``,
            NaN for species the checkpoint was not trained on so that such a
            structure fails loudly; ``None`` for raw model energies.
        num_neighbors_adaptive: Per-atom selection target.
        conditioning: Reads charge and spin multiplicity off the systems table
            for a conditioned checkpoint; ``None`` otherwise.
        no_shadow: Cut gradients through the adaptive cutoffs.
    """

    model: UPET = field(static=True)
    params: Any
    cutoff: Table[SystemId, jax.Array]
    shifts: jax.Array | None
    num_neighbors_adaptive: int = field(static=True)
    conditioning: Conditioning | None = field(static=True, default=None)
    no_shadow: bool = field(static=True, default=False)

    @classmethod
    def from_checkpoint(
        cls,
        folder,
        *,
        dtype=jnp.float32,
        add_offset=True,
        cutoff_override=None,
        num_neighbors_adaptive=None,
        conditioning=None,
        no_shadow=False,
    ) -> "UPETMliap":
        """Load a pet-jax checkpoint directory. ``dtype`` is the working
        dtype the params are cast to and inputs are cast to at the boundary;
        kUPS applications run under x64, PET-MAD checkpoints ship fp32."""
        params, metadata = load_checkpoint(folder)
        config = metadata["config"]
        model = UPET(**config)
        if model.system_conditioning and conditioning is None:
            raise ValueError(
                "this checkpoint is charge/spin conditioned; pass `conditioning` "
                "to read the per-system values off the systems table."
            )
        shifts = None
        if add_offset:
            shifts = np.full(model.max_atomic_number + 1, np.nan, dtype=np.float64)
            for z, value in metadata["shifts"].items():
                shifts[int(z)] = value
            shifts = jnp.asarray(shifts, dtype=dtype)
        cutoff = config["cutoff"] if cutoff_override is None else cutoff_override
        return cls(
            model=model,
            params=cast_floats(params, dtype),
            cutoff=Table((SystemId(0),), jnp.array([float(cutoff)])),
            shifts=shifts,
            num_neighbors_adaptive=(
                config["num_neighbors_adaptive"]
                if num_neighbors_adaptive is None
                else num_neighbors_adaptive
            ),
            conditioning=conditioning,
            no_shadow=no_shadow,
        )

    @property
    def dtype(self):
        return jax.tree.leaves(self.params)[0].dtype


class UPETInput(NamedTuple):
    """What the energy function gets as ``parameters``: the bundle and the
    capacity that carries ``k_sel`` and its fix."""

    mliap: UPETMliap
    k_sel: Capacity[int]


UPETGraphInput = GraphPotentialInput[
    UPETInput, IsUPETParticles, HasCell[AnyPeriodicity], Literal[2]
]


# -- the energy function --


def upet_energy(inp: UPETGraphInput) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Per-system UPET energies on a kUPS graph; see the module docstring for
    the steps. Differentiable in positions and cell through the adaptive
    cutoffs (``truncate_edges`` carries the implicit-function-theorem step)."""
    mliap, k_sel = inp.parameters
    model, dtype = mliap.model, mliap.dtype
    particles, systems, edges = inp.graph.particles, inp.graph.systems, inp.graph.edges
    p = particles.data
    n = particles.size

    # The dummy atom is row n, where kUPS already points every padded edge.
    valid = p.system.valid_mask
    atom_mask = jnp.pad(valid, (0, 1), constant_values=False)
    atomic_numbers = jnp.pad(p.atomic_numbers, (0, 1), constant_values=0)

    indices = edges.indices.indices_in(particles.keys)
    pair_mask = (indices < n).all(axis=-1)
    R_ij = edges.difference_vectors(particles, systems)[:, 0]
    order = center_order(indices[:, 0], pair_mask)
    centers, others = indices[order, 0], indices[order, 1]
    cell_shifts, pair_mask, R_ij = edges.shifts[order, 0], pair_mask[order], R_ij[order]
    reverse = corresponding_edges(centers, others, cell_shifts, pair_mask)

    truncated, truncation = truncate_edges(
        R_ij.astype(dtype),
        centers,
        others,
        reverse,
        pair_mask,
        atomic_numbers,
        atom_mask,
        k_sel.size,
        mliap.num_neighbors_adaptive,
        model.cutoff,
        model.cutoff_width_adaptive,
        method=model.adaptive_cutoff_method,
        no_shadow=mliap.no_shadow,
    )
    # pet-jax's overflow signal as a kUPS assertion: the fix grows k_sel on
    # the state and the step is rerun at the new static width.
    k_sel.generate_assertion(truncation.counts.max())

    conditioning = {}
    if mliap.conditioning is not None:
        charge, spin_multiplicity = mliap.conditioning(systems)
        per_atom = p.system.raw_indices
        conditioning = {
            "charge": jnp.pad(charge[per_atom], (0, 1)),
            "spin_multiplicity": jnp.pad(spin_multiplicity[per_atom], (0, 1)),
        }

    energies = model.apply(mliap.params, **truncated, **conditioning)
    energies = energies[:n].astype(p.positions.dtype)
    if mliap.shifts is not None:
        shifts = mliap.shifts[p.atomic_numbers].astype(energies.dtype)
        energies = energies + jnp.where(valid, shifts, 0.0)
    return WithPatch(p.system.sum_over(energies), IdPatch[Any]())


# -- factories, mirroring kups.potential.mliap.tojax --


def make_upet_potential(
    particles_view: View[Any, Table[ParticleId, IsUPETParticles]],
    systems_view: View[Any, Table[SystemId, HasCell[AnyPeriodicity]]],
    neighborlist_view: View[Any, NeighborList[Literal[2]]],
    model_view: View[Any, UPETMliap],
    k_sel_view: View[Any, Capacity[int]],
    gradient_lens: Lens[UPETGraphInput, Any],
    hessian_lens: Lens[Any, Any],
    hessian_idx_view: View[Any, Any],
    patch_idx_view: View[Any, PotentialOut[Any, Any]] | None = None,
    out_cache_lens: Lens[Any, KahanSummand[PotentialOut[Any, Any]]] | None = None,
) -> PotentialFromEnergy[Any, UPETGraphInput, Any, Any, Patch[Any]]:
    """The state-agnostic factory: views into the state for particles,
    systems, a cutoff-bound neighbour list, the model bundle and the
    ``k_sel`` capacity, plus the gradient and Hessian lenses kUPS wants."""
    graph = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=neighborlist_view,
        probe=None,
    )
    composer = FullSumComposer(
        GraphInputConstructor(graph, lambda s: UPETInput(model_view(s), k_sel_view(s)))
    )
    return PotentialFromEnergy(
        composer=composer,
        energy_fn=upet_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


class IsUPETState(
    IsState[IsUPETParticles, HasCell[AnyPeriodicity]],
    IsNeighborListState[IsUniversalNeighborlistParams],
    Protocol,
):
    """Particles, systems, neighbour-list parameters and ``upet_params``."""

    @property
    def upet_params(self) -> UPETParameters: ...


def make_upet_from_state(
    state: Lens[Any, IsUPETState],
    *,
    parameters: UPETMliap,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
    neighborlist_factory: NeighborListFactory[Any] = AdaptiveNeighborList.from_state,
) -> Potential[Any, Any, Any, Patch[Any]]:
    """Bind the factory to a state carrying ``particles``, ``systems``,
    ``neighborlist_params`` and ``upet_params``. ``gradient`` is a kUPS
    relaxation filter (``POSITIONS_AND_CELL``, ``FRECHET_FILTER``, ...);
    ``None`` computes energies only. The neighbour list is built at the
    bundle's cutoff; the ``k_sel`` capacity's fix writes back through
    ``state``."""
    gradient_lens = EMPTY_LENS if gradient is None else GRAPH_GEOMETRY.nest(gradient)
    k_sel_lens = state.focus(lambda x: x.upet_params).focus(lambda x: x.k_sel)

    def k_sel_view(s) -> Capacity[int]:
        sizing = state(s).upet_params
        return LensCapacity(sizing.k_sel, k_sel_lens, base=sizing.growth_base)

    def neighborlist_view(s) -> NeighborList[Literal[2]]:
        return neighborlist_factory(state(s), parameters.cutoff)

    return make_upet_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        neighborlist_view,
        lambda s: parameters,
        k_sel_view,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
    )


__all__ = [
    "UPETMliap",
    "UPETParameters",
    "UPETInput",
    "IsUPETParticles",
    "IsUPETState",
    "upet_energy",
    "make_upet_potential",
    "make_upet_from_state",
]
