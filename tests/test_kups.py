"""The kUPS adapter against the ASE calculator: same checkpoint, same
structures, energies, forces and virial; batched systems equal separate
ones; the overflow fix grows k_sel through kUPS's assertion loop."""

import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase import Atoms
from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress

kups = pytest.importorskip("kups")

from kups.application.potential.filter import POSITIONS_AND_CELL  # noqa: E402
from kups.core.cell import Cell, TriclinicFrame, to_lower_triangular  # noqa: E402
from kups.core.data import Index, Table  # noqa: E402
from kups.core.lens import lens  # noqa: E402
from kups.core.neighborlist import UniversalNeighborlistParameters  # noqa: E402
from kups.core.result import as_result_function  # noqa: E402
from kups.core.typing import ExclusionId, InclusionId, ParticleId, SystemId  # noqa: E402
from kups.core.utils.jax import dataclass  # noqa: E402

from petjax import UPETCalculator  # noqa: E402
from petjax.kups import UPETMliap, UPETParameters, make_upet_from_state  # noqa: E402
from petjax.structure import read_conditioning  # noqa: E402

# Frames of the mini set: three periodic cells of different sizes and one
# molecule (frame 5 has no cell and no pbc).
PERIODIC = (0, 1, 3)
MOLECULE = 5
VACUUM_BOX = 30.0


@dataclass
class Particles:
    positions: jax.Array
    atomic_numbers: jax.Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class Systems:
    cell: Cell
    charge: jax.Array
    spin_multiplicity: jax.Array


@dataclass
class State:
    particles: Table[ParticleId, Particles]
    systems: Table[SystemId, Systems]
    neighborlist_params: UniversalNeighborlistParameters
    upet_params: UPETParameters


def conditioning(systems):
    return systems.data.charge, systems.data.spin_multiplicity


def _kups_frame(atoms):
    """The same structure in kUPS's frame: lower-triangular cell, positions
    rotated along, a bounding box for a molecule. Energies are invariant,
    forces compare in this frame."""
    if not atoms.pbc.any():
        rotated = Atoms(
            atoms.numbers, positions=atoms.positions, cell=np.eye(3) * VACUUM_BOX
        )
    else:
        L, rotate = to_lower_triangular(jnp.asarray(atoms.cell.array))
        positions = np.asarray(rotate(jnp.asarray(atoms.positions)))
        rotated = Atoms(
            atoms.numbers, positions=positions, cell=np.asarray(L), pbc=atoms.pbc
        )
    rotated.info = dict(atoms.info)
    return rotated


def _state(frames, cutoff, k_sel):
    """One kUPS state holding ``frames`` (all in kUPS's frame, same pbc)."""
    pbc = tuple(bool(x) for x in frames[0].pbc)
    assert all(tuple(bool(x) for x in f.pbc) == pbc for f in frames)
    n = sum(len(f) for f in frames)
    system = np.repeat(np.arange(len(frames)), [len(f) for f in frames])
    particles = Table.arange(
        Particles(
            positions=jnp.asarray(np.concatenate([f.positions for f in frames])),
            atomic_numbers=jnp.asarray(np.concatenate([f.numbers for f in frames])),
            system=Index.integer(system, n=len(frames), label=SystemId),
            inclusion=Index.integer(system, n=len(frames), label=InclusionId),
            exclusion=Index.arange(n, label=ExclusionId),
        ),
        label=ParticleId,
    )
    cell = Cell.from_pbc(
        TriclinicFrame.from_matrix(jnp.asarray(np.stack([f.cell.array for f in frames]))),
        pbc,
    )
    charge, spin = zip(*(read_conditioning(f) for f in frames))
    systems = Table.arange(
        Systems(cell=cell, charge=jnp.asarray(charge), spin_multiplicity=jnp.asarray(spin)),
        label=SystemId,
    )
    cutoffs = Table.arange(jnp.full(len(frames), cutoff), label=SystemId)
    counts = Table.arange(jnp.asarray([len(f) for f in frames]), label=SystemId)
    nl_params = UniversalNeighborlistParameters.estimate(counts, systems, cutoffs)
    return State(particles, systems, nl_params, UPETParameters(k_sel=k_sel))


def _evaluate(potential, state):
    """Run with kUPS's assertion loop: fix capacities until the result holds."""
    fn = as_result_function(potential)
    for _ in range(10):
        result = fn(state)
        if not result.failed_assertions:
            return result.value.data, state
        state = result.fix_or_raise(state)
    raise RuntimeError("capacities did not settle")


def _virial(state, out):
    """Lower triangle of ``dE/d(strain) = r^T g + h^T G`` from kUPS's
    gradients, the way its stress observable assembles it. kUPS keeps ``G`` on
    the six lower-triangular entries of ``h`` only, which fixes the lower
    triangle of ``h^T G`` exactly for lower-triangular ``h``. The upper
    triangle is what a rotation-invariant model would make symmetric; PET is
    not one, so only the lower triangle is comparable to the raw virial."""
    r = np.asarray(state.particles.data.positions, dtype=np.float64)
    g = np.asarray(out.gradients.positions.data, dtype=np.float64)
    system = np.asarray(state.particles.data.system.raw_indices)
    cell = state.systems.data.cell
    h = np.asarray(cell.vectors, dtype=np.float64)
    G = np.asarray(
        cell.frame.vectors_gradient(out.gradients.cell.data.frame), dtype=np.float64
    )
    virial = np.zeros_like(h)
    for s in range(h.shape[0]):
        virial[s] = np.tril(r[system == s].T @ g[system == s] + h[s].T @ G[s])
    return virial


@pytest.fixture(scope="module")
def mliap(pet_mad_xs_checkpoint):
    return UPETMliap.from_checkpoint(pet_mad_xs_checkpoint)


@pytest.fixture(scope="module")
def calculator(pet_mad_xs_checkpoint):
    return UPETCalculator.from_checkpoint(pet_mad_xs_checkpoint, stress=True)


@pytest.fixture(scope="module")
def frames(mini_xyz):
    return [_kups_frame(a) for a in read(str(mini_xyz), index=":")]


@pytest.fixture(scope="module")
def potential(mliap):
    return make_upet_from_state(
        lens(lambda s: s, cls=State), parameters=mliap, gradient=POSITIONS_AND_CELL
    )


def _reference(calculator, atoms):
    """Energy, forces and the raw strain derivative ``dE/d(strain)``. The raw
    virial is read off the calculator's forward rather than its Voigt stress:
    PET is not rotationally equivariant by construction, so the raw virial is
    not symmetric, and the Voigt form averages the two triangles."""
    atoms = atoms.copy()
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    virial = None
    if atoms.pbc.any():
        raw = calculator._predict_fn(calculator._params, calculator._structure)["stress"]
        virial = np.asarray(raw, dtype=np.float64)
        symmetrized = voigt_6_to_full_3x3_stress(atoms.get_stress()) * atoms.get_volume()
        np.testing.assert_allclose(0.5 * (virial + virial.T), symmetrized, atol=1e-5)
    return energy, forces, virial


@pytest.mark.parametrize("index", (*PERIODIC, MOLECULE))
def test_matches_calculator(frames, mliap, calculator, potential, index):
    """Energy, forces and the lower triangle of the raw virial of one
    structure agree with the calculator to fp32 rounding (the raw lists differ:
    dense on the device vs vesin with a skin, but the adaptive selection sees
    the same pairs)."""
    atoms = frames[index]
    energy, forces, virial = _reference(calculator, atoms)
    state = _state([atoms], mliap.model.cutoff, k_sel=1)
    out, state = _evaluate(potential, state)

    assert out.total_energies.keys == (SystemId(0),)
    np.testing.assert_allclose(out.total_energies.data[0], energy, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(-out.gradients.positions.data, forces, atol=2e-4)
    if virial is not None:
        np.testing.assert_allclose(_virial(state, out)[0], np.tril(virial), atol=2e-3)


def test_batched_equals_separate(frames, mliap, potential):
    """Three cells in one state give each cell's own energy and forces."""
    batch = [frames[i] for i in PERIODIC]
    state = _state(batch, mliap.model.cutoff, k_sel=1)
    out, state = _evaluate(potential, state)
    system = np.asarray(state.particles.data.system.raw_indices)
    assert out.total_energies.keys == tuple(SystemId(i) for i in range(len(batch)))
    for s, atoms in enumerate(batch):
        single, single_state = _evaluate(potential, _state([atoms], mliap.model.cutoff, 1))
        np.testing.assert_allclose(
            out.total_energies.data[s], single.total_energies.data[0], rtol=1e-6, atol=1e-5
        )
        np.testing.assert_allclose(
            out.gradients.positions.data[system == s],
            single.gradients.positions.data,
            atol=1e-5,
        )


@pytest.mark.parametrize("index", (PERIODIC[0], MOLECULE))
def test_conditioned_matches_calculator(assets_dir, index):
    """A charge/spin conditioned checkpoint reads the per-system values off
    the systems table and agrees with the calculator reading ``atoms.info``."""
    checkpoint = assets_dir / "checkpoints" / "pet-omol-s"
    dataset = assets_dir / "test_mini_conditioned.xyz"
    if not (checkpoint / "model.msgpack").exists() or not dataset.exists():
        pytest.skip("pet-omol-s checkpoint or conditioned mini set missing")
    atoms = _kups_frame(read(str(dataset), index=index))
    assert atoms.info.get("charge") not in (None, 0) or atoms.info.get("spin_multiplicity")
    calculator = UPETCalculator.from_checkpoint(checkpoint)
    energy, forces, _ = _reference(calculator, atoms)

    mliap = UPETMliap.from_checkpoint(checkpoint, conditioning=conditioning)
    potential = make_upet_from_state(
        lens(lambda s: s, cls=State), parameters=mliap, gradient=POSITIONS_AND_CELL
    )
    out, _ = _evaluate(potential, _state([atoms], mliap.model.cutoff, k_sel=1))
    np.testing.assert_allclose(out.total_energies.data[0], energy, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(-out.gradients.positions.data, forces, atol=2e-4)


def test_nve_matches_ase(frames, mliap, calculator):
    """A few velocity-Verlet steps through kUPS's MD propagator land where
    ASE's do, with the forces seeded on the state the way the first half-kick
    expects them (kUPS's own driver starts from zero and warms up)."""
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.md.verlet import VelocityVerlet
    from kups.application.md.data import MdParameters, MdState, md_state_from_ase
    from kups.application.md.simulation import make_md_propagator
    from kups.application.utils.propagate import make_cycle_function
    from kups.core.lens import bind, identity_lens
    from kups.core.propagator import propagate_and_fix
    from kups.core.utils.jax import key_chain

    @dataclass
    class MDState(MdState):
        upet_params: UPETParameters

    steps, dt = 5, 1.0
    atoms = frames[PERIODIC[0]].copy()
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(0))
    Stationary(atoms)

    reference = atoms.copy()
    reference.calc = calculator
    VelocityVerlet(reference, dt * units.fs).run(steps)

    config = MdParameters(
        temperature=300.0,
        time_step=dt,
        friction_coefficient=0.0,
        thermostat_time_constant=1e10,
        target_pressure=0.0,
        pressure_coupling_time=1e10,
        compressibility=1e-5,
        minimum_scale_factor=1.0,
        integrator="verlet",
    )
    particles, systems = md_state_from_ase(atoms, config)
    # kUPS's time unit is ASE's, so the momenta carry over as they are.
    particles = bind(particles, lambda t: t.data.momenta).set(
        jnp.asarray(atoms.get_momenta())
    )
    cutoffs = Table.arange(jnp.full(1, mliap.model.cutoff), label=SystemId)
    nl_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, cutoffs
    )
    state = MDState(particles, systems, nl_params, jnp.array([0]), UPETParameters(k_sel=1))
    state_lens = identity_lens(MDState)
    potential = make_upet_from_state(
        state_lens, parameters=mliap, gradient=POSITIONS_AND_CELL
    )
    out, state = _evaluate(potential, state)
    state = bind(state, lambda s: s.particles.data.position_gradients).set(
        out.gradients.positions.data
    )

    cycle = make_cycle_function(make_md_propagator(state_lens, "verlet", potential))
    chain = key_chain(jax.random.key(0))
    for _ in range(steps):
        state = propagate_and_fix(cycle, next(chain), state)

    cell = atoms.cell.array
    frac = (
        np.asarray(state.particles.data.positions) - reference.positions
    ) @ np.linalg.inv(cell)
    frac -= np.round(frac)  # kUPS wraps positions into the cell
    assert np.abs(frac @ cell).max() < 1e-4
    np.testing.assert_allclose(
        np.asarray(state.particles.data.momenta), reference.get_momenta(), atol=1e-4
    )


def test_overflow_grows_k_sel(frames, mliap, potential):
    """Starting from k_sel=1 the first call fails its assertion; the fix grows
    k_sel to cover the largest selected count and the rerun holds, with the
    same energy as a generously sized start."""
    atoms = frames[PERIODIC[0]]
    fn = as_result_function(potential)
    state = _state([atoms], mliap.model.cutoff, k_sel=1)
    first = fn(state)
    assert first.failed_assertions
    fixed = first.fix_or_raise(state)
    assert fixed.upet_params.k_sel > 1
    second = fn(fixed)
    assert not second.failed_assertions

    out, generous = _evaluate(potential, _state([atoms], mliap.model.cutoff, k_sel=64))
    assert generous.upet_params.k_sel == 64
    np.testing.assert_allclose(
        second.value.data.total_energies.data, out.total_energies.data, rtol=1e-6
    )
