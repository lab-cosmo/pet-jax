"""pack_edges: selection-free fixed-width packing of a flat NL."""

import numpy as np
import jax

import pytest
from ase.build import bulk, molecule

from petjax import pack_edges
from petjax.structure import to_structure

CUTOFF = 3.5


def _structure(atoms):
    return jax.tree.map(np.asarray, to_structure(atoms, cutoff=CUTOFF, skin=0.0))


def _pack(structure, k):
    from petjax.utils import edge_displacements

    R_ij = edge_displacements(
        structure["positions"],
        structure["centers"],
        structure["others"],
        structure["cell_shifts"],
        structure["cell"],
    )
    return pack_edges(
        R_ij,
        structure["centers"],
        structure["others"],
        structure["reverse"],
        structure["pair_mask"],
        structure["atomic_numbers"],
        structure["atom_mask"],
        k,
    )


def _max_count(structure):
    real = structure["pair_mask"]
    return int(np.bincount(structure["centers"][real]).max())


@pytest.fixture(params=["crystal", "molecule"])
def structure(request):
    if request.param == "crystal":
        atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64) * [2, 2, 2]
    else:
        atoms = molecule("H2O")
    return _structure(atoms)


def test_pack_edges_round_trip(structure):
    """Every unmasked pair survives packing exactly once; nothing else does."""
    k = _max_count(structure)
    packed, truncation = _pack(structure, k)
    assert not bool(truncation.overflow)
    assert packed["pair_cutoffs"] is None

    real = np.asarray(packed["pair_mask"])
    got = {
        (int(c), int(o))
        for c, o in zip(
            np.asarray(packed["centers"])[real], np.asarray(packed["neighbors"])[real]
        )
    }
    mask = structure["pair_mask"]
    want = {
        (int(c), int(o))
        for c, o in zip(structure["centers"][mask], structure["others"][mask])
    }
    assert got == want
    assert int(real.sum()) == int(mask.sum())


def test_pack_edges_reverse_involution(structure):
    k = _max_count(structure)
    packed, _ = _pack(structure, k)
    rev = np.asarray(packed["reverse"])
    real = np.flatnonzero(np.asarray(packed["pair_mask"]))
    assert np.array_equal(rev[rev[real]], real)


def test_pack_edges_overflow_on_undersized_width(structure):
    k = _max_count(structure)
    if k < 2:
        pytest.skip("needs at least 2 neighbors on some center")
    _, truncation = _pack(structure, k - 1)
    assert bool(truncation.overflow)


def test_pack_edges_truncation(structure):
    """The accounting next to the packed dict: no cutoffs, every unmasked pair
    selected, per-center counts, and overflow iff a count exceeds the width."""
    _, t = _pack(structure, _max_count(structure))
    assert t.atomic_cutoffs is None and t.pair_cutoffs is None
    assert np.array_equal(t.selected, structure["pair_mask"])
    N = structure["positions"].shape[0]
    counts = np.bincount(structure["centers"][structure["pair_mask"]], minlength=N)
    assert np.array_equal(t.counts, counts)
    assert not bool(t.overflow)


def test_truncate_edges_matches_select_edges(structure):
    """The in-JIT selection reports the same accounting as the host-side
    ``select_edges`` on the same structure, and overflow iff ``k_sel`` is below
    the largest count."""
    from petjax import select_edges, truncate_edges
    from petjax.utils import edge_displacements

    hypers = dict(num_neighbors_adaptive=4, cutoff=CUTOFF, cutoff_width_adaptive=0.5)
    sel = select_edges(structure, **hypers, method="grid")
    k_sel = int(sel.counts.max())
    R_ij = edge_displacements(
        structure["positions"],
        structure["centers"],
        structure["others"],
        structure["cell_shifts"],
        structure["cell"],
    )

    def run(k):
        return truncate_edges(
            R_ij,
            structure["centers"],
            structure["others"],
            structure["reverse"],
            structure["pair_mask"],
            structure["atomic_numbers"],
            structure["atom_mask"],
            k,
            **hypers,
            method="grid",
        )

    _, t = run(k_sel)
    assert np.allclose(t.atomic_cutoffs, sel.atomic_cutoffs, rtol=1e-6, atol=0)
    assert np.allclose(t.pair_cutoffs, sel.pair_cutoffs, rtol=1e-6, atol=0)
    assert np.array_equal(t.selected, sel.selected)
    assert np.array_equal(t.counts, sel.counts)
    assert not bool(t.overflow)

    if k_sel > 1:
        _, t = run(k_sel - 1)
        assert bool(t.overflow)
        assert np.array_equal(t.counts, sel.counts)


def test_pack_edges_forward_smoke(structure):
    """UPET consumes the packed layout with pair_cutoffs=None (scalar-cutoff bump)."""
    import jax.numpy as jnp

    from petjax import UPET

    model = UPET(
        d_pet=8,
        d_node=8,
        d_head=8,
        d_feedforward=8,
        num_heads=2,
        num_attention_layers=1,
        num_gnn_layers=1,
        cutoff=CUTOFF,
    )
    packed, truncation = _pack(structure, _max_count(structure))
    assert not bool(truncation.overflow)
    params = model.init(jax.random.key(0), **packed)
    energy = model.apply(params, **packed)
    assert bool(jnp.all(jnp.isfinite(energy)))


# -- select_edges: the adaptive selection's mask, host-side --


def test_select_edges_mask(structure):
    """The host-side selection is internally consistent (pair cutoffs are the
    endpoint means of the atomic ones, the mask is ``r_ij <= pair_cutoff`` on
    unmasked pairs, symmetric under ``reverse``) and matches the eager selection
    core on the same displacements."""
    from petjax import select_edges
    from petjax.select import _select_edges
    from petjax.utils import edge_displacements, safe_norm

    hypers = dict(num_neighbors_adaptive=4, cutoff=CUTOFF, cutoff_width_adaptive=0.5)
    sel = select_edges(structure, **hypers, method="grid")
    centers, others, pair_mask = (structure[k] for k in ("centers", "others", "pair_mask"))
    N = structure["positions"].shape[0]

    assert all(isinstance(x, np.ndarray) for x in sel)
    assert sel.atomic_cutoffs.shape == (N,)
    assert sel.pair_cutoffs.shape == centers.shape
    assert sel.selected.dtype == bool and sel.selected.shape == centers.shape
    assert not sel.selected[~pair_mask].any()
    assert np.array_equal(sel.selected[structure["reverse"]], sel.selected)
    assert np.array_equal(sel.counts, np.bincount(centers[sel.selected], minlength=N))

    R_ij = edge_displacements(
        structure["positions"], centers, others, structure["cell_shifts"], structure["cell"]
    )
    r_ij = np.asarray(safe_norm(R_ij, axis=-1))
    pair_cutoffs = (sel.atomic_cutoffs[centers] + sel.atomic_cutoffs[others]) / 2
    assert np.array_equal(sel.pair_cutoffs, pair_cutoffs)
    assert np.array_equal(sel.selected, (r_ij <= sel.pair_cutoffs) & pair_mask)

    # The jitted kernel and the eager core agree to float32 rounding on the
    # cutoffs (XLA fuses the reductions differently) and exactly on the mask.
    eager = _select_edges(R_ij, centers, others, pair_mask, N, **hypers, method="grid")
    assert np.allclose(sel.atomic_cutoffs, eager[0], rtol=1e-6, atol=0)
    assert np.allclose(sel.pair_cutoffs, eager[1], rtol=1e-6, atol=0)
    assert np.array_equal(sel.selected, np.asarray(eager[2]))


def test_masked_slots_carry_no_features(structure):
    """Padded pair slots leave the backbone at exactly zero. The reverse gather
    is the one consumer of a masked slot, so anything left there leaks into an
    edge whose reciprocal was trimmed upstream."""
    from petjax.model import Backbone

    backbone = Backbone(
        d_pet=8,
        d_node=8,
        d_feedforward=8,
        num_heads=2,
        num_attention_layers=1,
        num_gnn_layers=2,
        cutoff=CUTOFF,
    )
    packed, _ = _pack(structure, _max_count(structure) + 5)
    inputs = {k: v for k, v in packed.items() if k != "pair_cutoffs"}
    params = backbone.init(jax.random.key(0), **inputs, pair_cutoffs=None)
    _, messages, _ = backbone.apply(params, **inputs, pair_cutoffs=None)
    assert np.all(np.asarray(messages)[~np.asarray(packed["pair_mask"])] == 0.0)


# -- solver variants (Express.solver) against the stock solver --

# (num_neighbors_adaptive, cutoff, cutoff_width_adaptive) of the two solver
# parameter sets in use: PET-MAD v1.6 and the xs grid checkpoints' values.
SOLVER_HYPERS = {"k16_w1": (16, 7.5, 1.0), "k8_w05": (8, 7.5, 0.5)}

SOLVER_CASES = {
    "sorted_segment_sum": dict(solver="sorted_segment_sum"),
}


def _liquid(n=64, rho=0.09, dmin=1.8, seed=0):
    """Random positions at a dense liquid's density with a minimum distance: no
    shells, so roots land anywhere between neighbours."""
    from ase import Atoms

    rng = np.random.default_rng(seed)
    L = (n / rho) ** (1 / 3)
    positions = []
    while len(positions) < n:
        p = rng.uniform(0, L, 3)
        if all(np.linalg.norm((p - q + L / 2) % L - L / 2) >= dmin for q in positions):
            positions.append(p)
    numbers = rng.choice([40, 8, 8], size=n)
    return Atoms(numbers=numbers, positions=positions, cell=np.eye(3) * L, pbc=True)


def _isolated_and_dimer():
    from ase import Atoms

    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74], [20.0, 20.0, 20.0]]
    return Atoms("H2O", positions=positions)


@pytest.fixture(scope="module")
def solver_structures(mini_xyz):
    from ase.io import read

    atoms = [read(str(mini_xyz), index=i) for i in (0, 5, 6, 11)]
    atoms += [_liquid(), _isolated_and_dimer()]
    return [to_structure(a, cutoff=7.5, skin=0.0) for a in atoms]


def _solve(structure, hypers, dtype, express=None):
    """Cutoffs, selection and d(sum_i c_i r_i)/d positions of ``_select_edges``."""
    import jax.numpy as jnp

    from petjax.select import _select_edges
    from petjax.utils import edge_displacements

    k, cutoff, width = hypers
    N = structure["positions"].shape[0]
    weights = jnp.asarray(np.random.default_rng(1).uniform(0.5, 1.5, N), dtype)

    def cutoffs(positions):
        R_ij = edge_displacements(
            positions,
            structure["centers"],
            structure["others"],
            jnp.asarray(structure["cell_shifts"], dtype),
            jnp.asarray(structure["cell"], dtype),
        )
        out = _select_edges(
            R_ij,
            structure["centers"],
            structure["others"],
            structure["pair_mask"],
            N,
            k,
            cutoff,
            width,
            method="solver",
            express=express,
        )
        return jnp.sum(weights * out[0]), out

    positions = jnp.asarray(structure["positions"], dtype)
    grad, out = jax.grad(cutoffs, has_aux=True)(positions)
    return np.asarray(out[0]), np.asarray(out[2]), np.asarray(grad)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("hypers", list(SOLVER_HYPERS), ids=list(SOLVER_HYPERS))
@pytest.mark.parametrize("case", list(SOLVER_CASES))
def test_solver_variant_matches_stock(case, hypers, dtype, solver_structures):
    """Same root (1 ulp in float32, a few in float64), same selected pairs, and
    the same IFT gradient as the stock solver, on the mini frames, a dense
    liquid, and an isolated atom next to a dimer."""
    from petjax.express import Express

    express = Express(**SOLVER_CASES[case])
    with jax.enable_x64(dtype == "float64"):
        for structure in solver_structures:
            c0, s0, g0 = _solve(structure, SOLVER_HYPERS[hypers], dtype)
            c1, s1, g1 = _solve(structure, SOLVER_HYPERS[hypers], dtype, express)
            ulps = 1 if dtype == "float32" else 4
            np.testing.assert_array_less(np.abs(c1 - c0), ulps * np.spacing(c0) * 1.001)
            assert np.array_equal(s1, s0)
            scale = np.abs(g0).max()
            rtol = 1e-5 if dtype == "float32" else 1e-11
            np.testing.assert_allclose(g1, g0, rtol=0, atol=rtol * scale)
