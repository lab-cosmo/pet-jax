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


def test_pack_edges_pair_cutoffs(structure):
    """Per-pair cutoffs travel with their pairs, and a constant one equal to
    the model cutoff reproduces the scalar-cutoff forward."""
    from petjax import UPET
    from petjax.utils import edge_displacements

    k = _max_count(structure)
    R_ij = edge_displacements(
        structure["positions"],
        structure["centers"],
        structure["others"],
        structure["cell_shifts"],
        structure["cell"],
    )
    args = (
        R_ij,
        structure["centers"],
        structure["others"],
        structure["reverse"],
        structure["pair_mask"],
        structure["atomic_numbers"],
        structure["atom_mask"],
        k,
    )
    tag = np.arange(len(structure["centers"]), dtype=np.float32)
    packed, truncation = pack_edges(*args, pair_cutoffs=tag)
    np.testing.assert_array_equal(np.asarray(truncation.pair_cutoffs), tag)
    real = np.asarray(packed["pair_mask"])
    by_tag = np.asarray(packed["pair_cutoffs"])[real].astype(int)
    np.testing.assert_array_equal(
        np.asarray(packed["centers"])[real], structure["centers"][by_tag]
    )
    np.testing.assert_array_equal(
        np.asarray(packed["neighbors"])[real], structure["others"][by_tag]
    )

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
    scalar, _ = pack_edges(*args)
    constant, _ = pack_edges(*args, pair_cutoffs=np.full_like(tag, CUTOFF))
    params = model.init(jax.random.key(0), **scalar)
    np.testing.assert_allclose(
        model.apply(params, **constant), model.apply(params, **scalar), rtol=1e-6, atol=1e-7
    )


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
    "blocked_prefix_sum": dict(solver="blocked_prefix_sum"),
    "sorted_window": dict(solver="sorted_window"),
    "sorted_window_6": dict(solver="sorted_window", solver_iterations=6),
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


def _solve(structure, hypers, dtype, express=None, seed=None):
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
            seed=seed,
        )
        return jnp.sum(weights * out[0]), out

    positions = jnp.asarray(structure["positions"], dtype)
    grad, out = jax.grad(cutoffs, has_aux=True)(positions)
    return np.asarray(out[0]), np.asarray(out[2]), np.asarray(grad)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("hypers", list(SOLVER_HYPERS), ids=list(SOLVER_HYPERS))
@pytest.mark.parametrize("case", list(SOLVER_CASES))
def test_solver_variant_matches_stock(case, hypers, dtype, solver_structures):
    """Same root, same selected pairs and the same IFT gradient as the stock
    solver, on the mini frames, a dense liquid, and an isolated atom next to a
    dimer. The root differs only by summation order: F is summed in another
    order, and its rounding divided by dF/dr lands on the cutoff (a few ulps
    where the count is flat, as in a molecule)."""
    from petjax.express import Express

    if (case, hypers, dtype) == ("sorted_window", "k8_w05", "float64"):
        # The value still matches (the IFT step is one more Newton step), but
        # the gradient's dF/dr is taken at the unconverged loop iterate.
        pytest.skip("4 window steps converge to float32 only; w=0.5 needs 6 in float64")
    express = Express(**SOLVER_CASES[case])
    with jax.enable_x64(dtype == "float64"):
        for structure in solver_structures:
            c0, s0, g0 = _solve(structure, SOLVER_HYPERS[hypers], dtype)
            c1, s1, g1 = _solve(structure, SOLVER_HYPERS[hypers], dtype, express)
            atol, gtol = (4e-6, 1e-4) if dtype == "float32" else (1e-14, 1e-12)
            np.testing.assert_allclose(c1, c0, rtol=0, atol=atol)
            assert np.array_equal(s1, s0)
            np.testing.assert_allclose(g1, g0, rtol=0, atol=gtol * np.abs(g0).max())


def _truncate(structure, k_sel, express, hypers=SOLVER_HYPERS["k16_w1"]):
    from petjax import truncate_edges
    from petjax.utils import edge_displacements

    R_ij = edge_displacements(
        structure["positions"],
        structure["centers"],
        structure["others"],
        structure["cell_shifts"],
        structure["cell"],
    )
    k, cutoff, width = hypers
    return truncate_edges(
        R_ij,
        structure["centers"],
        structure["others"],
        structure["reverse"],
        structure["pair_mask"],
        structure["atomic_numbers"],
        structure["atom_mask"],
        k_sel,
        k,
        cutoff,
        width,
        method="solver",
        express=express,
    )


@pytest.mark.parametrize("hypers", list(SOLVER_HYPERS), ids=list(SOLVER_HYPERS))
def test_sorted_window_pack_matches_stock(hypers, solver_structures):
    """The candidate pack selects the same pairs per center as the stock pack
    (in distance order instead of list order), with the same counts, and its
    reverse map points every packed pair at its packed reverse."""
    from petjax.express import Express

    window = Express(solver="sorted_window")
    for structure in solver_structures:
        structure = jax.tree.map(np.asarray, structure)
        _, t0 = _truncate(structure, 64, None, SOLVER_HYPERS[hypers])
        k_sel = max(int(np.asarray(t0.counts).max()), 1)
        p0, t0 = _truncate(structure, k_sel, None, SOLVER_HYPERS[hypers])
        p1, t1 = _truncate(structure, k_sel, window, SOLVER_HYPERS[hypers])
        assert not bool(t0.overflow) and not bool(t1.overflow)
        assert np.array_equal(t1.selected, t0.selected)
        assert np.array_equal(t1.counts, t0.counts)

        # Same pair in each row, as a set: key a packed slot by its raw pair.
        def rows(packed):
            real = np.asarray(packed["pair_mask"])
            centers = np.asarray(packed["centers"])[real]
            key = np.stack(
                [np.asarray(packed["neighbors"])[real], *np.asarray(packed["R_ij"])[real].T]
            )
            return {(int(c), tuple(np.round(k, 6))) for c, k in zip(centers, key.T)}

        assert rows(p1) == rows(p0)
        assert np.array_equal(np.asarray(p1["pair_mask"]), np.asarray(p0["pair_mask"]))
        mask = np.asarray(p0["pair_mask"])
        np.testing.assert_allclose(
            np.sort(np.asarray(p1["pair_cutoffs"])[mask]),
            np.sort(np.asarray(p0["pair_cutoffs"])[mask]),
            rtol=1e-6,
        )

        # The reverse slot holds the reversed pair: same atoms, opposite R_ij.
        real = np.flatnonzero(np.asarray(p1["pair_mask"]))
        rev = np.asarray(p1["reverse"])[real]
        assert np.array_equal(np.asarray(p1["reverse"])[rev], real)
        assert np.array_equal(
            np.asarray(p1["centers"])[rev], np.asarray(p1["neighbors"])[real]
        )
        np.testing.assert_allclose(
            np.asarray(p1["R_ij"])[rev], -np.asarray(p1["R_ij"])[real], atol=1e-5
        )


def test_sorted_window_overflow(solver_structures):
    """Each of the three ways the window path can miss a pair raises overflow:
    a row selecting more than k_sel, a candidate prefix too short to rule out a
    selectable pair, and a solver window too small for the bracket."""
    from petjax.express import Express

    liquid = jax.tree.map(np.asarray, solver_structures[4])
    _, t = _truncate(liquid, 64, None)
    k_sel = int(np.asarray(t.counts).max())
    window = Express(solver="sorted_window")
    assert not bool(_truncate(liquid, k_sel, window)[1].overflow)
    assert bool(_truncate(liquid, k_sel - 1, window)[1].overflow)
    assert bool(
        _truncate(liquid, k_sel, Express(solver="sorted_window", candidate_extra=0))[
            1
        ].overflow
    )
    assert bool(
        _truncate(liquid, k_sel, Express(solver="sorted_window", window_capacity=4))[
            1
        ].overflow
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("hypers", list(SOLVER_HYPERS), ids=list(SOLVER_HYPERS))
@pytest.mark.parametrize(
    "solver", ["segment_sum", "sorted_segment_sum", "blocked_prefix_sum"]
)
def test_seeded_solve(solver, hypers, dtype, solver_structures):
    """Seeded with a frame's cutoffs, the solve on that frame after a small
    random step converges in 3 Newton steps to the unseeded root, with the
    same selection and gradient (5 steps in float64 at w=0.5); a seed with
    ``inf`` displacement is ignored.
    The reference takes 40 steps: in float64 at w=0.5 the stock 10 can stop
    short (a shoulder between shells makes Newton bisect). The blocked sum's
    root is defined only to its own rounding, so two iteration paths may land
    up to twice that apart."""
    from petjax.express import Express

    unseeded = Express(solver=solver, solver_iterations=40)
    rng = np.random.default_rng(3)
    if solver == "blocked_prefix_sum":
        atol = 8e-6 if dtype == "float32" else 4e-14
    with jax.enable_x64(dtype == "float64"):
        for structure in solver_structures:
            seed, _, _ = _solve(structure, SOLVER_HYPERS[hypers], dtype, unseeded)
            step = rng.normal(scale=0.01, size=structure["positions"].shape)
            step *= structure["atom_mask"][:, None]
            displaced = {**structure, "positions": structure["positions"] + step}
            delta = np.linalg.norm(step, axis=1).max()

            def solve(iterations, delta=delta):
                express = Express(
                    solver=solver, seed_cutoffs=True, solver_iterations=iterations
                )
                return _solve(
                    displaced, SOLVER_HYPERS[hypers], dtype, express, seed=(seed, delta)
                )

            c0, s0, g0 = _solve(displaced, SOLVER_HYPERS[hypers], dtype, unseeded)
            if solver != "blocked_prefix_sum":
                atol = 2 * np.spacing(c0)
            c1, s1, g1 = solve(3)
            np.testing.assert_array_less(np.abs(c1 - c0), atol * 1.001)
            assert np.array_equal(s1, s0)
            if (dtype, hypers) == ("float64", "k8_w05"):
                # The IFT step makes the value exact, but takes its slope at
                # the iterate: the float64 gradient at w=0.5 wants 5 steps.
                _, _, g1 = solve(5)
            gtol = 1e-4 if dtype == "float32" else 1e-12
            np.testing.assert_allclose(g1, g0, rtol=0, atol=gtol * np.abs(g0).max())

            # Ignored: the stock 10 steps from [0, cutoff], whatever was asked.
            c2, _, _ = solve(3, delta=np.inf)
            c3, _, _ = _solve(
                displaced, SOLVER_HYPERS[hypers], dtype, Express(solver=solver)
            )
            np.testing.assert_array_equal(c2, c3)
