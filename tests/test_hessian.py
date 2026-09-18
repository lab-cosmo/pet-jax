"""Tests for the positions-Hessian: `petjax.hessian` factories on the
calculator's own inputs, and `UPETCalculator.hessian` end to end.

Cubic Si8 is fully connected at one hop, so the sparsity pattern is complete
at any `hops >= 1` and sparse-equals-dense here checks the plumbing (pattern,
coloring, decompression, aux, padding), not the hop count. The hop count is
pinned in `test_hop_count.py` on a graph deep enough to show it.
"""

import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.build import bulk

from petjax import UPETCalculator, select_edges
from petjax.hessian import (
    get_dense_hessian_fn,
    get_energy_fn,
    get_sparse_hessian_fn,
    selected_adjacency,
)

asdex = pytest.importorskip("asdex")

# Structural zeros and the sparse-vs-dense agreement below rely on fp64.
jax.config.update("jax_enable_x64", True)


def _atoms():
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)  # 8 atoms, periodic
    atoms.rattle(0.1, seed=1)  # break symmetry → generic couplings
    return atoms


@pytest.fixture(scope="module")
def calc(pet_mad_xs_checkpoint):
    """fp64, no shadow: the sparse path's calculator."""
    return UPETCalculator.from_checkpoint(
        pet_mad_xs_checkpoint, default_dtype="float64", no_shadow=True, stress=False
    )


@pytest.fixture(scope="module")
def shadow_calc(pet_mad_xs_checkpoint):
    """fp64 with shadow forces: the dense reference whose forces are an exact
    gradient of the energy."""
    return UPETCalculator.from_checkpoint(
        pet_mad_xs_checkpoint, default_dtype="float64", stress=False
    )


@pytest.fixture(scope="module")
def inputs(calc):
    """`(params, positions, structure)` for Si8, built by the calculator."""
    atoms = _atoms()
    calc._build_structure(atoms)
    # Device arrays throughout: `asdex.check_hessian_correctness` traces the
    # energy with the structure closed over, and a host int array indexed by a
    # tracer would not survive that.
    structure = jax.tree.map(jnp.asarray, calc._structure)
    return calc._params, structure["positions"], structure


def _energy_fn(calc):
    return get_energy_fn(calc._model, no_shadow=True)


def _exact_hops(calc):
    return 2 * calc._model.num_gnn_layers + 1


def _coloring(calc, structure, hops):
    from petjax.sparse import hessian_coloring

    selected = select_edges(
        structure,
        calc._model.num_neighbors_adaptive,
        calc._model.cutoff,
        calc._model.cutoff_width_adaptive,
        method=calc._model.adaptive_cutoff_method,
    )
    n_real = int(structure["atom_mask"].sum())
    centers, others = selected_adjacency(structure, selected, n_real)
    return hessian_coloring(centers, others, structure["positions"].shape[0], hops)


def _reference(calc, inputs):
    """Dense `jax.hessian` of the same energy, `(N, 3, N, 3)` numpy."""
    params, positions, structure = inputs
    energy_fn = _energy_fn(calc)
    return np.asarray(jax.hessian(lambda p: energy_fn(params, p, structure)[0])(positions))


# -- factories --


def test_sparse_matches_reference(calc, inputs):
    """Sparse at exact hops equals the dense JAX Hessian on the real DOFs, with
    the overflow flag surfaced as aux and asserted false."""
    params, positions, structure = inputs
    coloring = _coloring(calc, structure, _exact_hops(calc))
    hessian = jax.jit(get_sparse_hessian_fn(_energy_fn(calc), coloring))

    H, overflow = hessian(params, positions, structure)
    assert not bool(overflow)
    H = np.asarray(H.todense())
    H_ref = _reference(calc, inputs)
    n = int(structure["atom_mask"].sum())

    assert H.shape == H_ref.shape == (positions.shape[0], 3, positions.shape[0], 3)
    np.testing.assert_allclose(H[:n, :, :n], H_ref[:n, :, :n], atol=1e-8)
    assert np.abs(H_ref).max() > 1e-3  # sanity: not trivially zero


def test_dense_matches_reference(calc, inputs):
    params, positions, structure = inputs
    hessian = jax.jit(get_dense_hessian_fn(_energy_fn(calc)))
    H, overflow = hessian(params, positions, structure)
    assert not bool(overflow)
    np.testing.assert_allclose(np.asarray(H), _reference(calc, inputs), atol=1e-8)


@pytest.mark.parametrize("kind", ["sparse", "dense"])
def test_chunk_size_and_remat_are_noops(calc, inputs, kind):
    """chunk_size bounds memory and remat trades compute for it; neither may
    change the result."""
    params, positions, structure = inputs
    energy_fn = _energy_fn(calc)
    if kind == "sparse":
        coloring = _coloring(calc, structure, _exact_hops(calc))

        def build(**kw):
            return get_sparse_hessian_fn(energy_fn, coloring, **kw)
    else:

        def build(**kw):
            return get_dense_hessian_fn(energy_fn, **kw)

    def run(**kw):
        H, _ = jax.jit(build(**kw))(params, positions, structure)
        return np.asarray(H.todense() if hasattr(H, "todense") else H)

    H_plain = run()
    np.testing.assert_allclose(run(chunk_size=1), H_plain, atol=1e-10)
    np.testing.assert_allclose(run(remat=True), H_plain, atol=1e-10)


def test_lifted_coloring_matches_asdex(calc, inputs):
    from petjax.sparse import sparsity_patterns

    _, _, structure = inputs
    hops = _exact_hops(calc)
    ours = _coloring(calc, structure, hops)
    selected = select_edges(
        structure,
        calc._model.num_neighbors_adaptive,
        calc._model.cutoff,
        calc._model.cutoff_width_adaptive,
        method=calc._model.adaptive_cutoff_method,
    )
    n_real = int(structure["atom_mask"].sum())
    centers, others = selected_adjacency(structure, selected, n_real)
    _, coord = sparsity_patterns(centers, others, structure["positions"].shape[0], hops)
    ref = asdex.hessian_coloring_from_sparsity(coord, mode="fwd_over_rev")
    assert ours.num_colors == ref.num_colors
    assert np.array_equal(np.asarray(ours.colors), np.asarray(ref.colors))


def test_check_hessian_correctness(calc, inputs):
    """asdex's own verifier on the pattern + coloring, independent of our
    wrapper. `method="dense"` because the `matvec` path assumes a 2-D Hessian
    and chokes on the native `(N, 3, N, 3)` output."""
    params, positions, structure = inputs
    energy_fn = _energy_fn(calc)
    coloring = _coloring(calc, structure, _exact_hops(calc))
    asdex.check_hessian_correctness(
        lambda p: energy_fn(params, p, structure)[0], positions, coloring, method="dense"
    )


# -- UPETCalculator.hessian --


def test_exact_equals_dense_no_shadow(calc):
    atoms = _atoms()
    n = len(atoms)
    H_sparse = calc.hessian(atoms, hops="exact")
    H_dense = calc.hessian(atoms, hops=None, no_shadow=True)

    assert H_sparse.shape == H_dense.shape == (n, 3, n, 3)
    assert H_sparse.dtype == np.float64
    assert calc._N_padded > n  # the dummy slot at least: padding was stripped
    np.testing.assert_allclose(H_sparse, H_dense, atol=1e-10)
    np.testing.assert_allclose(H_sparse, H_sparse.transpose(2, 3, 0, 1), atol=1e-10)


def test_truncation_drops_and_folds(calc):
    """What a truncated pattern does: entries outside it are structural zeros,
    and the couplings it drops fold into the retained entries, because the
    compression assumes them zero. Si8 is complete at one hop, so only
    `hops=0` (on-site blocks) truncates here; then every atom has the same
    three colors and the decompressed on-site block is the row sum of the exact
    Hessian over all atoms."""
    atoms = _atoms()
    H_exact = calc.hessian(atoms, hops="exact")
    # (n, n, 3, 3): one 3x3 block per atom pair, so a pair mask indexes blocks.
    blocks_onsite = calc.hessian(atoms, hops=0).transpose(0, 2, 1, 3)
    diag = np.eye(len(atoms), dtype=bool)
    assert np.all(blocks_onsite[~diag] == 0.0)
    np.testing.assert_allclose(blocks_onsite[diag], H_exact.sum(axis=2), atol=1e-10)
    assert np.abs(H_exact.transpose(0, 2, 1, 3)[~diag]).max() > 1e-3


def test_dense_shadow_matches_finite_differences(shadow_calc):
    """Central differences of the shadow calculator's forces (an exact energy
    gradient) against the dense shadow Hessian: the one check that does not
    go through JAX second-order AD at all."""
    atoms = _atoms()
    H = shadow_calc.hessian(atoms)
    h = 1e-4
    fd = np.zeros_like(H)
    for a in range(len(atoms)):
        for k in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.calc = minus.calc = shadow_calc
            plus.positions[a, k] += h
            minus.positions[a, k] -= h
            fd[a, k] = -(plus.get_forces() - minus.get_forces()) / (2 * h)
    np.testing.assert_allclose(fd, H, atol=1e-4)


def test_shadow_coupling_is_not_negligible(calc, shadow_calc):
    """Dense with and without shadow coupling differ by a visible amount, so
    the `no_shadow` guard on the sparse path protects something real."""
    atoms = _atoms()
    H_shadow = shadow_calc.hessian(atoms)
    H_no_shadow = calc.hessian(atoms)  # calc was built with no_shadow=True
    assert np.abs(H_shadow - H_no_shadow).max() > 1e-3


def test_no_shadow_resolves_to_calculator_setting(calc, shadow_calc):
    atoms = _atoms()
    np.testing.assert_allclose(
        shadow_calc.hessian(atoms), shadow_calc.hessian(atoms, no_shadow=False), atol=1e-12
    )
    np.testing.assert_allclose(
        calc.hessian(atoms), calc.hessian(atoms, no_shadow=True), atol=1e-12
    )


def test_guards(calc, shadow_calc, pet_mad_xs_checkpoint):
    atoms = _atoms()
    with pytest.raises(ValueError, match="no_shadow"):
        shadow_calc.hessian(atoms, hops="exact", no_shadow=False)
    for bad in (-1, 1.5, "five", True):
        with pytest.raises(ValueError, match="hops"):
            calc.hessian(atoms, hops=bad)
    fp32 = UPETCalculator.from_checkpoint(pet_mad_xs_checkpoint, stress=False)
    with pytest.raises(ValueError, match="float64"):
        fp32.hessian(atoms)


def test_selection_recomputed_and_coloring_cached(calc, monkeypatch):
    """The selection runs every call (positions move under the Verlet skin
    without a rebuild); the coloring and the jitted fn are reused while the
    selected pair set is unchanged."""
    import petjax.calculator as calculator_module

    calls = []
    real_select_edges = calculator_module.select_edges

    def counting(*args, **kwargs):
        calls.append(1)
        return real_select_edges(*args, **kwargs)

    monkeypatch.setattr(calculator_module, "select_edges", counting)

    atoms = _atoms()
    calc.hessian(atoms, hops="exact")
    coloring, fn = calc._coloring_cache[1], calc._hessian_cache[1]
    calc.hessian(atoms, hops="exact")
    assert len(calls) == 2
    assert calc._coloring_cache[1] is coloring
    assert calc._hessian_cache[1] is fn

    calc.hessian(atoms, hops=1)  # new configuration: recolored, retraced
    assert calc._coloring_cache[1] is not coloring
    assert calc._hessian_cache[1] is not fn


def test_selection_change_under_skin_stays_correct(calc):
    """The case the per-call selection exists for: a displacement below the
    Verlet skin changes the selected pair set without a raw-NL rebuild. The
    coloring must be rebuilt, not served stale, and the sparse Hessian must
    still equal the dense one.

    Needs a cell whose selection can change at all: on Si8 every atom pair is
    selected already. Si16 at the trained budget flips pairs within 0.2 Å. Its
    exact pattern is still complete, so the equality would hold for any
    coloring; what this pins is that the change is seen with no rebuild and
    answered with a fresh coloring rather than a stale one (a stale one built
    for another pair set would be caught by the lift's guards or by a shape
    mismatch, not by silently wrong numbers)."""
    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 1, 1)
    atoms.rattle(0.05, seed=1)
    calc.hessian(atoms, hops="exact")
    key, stats = calc._coloring_cache[0], calc.debug_stats
    # Push every atom along a fixed random direction, 0.02 Å per step, up to
    # 0.2 Å: below the 0.5 Å skin's rebuild threshold (2 * max displacement).
    step = np.random.default_rng(3).normal(size=atoms.positions.shape)
    step *= 0.02 / np.linalg.norm(step, axis=1).max()
    for _ in range(10):
        atoms.positions += step
        calc.hessian(atoms, hops="exact")
        if calc._coloring_cache[0] != key:
            break
    else:
        pytest.fail("no selection change under the skin within 0.2 Å")
    assert calc.debug_stats is stats  # same rebuild: no raw-NL update
    H_sparse = calc.hessian(atoms, hops="exact")
    H_dense = calc.hessian(atoms, hops=None, no_shadow=True)
    np.testing.assert_allclose(H_sparse, H_dense, atol=1e-10)


@pytest.mark.parametrize("hops", ["exact", None])
def test_overflow_retry(calc, hops):
    """An undersized `k_sel` makes the energy report overflow; `hessian` must
    rebuild with `force_recompute_k_sel=True` and recover, like `calculate`."""
    atoms = _atoms()
    H_ref = calc.hessian(atoms, hops=hops, no_shadow=True)
    calc._build_structure(atoms)
    calc._structure = {**calc._structure, "k_sel_sizer": jnp.zeros(2, dtype=bool)}
    calc._k_sel = 2
    H = calc.hessian(atoms, hops=hops, no_shadow=True)
    assert calc.debug_stats["overflow_retry"]
    assert calc._k_sel > 2
    np.testing.assert_allclose(H, H_ref, atol=1e-10)
