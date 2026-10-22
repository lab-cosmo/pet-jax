"""The express switches against the stock path, in fp64.

Every entry of ``CASES`` is run through the calculator on the same frames as
the stock path and must reproduce its energies, forces and stress to summation
order. A switch that cannot pass this test is not exact and does not belong in
``Express``. Streams add their settings to ``CASES``; combinations go in too.
"""

import numpy as np

import pytest
from ase.io import read

from petjax import UPETCalculator
from petjax.express import Express

# Periodic 9 atoms, two molecules (3 and 24 atoms), periodic 108 atoms.
FRAMES = (0, 5, 6, 11)
CHECKPOINTS = ("pet-mad-xs", "pet-mad-xs-v1.6")  # grid and solver cutoffs

CASES = {
    "stock": Express(),
    "flat": Express(tokens="flat"),
    # k_sel fits one tile: no extra tiles in the graph
    "flat-tile32": Express(tokens="flat", attention_tile=32),
    # narrow enough that most atoms need an extra tile (all 108 of the large
    # frame on the 16-neighbour checkpoint), while the widest atom (21) fits two
    "flat-tile12": Express(tokens="flat", attention_tile=12),
}

RTOL_ENERGY = 1e-11
ATOL_FORCES = 1e-10
ATOL_STRESS = 1e-10


def predictions(checkpoint, frames, express, **kwargs):
    calc = UPETCalculator.from_checkpoint(
        str(checkpoint), default_dtype="float64", stress=True, express=express, **kwargs
    )
    out = []
    for atoms in frames:
        atoms = atoms.copy()
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress() if atoms.pbc.any() else None
        out.append((energy, forces, stress, dict(calc.call_stats)))
    return out


def assert_equivalent(reference, candidate):
    for (e0, f0, s0, c0), (e1, f1, s1, c1) in zip(reference, candidate):
        assert abs(e1 - e0) <= RTOL_ENERGY * abs(e0), f"energy {e0} vs {e1}"
        np.testing.assert_allclose(f1, f0, rtol=0, atol=ATOL_FORCES)
        if s0 is not None:
            np.testing.assert_allclose(s1, s0, rtol=0, atol=ATOL_STRESS)
        assert c1["k_max"] == c0["k_max"]
        assert abs(c1["max_cutoff"] - c0["max_cutoff"]) < 1e-9


@pytest.fixture(scope="module")
def frames(mini_xyz):
    return [read(str(mini_xyz), index=i) for i in FRAMES]


@pytest.fixture(scope="module", params=CHECKPOINTS)
def checkpoint(request, assets_dir):
    path = assets_dir / "checkpoints" / request.param
    if not (path / "model.msgpack").exists():
        pytest.skip(f"{request.param} checkpoint missing")
    return path


@pytest.fixture(scope="module")
def stock(checkpoint, frames):
    return predictions(checkpoint, frames, Express())


@pytest.mark.parametrize("name", [n for n in CASES if n != "stock"])
def test_express_matches_stock(name, checkpoint, frames, stock):
    assert_equivalent(stock, predictions(checkpoint, frames, CASES[name]))


def test_call_stats(checkpoint, frames, stock):
    """The per-call reach: the largest cutoff lies in the solver's range and
    the largest count is what the host-side selection would size from."""
    from petjax.convert import load_checkpoint
    from petjax.select import select_edges
    from petjax.structure import to_structure

    params, metadata = load_checkpoint(checkpoint)
    config = metadata["config"]
    for atoms, (_, _, _, stats) in zip(frames, stock):
        assert config["cutoff"] / 16 <= stats["max_cutoff"] <= config["cutoff"]
        structure = to_structure(atoms, config["cutoff"], skin=0.5)
        selection = select_edges(
            structure,
            config["num_neighbors_adaptive"],
            config["cutoff"],
            config["cutoff_width_adaptive"],
            method=config["adaptive_cutoff_method"],
        )
        assert stats["k_max"] == int(selection.counts.max())


def test_express_validation():
    assert Express.from_any(None).is_stock()
    assert Express.from_any({"solver": "sorted_window"}).solver == "sorted_window"
    assert Express().model_updates() == {}
    assert Express(tokens="flat").model_updates() == {"tokens": "flat"}
    with pytest.raises(ValueError):
        Express(solver="fast")
    with pytest.raises(ValueError):
        Express(attention_tile=32)  # needs flat tokens
    with pytest.raises(ValueError):
        Express(tokens="flat", fold_extra_block=True)  # needs a tile


def test_express_recorded(checkpoint, frames):
    calc = UPETCalculator.from_checkpoint(
        str(checkpoint), express={"solver": "segment_sum"}
    )
    atoms = frames[0].copy()
    atoms.calc = calc
    atoms.get_potential_energy()
    assert calc.debug_stats["express"] == Express().as_dict()
