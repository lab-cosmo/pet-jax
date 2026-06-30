"""Shared test fixtures and configuration.

The CI-friendly default runs the "mini" suite: 12 structures in
`tests/assets/test_mini.xyz` and the LFS-tracked `pet-mad-xs` checkpoint
under `tests/assets/checkpoints/pet-mad-xs/`.

The extended suite covers the full `test_s` / `test_m` / `test_l` datasets
and (optionally) `pet-mad-s`. Extended assets are gitignored; populate them
locally with `petjax-convert` (for the pet-mad-s checkpoint) and local
inference runs (for the larger reference prediction files). Opt in with
`pytest --run-extended`; extended tests that cannot find their inputs skip
individually.
"""

from pathlib import Path

import pytest

ASSETS = Path(__file__).parent / "assets"


def pytest_addoption(parser):
    parser.addoption(
        "--run-extended",
        action="store_true",
        default=False,
        help="Run the extended test suite (full test_s/m/l + pet-mad-s).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "extended: mark a test as part of the extended local-only suite",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-extended"):
        return
    skip_extended = pytest.mark.skip(reason="extended suite (pass --run-extended)")
    for item in items:
        if "extended" in item.keywords:
            item.add_marker(skip_extended)


@pytest.fixture(scope="session")
def assets_dir():
    return ASSETS


@pytest.fixture(scope="session")
def mini_xyz():
    path = ASSETS / "test_mini.xyz"
    if not path.exists():
        pytest.skip("tests/assets/test_mini.xyz is missing — did you pull git-lfs objects?")
    return path


@pytest.fixture(scope="session")
def mini_predictions_xs():
    path = ASSETS / "predictions" / "test_mini_pet-mad-xs.xyz"
    if not path.exists():
        pytest.skip("mini reference predictions missing (git-lfs pull?)")
    return path


@pytest.fixture(scope="session")
def mini_predictions_xs_direct():
    path = ASSETS / "predictions" / "test_mini_pet-mad-xs_direct.xyz"
    if not path.exists():
        pytest.skip("mini direct reference predictions missing (git-lfs pull?)")
    return path


@pytest.fixture(scope="session")
def pet_mad_xs_checkpoint():
    path = ASSETS / "checkpoints" / "pet-mad-xs"
    if not (path / "model.msgpack").exists():
        pytest.skip(
            "pet-mad-xs checkpoint missing — git-lfs pull or `petjax-convert pet-mad-xs`"
        )
    return path


@pytest.fixture(scope="session")
def pet_mad_s_checkpoint():
    path = ASSETS / "checkpoints" / "pet-mad-s"
    if not (path / "model.msgpack").exists():
        pytest.skip("pet-mad-s checkpoint missing (run `petjax-convert pet-mad-s`)")
    return path
