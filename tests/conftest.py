"""Shared test fixtures and configuration.

The CI-friendly default runs the "mini" suite: 12 structures in
`tests/assets/test_mini.xyz` against every checkpoint in `MINI_RELEASES`.

Converted checkpoints are gitignored build artifacts — run
`tox -e fetch-checkpoints` to rebuild them from the upstream `.ckpt` files.
Only the reference `.xyz` predictions are tracked.

The extended suite covers the full `test_s` / `test_m` / `test_l` datasets
and (optionally) `pet-mad-s`. Extended assets are gitignored too; populate
them locally with `petjax-convert` (for the pet-mad-s checkpoint) and
`tests/generate_references.py` (for the larger reference prediction files).
Opt in with `pytest --run-extended`; extended tests that cannot find their
inputs skip individually.
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
        pytest.skip("tests/assets/test_mini.xyz is missing")
    return path


# PET-MAD releases the mini suite runs against. Two of them, because v1.6
# renamed the direct-force readout target (`non_conservative_forces` ->
# `non_conservative_force`) without bumping the checkpoint version, so only a
# real checkpoint of each spelling pins that the converter handles both. The
# v1.5 assets keep the unversioned `pet-mad-xs` name that the extended suite's
# local-only prediction files are also named after.
MINI_RELEASES = ("pet-mad-xs", "pet-mad-xs-v1.6")


@pytest.fixture(scope="session", params=MINI_RELEASES)
def mini_release(request):
    """(checkpoint dir, conservative reference, direct reference) per release."""
    name = request.param
    checkpoint = ASSETS / "checkpoints" / name
    conservative = ASSETS / "predictions" / f"test_mini_{name}.xyz"
    direct = ASSETS / "predictions" / f"test_mini_{name}_direct.xyz"
    missing = [
        path.name
        for path in (checkpoint / "model.msgpack", conservative, direct)
        if not path.exists()
    ]
    if missing:
        pytest.skip(f"mini assets for {name} missing: {', '.join(missing)}")
    return checkpoint, conservative, direct


@pytest.fixture(scope="session")
def pet_mad_xs_checkpoint():
    path = ASSETS / "checkpoints" / "pet-mad-xs"
    if not (path / "model.msgpack").exists():
        pytest.skip(
            "pet-mad-xs (v1.5) checkpoint missing — run "
            "`petjax-convert pet-mad-xs --version 1.5.0`"
        )
    return path


@pytest.fixture(scope="session")
def pet_omat_xs_checkpoint():
    path = ASSETS / "checkpoints" / "pet-omat-xs"
    if not (path / "model.msgpack").exists():
        pytest.skip(
            "pet-omat-xs checkpoint missing — run `tox -e fetch-checkpoints` or "
            "`petjax-convert pet-omat-xs-v1.0.0`"
        )
    return path


@pytest.fixture(scope="session")
def pet_mad_s_checkpoint():
    path = ASSETS / "checkpoints" / "pet-mad-s"
    if not (path / "model.msgpack").exists():
        pytest.skip("pet-mad-s checkpoint missing (run `petjax-convert pet-mad-s`)")
    return path
