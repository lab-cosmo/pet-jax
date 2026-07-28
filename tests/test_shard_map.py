"""Adaptive cutoffs under shard_map (data-parallel training).

Callers such as iris wrap the step in `jax.shard_map` over a data-parallel
mesh axis, which makes every array derived from the local batch "varying"
over that axis. Code traced inside must keep loop carries varying too. The
device count is forced in conftest, so these run on CPU -- no GPU needed.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import pytest

from petjax.select import get_adaptive_cutoffs, get_adaptive_cutoffs_solver

N_ATOMS = 64
N_PAIRS = 512
CUTOFF = 7.5
CUTOFF_WIDTH = 1.0
NUM_NEIGHBORS = 16

pytestmark = pytest.mark.skipif(
    jax.device_count() < 2, reason="needs >= 2 devices (XLA_FLAGS device count)"
)


def _inputs(n_dev):
    rng = np.random.default_rng(0)
    return (
        jnp.asarray(rng.integers(0, N_ATOMS, size=(n_dev, N_PAIRS)), dtype=jnp.int32),
        jnp.asarray(rng.uniform(0.5, CUTOFF, size=(n_dev, N_PAIRS)), dtype=jnp.float32),
        jnp.asarray(rng.random((n_dev, N_PAIRS)) > 0.1),
    )


def _sharded(fn, mesh):
    """Mirror the training wrapper: shard on the leading axis, squeeze, call."""

    @jax.jit
    def wrapped(*args):
        @jax.shard_map(mesh=mesh, in_specs=(P("dp"),) * len(args), out_specs=P("dp"))
        def inner(*args):
            return fn(*[jnp.squeeze(a, 0) for a in args])[None]

        return inner(*args)

    return wrapped


@pytest.mark.parametrize(
    "method",
    [get_adaptive_cutoffs_solver, get_adaptive_cutoffs],
    ids=["solver", "grid"],
)
def test_adaptive_cutoffs_match_single_device(method):
    n_dev = jax.device_count()
    centers, r_ij, pair_mask = _inputs(n_dev)

    def solve(centers, r_ij, pair_mask):
        return method(
            centers, r_ij, pair_mask, NUM_NEIGHBORS, N_ATOMS, CUTOFF, CUTOFF_WIDTH
        )

    ref = jnp.stack([solve(centers[i], r_ij[i], pair_mask[i]) for i in range(n_dev)])

    mesh = jax.make_mesh((n_dev,), ("dp",))
    shard = NamedSharding(mesh, P("dp"))
    args = [jax.device_put(x, shard) for x in (centers, r_ij, pair_mask)]
    got = _sharded(solve, mesh)(*args)

    # 1 ULP: XLA fuses the segment sums differently under shard_map
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-6, atol=1e-6)

    with jax.set_mesh(mesh):  # the scalar output lives on the mesh
        grad = jax.jit(
            jax.grad(lambda r: _sharded(solve, mesh)(args[0], r, args[2]).sum())
        )(args[1])
    assert jnp.all(jnp.isfinite(grad))
