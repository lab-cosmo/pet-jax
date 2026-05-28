"""Shared utilities."""

import jax
import jax.numpy as jnp


def cutoff_bump(r, cutoff, width=0.5):
    """Smooth bump cutoff function."""
    x = (r - (cutoff - width)) / width
    x_safe = jnp.clip(x, 1e-6, 1 - 1e-6)
    bump = 0.5 * (1 + jnp.tanh(1 / jnp.tan(jnp.pi * x_safe)))
    return jnp.where(x <= 0, 1.0, jnp.where(x >= 1, 0.0, bump))


def edge_displacements(positions, centers, others, cell_shifts, cell):
    """Per-edge displacement vectors ``R_j - R_i + S @ h`` in Cartesian coords."""
    return positions[others] - positions[centers] + cell_shifts @ cell


def safe_norm(x, axis=-1, eps=1e-15):
    # PET-style: the +eps keeps sqrt's gradient finite at x=0. e3x-style
    # safe_norm (scaled + custom_jvp) is cleaner, but the very small diff
    # is enough to break some equivalance tests...
    return jnp.sqrt(jnp.sum(jax.lax.square(x), axis=axis) + eps)


def cast_floats(tree, dtype):
    """Cast floating-point leaves of a pytree to ``dtype``.

    Non-array leaves (e.g. Python scalars in metadata) are left alone;
    numpy arrays are converted to JAX arrays via ``jnp.asarray``.
    """

    def _cast(x):
        if not hasattr(x, "dtype"):
            return x
        if jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.asarray(x, dtype=dtype)
        return x

    return jax.tree.map(_cast, tree)
