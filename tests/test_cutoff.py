import numpy as np
import jax.numpy as jnp

from petjax.utils import cutoff_bump


def test_bump_tail_in_float32_follows_float64():
    """The attention takes log(f), so f must not round to zero in the tail."""
    r = np.linspace(4.5, 4.9999, 2001).astype(np.float32)
    f32 = np.asarray(cutoff_bump(jnp.asarray(r), 5.0, 0.5))
    x = np.clip((r.astype(np.float64) - 4.5) / 0.5, 1e-6, 1 - 1e-6)
    f64 = 1 / (1 + np.exp(-2 / np.tan(np.pi * x)))
    tail = f64 > 1e-30
    assert tail.sum() > 1900
    np.testing.assert_allclose(f32[tail], f64[tail], rtol=1e-3)
