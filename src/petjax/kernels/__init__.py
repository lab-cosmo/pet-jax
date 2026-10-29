"""Hand-written GPU kernels (CuTe DSL) behind ``Express`` switches.

Optional: ``pip install pet-jax[cute]`` (``nvidia-cutlass-dsl[cu13]``, linux and
CUDA 13 only). Every kernel reproduces the stock JAX path to fp32 round-off and
comes with its backward, so ``jax.grad`` through it works like through the
stock path. The modules import ``cutlass`` lazily; ``available()`` says
whether the toolchain is there.
"""

import importlib.util


def available():
    """Whether the CuTe DSL and its JAX integration can be imported."""
    if importlib.util.find_spec("cutlass") is None:
        return False
    import cutlass.jax

    return bool(cutlass.jax.is_available())


__all__ = ["available"]
