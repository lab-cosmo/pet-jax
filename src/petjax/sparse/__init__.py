"""Hessian sparsity pattern and star coloring on a pair list, via asdex.

The asdex-bound layer of the sparse-Hessian path; requires the ``sparse``
extra (``asdex``, ``scipy``). Model-agnostic on purpose: pair list in, asdex
objects out, no PET vocabulary, so that it can move into ``asdex`` later. The
PET glue that consumes it lives in ``petjax.hessian``; the entry point is
``UPETCalculator.hessian``.
"""

from .coloring import hessian_coloring, lift_atom_coloring
from .pattern import sparsity_patterns

__all__ = ["hessian_coloring", "lift_atom_coloring", "sparsity_patterns"]
