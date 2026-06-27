"""Forward + autodiff wrapper for UPETCalculator — the in-JIT entry point.

Given a structure dict, ``get_predict_fn`` returns a JIT'd ``predict_fn(
params, structure) -> {energy, forces, stress, overflow}`` — the only
``@jax.jit`` site in this module. ``params`` is a runtime argument, not
closured; forces are read off ``grad["positions"]``; stress comes from
differentiating w.r.t. the strain argument. The same closure serves any
params, and the training form ``value_and_grad(loss, argnums=0)(params,
batch)`` comes for free.
"""

import jax
import jax.numpy as jnp

from .select import truncate

# -- predict_fn factory --


def get_predict_fn(model, stress=True, no_shadow=False, num_neighbors_adaptive=None):
    """Build a JIT-compiled ``predict_fn(params, structure) -> dict``.

    Shape-agnostic: shapes are read off the input arrays (``positions.shape[0]``
    and ``k_sel_sizer.shape[-1]``); ``params`` is a runtime arg, not closured,
    so the same closure serves any params and a training-side
    ``value_and_grad(loss, argnums=0)(params, batch)`` works without rebuild.

    The model carries its own metadata: ``cutoff_width`` is read off ``model``.
    ``num_neighbors_adaptive`` (the per-atom selection target) defaults to
    ``model.num_neighbors_adaptive`` but can be overridden by the caller — the
    value is closured into the forward, so the k_sel sizing on the calculator
    side must be passed the same value. Energy returned already includes the
    energy scale (a loaded parameter); composition shifts are the caller's
    responsibility (applied post-JIT in fp64 by ``UPETCalculator``). See
    ``_select_and_predict`` for ``no_shadow``.
    """
    probes = model.get_probes()
    if num_neighbors_adaptive is None:
        num_neighbors_adaptive = model.num_neighbors_adaptive
    cutoff_width = model.cutoff_width

    def energy_fn(params, structure, epsilon=None):
        return _select_and_predict(
            model,
            params,
            structure,
            probes,
            num_neighbors_adaptive,
            cutoff_width,
            epsilon=epsilon,
            no_shadow=no_shadow,
        )

    if stress:

        @jax.jit
        def predict_fn(params, structure):
            def e_fn(p, s, eps):
                return energy_fn(p, s, epsilon=eps)

            (energy, overflow), (grad_structure, virial) = jax.value_and_grad(
                e_fn, argnums=(1, 2), has_aux=True, allow_int=True
            )(params, structure, jnp.zeros((3, 3)))
            forces = -grad_structure["positions"] * structure["atom_mask"][..., None]
            return {
                "energy": energy,
                "forces": forces,
                "stress": virial,
                "overflow": overflow,
            }

    else:

        @jax.jit
        def predict_fn(params, structure):
            def e_fn(p, s):
                return energy_fn(p, s)

            (energy, overflow), grad_structure = jax.value_and_grad(
                e_fn, argnums=1, has_aux=True, allow_int=True
            )(params, structure)
            forces = -grad_structure["positions"] * structure["atom_mask"][..., None]
            return {
                "energy": energy,
                "forces": forces,
                "overflow": overflow,
            }

    return predict_fn


# -- energy-fn body --


def _select_and_predict(
    model,
    params,
    structure,
    probes,
    num_neighbors_adaptive,
    cutoff_width,
    epsilon=None,
    no_shadow=False,
):
    """Thin: (optional) strain → truncate → model → sum.

    Returns ``(total_energy, overflow)``; composition shifts are applied
    post-JIT by the caller in fp64. ``no_shadow=True`` cuts gradients through
    the adaptive-cutoff function while leaving its values in the forward pass.

    The model's per-atom output is already scaled by the energy-scale parameter
    and masked by ``atom_mask`` — we just sum.
    """
    if epsilon is not None:
        structure = apply_strain(structure, epsilon)

    truncated, overflow = truncate(
        structure, probes, cutoff_width, num_neighbors_adaptive, no_shadow=no_shadow
    )
    per_atom = model.apply(params, **truncated)
    return jnp.sum(per_atom), overflow


def apply_strain(structure, epsilon):
    """Homogeneous strain ``F = I + epsilon`` applied to positions and cell;
    a structure→structure transform used to take ``d/d epsilon`` for stress."""
    F = jnp.eye(3) + epsilon
    return {
        **structure,
        "positions": structure["positions"] @ F,
        "cell": structure["cell"] @ F,
    }
