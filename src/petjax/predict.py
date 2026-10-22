"""Forward + autodiff wrapper for UPETCalculator — the in-JIT entry point.

Given a structure dict, ``get_predict_fn`` returns a JIT'd ``predict_fn(
params, structure) -> {energy, forces, stress, overflow}`` — the only
``@jax.jit`` site in this module. ``params`` is a runtime argument, not
closured. Forces/stress come from autodiff of the energy (forces off
``grad["positions"]``, stress off the strain argument), or straight from the
model's non-conservative heads when enabled — autodiff is skipped for
whichever output a head serves. The training form ``value_and_grad(loss,
argnums=0)(params, batch)`` comes for free.
"""

import jax
import jax.numpy as jnp

from .express import Express
from .select import truncate

# -- predict_fn factory --


def get_predict_fn(
    model, stress=True, no_shadow=False, num_neighbors_adaptive=None, express=None
):
    """Build a JIT-compiled ``predict_fn(params, structure) -> dict``.

    Shape-agnostic: shapes are read off the input arrays (``positions.shape[0]``
    and ``k_sel_sizer.shape[-1]``); ``params`` is a runtime arg, not closured,
    so the same closure serves any params and a training-side
    ``value_and_grad(loss, argnums=0)(params, batch)`` works without rebuild.

    The model carries its own metadata: the adaptive-selection hypers
    (``cutoff``, ``cutoff_width_adaptive``, ``adaptive_cutoff_method``) are
    read off ``model`` and closured as trace-time constants.
    ``num_neighbors_adaptive`` (the per-atom selection target) defaults to
    ``model.num_neighbors_adaptive`` but can be overridden by the caller — the
    value is closured into the forward, so the k_sel sizing on the calculator
    side must be passed the same value. Energy returned already includes the
    energy scale; composition shifts are the caller's
    responsibility (applied post-JIT in fp64 by ``UPETCalculator``). See
    ``_select_and_predict`` for ``no_shadow``. ``express`` (an ``Express``, a
    dict of its fields, or ``None`` for the stock path) selects the
    performance switches; the model must carry the matching attributes
    (``Express.model_updates``), which ``UPETCalculator`` arranges.

    Besides energy, forces, stress and ``overflow``, the result carries the
    selection's reach for the caller's guards: ``max_cutoff`` (largest adaptive
    cutoff of a real atom) and ``k_max`` (largest selected count).
    """
    express = Express.from_any(express)
    if num_neighbors_adaptive is None:
        num_neighbors_adaptive = model.num_neighbors_adaptive
    cutoff_width_adaptive = model.cutoff_width_adaptive
    adaptive_cutoff_method = model.adaptive_cutoff_method
    cutoff = model.cutoff

    # Which outputs need autodiff vs. come straight from a non-conservative head.
    direct_forces = bool(model.direct_forces)
    direct_stress = bool(model.direct_stress) and stress
    ad_forces = not direct_forces  # forces via d/d positions
    ad_stress = stress and not direct_stress  # stress via d/d strain

    def energy_fn(params, structure, epsilon=None):
        return _select_and_predict(
            model,
            params,
            structure,
            num_neighbors_adaptive,
            cutoff,
            cutoff_width_adaptive,
            adaptive_cutoff_method,
            epsilon=epsilon,
            no_shadow=no_shadow,
            express=express,
        )

    def strained_energy_fn(p, s, eps):
        return energy_fn(p, s, epsilon=eps)

    @jax.jit
    def predict_fn(params, structure):
        atom_mask = structure["atom_mask"][..., None]
        if ad_forces and ad_stress:
            (energy, aux), (grad_structure, virial) = jax.value_and_grad(
                strained_energy_fn, argnums=(1, 2), has_aux=True, allow_int=True
            )(params, structure, jnp.zeros((3, 3)))
            forces = -grad_structure["positions"] * atom_mask
            stress_out = virial
        elif ad_forces:  # AD forces; direct (or no) stress
            (energy, aux), grad_structure = jax.value_and_grad(
                energy_fn, argnums=1, has_aux=True, allow_int=True
            )(params, structure)
            forces = -grad_structure["positions"] * atom_mask
            stress_out = aux.get("stress")
        elif ad_stress:  # direct forces; AD stress
            (energy, aux), virial = jax.value_and_grad(
                strained_energy_fn, argnums=2, has_aux=True, allow_int=True
            )(params, structure, jnp.zeros((3, 3)))
            forces = aux["forces"]
            stress_out = virial
        else:  # both direct — no autodiff
            energy, aux = energy_fn(params, structure)
            forces = aux["forces"]
            stress_out = aux.get("stress")

        result = {
            "energy": energy,
            "forces": forces,
            "overflow": aux["overflow"],
            "max_cutoff": aux["max_cutoff"],
            "k_max": aux["k_max"],
        }
        if stress:
            result["stress"] = stress_out
        return result

    return predict_fn


# -- energy-fn body --


def _select_and_predict(
    model,
    params,
    structure,
    num_neighbors_adaptive,
    cutoff,
    cutoff_width_adaptive,
    adaptive_cutoff_method,
    epsilon=None,
    no_shadow=False,
    express=None,
):
    """Thin: (optional) strain → truncate → model → (energy, aux).

    Returns ``(total_energy, aux)`` where ``aux`` always carries ``overflow``,
    ``max_cutoff`` and ``k_max`` and, for a direct-head model, the per-call ``forces`` (``[N, 3]``) and
    ``stress`` (``[3, 3]``, summed over atoms). Composition shifts are applied
    post-JIT by the caller in fp64. ``no_shadow=True`` cuts gradients through
    the adaptive-cutoff function while leaving its values in the forward pass.

    The model's per-atom outputs are already scaled
    and masked by ``atom_mask`` — we just sum.
    """
    if epsilon is not None:
        structure = apply_strain(structure, epsilon)

    truncated, truncation = truncate(
        structure,
        num_neighbors_adaptive,
        cutoff,
        cutoff_width_adaptive,
        method=adaptive_cutoff_method,
        no_shadow=no_shadow,
        express=express,
    )
    out = model.apply(params, **truncated)
    atom_mask = structure["atom_mask"]
    aux = {
        "overflow": truncation.overflow,
        "max_cutoff": jnp.max(jnp.where(atom_mask, truncation.atomic_cutoffs, 0.0)),
        "k_max": jnp.max(jnp.where(atom_mask, truncation.counts, 0)),
    }
    if isinstance(out, dict):
        energy = jnp.sum(out["energy"])
        if "forces" in out:
            # Raw per-atom output; net-force removal is applied calculator-side.
            aux["forces"] = out["forces"]
        if "stress" in out:
            aux["stress"] = jnp.sum(out["stress"], axis=0)
        return energy, aux
    return jnp.sum(out), aux


def apply_strain(structure, epsilon):
    """Homogeneous strain ``F = I + epsilon`` applied to positions and cell;
    a structure→structure transform used to take ``d/d epsilon`` for stress."""
    F = jnp.eye(3) + epsilon
    return {
        **structure,
        "positions": structure["positions"] @ F,
        "cell": structure["cell"] @ F,
    }
