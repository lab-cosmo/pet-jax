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
    energy scale; composition shifts are the caller's
    responsibility (applied post-JIT in fp64 by ``UPETCalculator``). See
    ``_select_and_predict`` for ``no_shadow``.
    """
    probes = model.get_probes()
    if num_neighbors_adaptive is None:
        num_neighbors_adaptive = model.num_neighbors_adaptive
    cutoff_width = model.cutoff_width

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
            probes,
            num_neighbors_adaptive,
            cutoff_width,
            epsilon=epsilon,
            no_shadow=no_shadow,
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

        result = {"energy": energy, "forces": forces, "overflow": aux["overflow"]}
        if stress:
            result["stress"] = stress_out
        return result

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
    """Thin: (optional) strain → truncate → model → (energy, aux).

    Returns ``(total_energy, aux)`` where ``aux`` always carries ``overflow``
    and, for a direct-head model, the per-call ``forces`` (``[N, 3]``) and
    ``stress`` (``[3, 3]``, summed over atoms). Composition shifts are applied
    post-JIT by the caller in fp64. ``no_shadow=True`` cuts gradients through
    the adaptive-cutoff function while leaving its values in the forward pass.

    The model's per-atom outputs are already scaled
    and masked by ``atom_mask`` — we just sum.
    """
    if epsilon is not None:
        structure = apply_strain(structure, epsilon)

    truncated, overflow = truncate(
        structure, probes, cutoff_width, num_neighbors_adaptive, no_shadow=no_shadow
    )
    out = model.apply(params, **truncated)
    aux = {"overflow": overflow}
    if isinstance(out, dict):
        energy = jnp.sum(out["energy"])
        if "forces" in out:
            # Raw per-atom head output (does not sum to zero); the net-force
            # (drift) removal is an inference-only correction applied calculator
            # -side, so predict stays faithful for a training value_and_grad.
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
