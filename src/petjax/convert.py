"""Convert metatrain PET ``.ckpt`` files to pet-jax's Flax msgpack layout.

Reads the Lightning-style ``.ckpt`` directly — no ``mtt export`` / TorchScript
intermediate. Accepts both layouts published on Hugging Face
(``lab-cosmo/upet``): bare PET checkpoints (the pet-omat / pet-omad / … lines)
and LLPR-wrapped ones (the PET-MAD releases; wrapper v3 or v4 — the wrapper
state itself is never read). Inner PET checkpoint versions 10 through 16 are
supported; the between-version differences (new hypers, the v13 scaler split,
the v16 ``backend.`` state-dict prefix) are absorbed here, mirroring
metatrain's own upgrade rules. Older checkpoints fail hard — run
``mtt upgrade``; newer ones fail hard until pet-jax catches up.

Writes:
    {output_dir}/model.msgpack    -- Flax parameter tree
    {output_dir}/metadata.yaml    -- config, shifts
"""

import numpy as np
import jax.numpy as jnp

import io
import re
import zipfile
from pathlib import Path

from marathon.io import write_msgpack, write_yaml

# -- accepted checkpoint versions --

OUTER_ARCH = "llpr"
OUTER_VERSIONS = (3, 4)  # v3→v4 only reworks LLPR covariance buffers, never read
INNER_ARCH = "pet"
INNER_MIN_VERSION = 10  # v9 and older imply the Cosine cutoff era — rejected anyway
INNER_MAX_VERSION = 16

# -- architecture knobs pet-jax hard-implements (must match the checkpoint) --

REQUIRED_HYPERS = {
    "normalization": "RMSNorm",
    "activation": "SwiGLU",
    "transformer_type": "PreLN",
    "featurizer_type": "feedforward",
    "cutoff_function": "Bump",
    "zbl": False,
}

# -- subset of model_hypers pet-jax's UPET actually consumes --

CONFIG_KEYS = (
    "d_pet",
    "d_node",
    "d_head",
    "d_feedforward",
    "num_heads",
    "num_attention_layers",
    "num_gnn_layers",
    "cutoff",
    "cutoff_width",
    "num_neighbors_adaptive",
)


# -- public API --


def convert_checkpoint(ckpt_path, output_dir):
    """Convert a metatrain ``.ckpt`` to pet-jax's ``model.msgpack`` + ``metadata.yaml``."""
    import metatomic.torch  # noqa: F401  (registers ModelMetadata ScriptObject)
    import torch

    ckpt_path = Path(ckpt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")

    pet_ckpt = _unwrap_pet_checkpoint(ckpt)
    # metatrain ckpt v16 moved the PET core under a `backend.` submodule;
    # strip the prefix so one rename pipeline serves all versions. Scaler /
    # additive-model keys were not moved, so metadata extraction is unaffected.
    state_dict = {
        k.removeprefix("backend."): v for k, v in pet_ckpt["best_model_state_dict"].items()
    }
    _check_single_readout(state_dict)
    meta = _extract_metadata(pet_ckpt)

    flat = _convert_state_dict(state_dict)
    n_rows = meta["config"]["max_atomic_number"] + 1
    _scatter_species_embeddings(flat, meta["atomic_types"], n_rows)

    # Scales live in the parameter tree, not metadata. force_scale is
    # per-species (scattered to Z rows); energy/stress scales are scalars.
    rows = np.asarray([int(z) for z in meta["atomic_types"]])
    flat["energy_scale"] = jnp.array([meta["energy_scale"]], dtype=jnp.float32)
    if meta["force_scale"] is not None:
        force_scale = np.zeros(n_rows, dtype=np.float32)
        force_scale[rows] = meta["force_scale"]
        flat["force_scale"] = jnp.array(force_scale)
    if meta["stress_scale"] is not None:
        flat["stress_scale"] = jnp.array([meta["stress_scale"]], dtype=jnp.float32)
    params = _unflatten(flat)

    write_msgpack(output_dir / "model.msgpack", params)
    metadata = {
        "config": meta["config"],
        "shifts": meta["shifts"],
    }
    write_yaml(output_dir / "metadata.yaml", metadata)

    print(f"Saved to {output_dir}/")
    print(f"  config: {meta['config']}")
    print(f"  energy_scale: {meta['energy_scale']:.6f}")
    print(f"  num atomic types: {len(meta['atomic_types'])}")
    return params, metadata


def load_checkpoint(checkpoint_dir):
    """Load a pet-jax checkpoint (``model.msgpack`` + ``metadata.yaml``).

    Configs written before the adaptive-selection hypers existed are upgraded
    in place, mirroring metatrain's own checkpoint migration: absent
    ``adaptive_cutoff_method`` means "grid" (ckpt v11→v12 rule), absent
    ``cutoff_width_adaptive`` means the shared ``cutoff_width`` (v13→v14 rule).
    """
    from marathon.io import read_msgpack, read_yaml

    checkpoint_dir = Path(checkpoint_dir)
    params = read_msgpack(checkpoint_dir / "model.msgpack")
    metadata = read_yaml(checkpoint_dir / "metadata.yaml")
    config = metadata["config"]
    config.setdefault("adaptive_cutoff_method", "grid")
    config.setdefault("cutoff_width_adaptive", config["cutoff_width"])
    return params, metadata


# -- ckpt unwrap + metadata extraction --


def _unwrap_pet_checkpoint(ckpt):
    """Validate versions and return the inner PET dict — either the checkpoint
    itself (bare PET, e.g. the pet-omat line) or the one nested inside the LLPR
    wrapper (PET-MAD releases). The wrapper's own state is never read, so any
    wrapper version that keeps the inner checkpoint at
    ``wrapped_model_checkpoint`` is acceptable."""
    arch = ckpt.get("architecture_name")
    if arch == OUTER_ARCH:
        outer_ver = ckpt.get("model_ckpt_version")
        if outer_ver not in OUTER_VERSIONS:
            raise ValueError(
                f"pet-jax accepts LLPR wrapper versions {OUTER_VERSIONS}; got "
                f"v{outer_ver}. Run `uv run --with metatrain mtt upgrade` on the "
                f".ckpt or fetch a newer release."
            )
        inner = ckpt.get("wrapped_model_checkpoint")
        if not isinstance(inner, dict):
            raise ValueError(
                "missing 'wrapped_model_checkpoint' in outer ckpt — the LLPR "
                "wrapper is expected to carry the PET model nested inside."
            )
    elif arch == INNER_ARCH:
        inner = ckpt
    else:
        raise ValueError(
            f"pet-jax expects a PET checkpoint, bare ({INNER_ARCH!r}) or "
            f"LLPR-wrapped ({OUTER_ARCH!r}); got architecture_name={arch!r}."
        )

    inner_arch = inner.get("architecture_name")
    inner_ver = inner.get("model_ckpt_version")
    if inner_arch != INNER_ARCH:
        raise ValueError(f"pet-jax expects inner {INNER_ARCH!r}; got {inner_arch!r}.")
    if not isinstance(inner_ver, int) or inner_ver < INNER_MIN_VERSION:
        raise ValueError(
            f"pet-jax accepts PET checkpoint versions {INNER_MIN_VERSION}.."
            f"{INNER_MAX_VERSION}; got v{inner_ver}. Run `mtt upgrade` on the "
            f"source ckpt."
        )
    if inner_ver > INNER_MAX_VERSION:
        raise ValueError(
            f"pet-jax accepts PET checkpoint versions {INNER_MIN_VERSION}.."
            f"{INNER_MAX_VERSION}; got v{inner_ver}, which is newer than this "
            f"pet-jax release knows about — update pet-jax."
        )

    return inner


def _check_single_readout(state_dict):
    """Reject multi-readout checkpoints: ``UPET`` consumes only readout head 0
    (``num_readout_layers == 1``, which the feedforward featurizer guarantees),
    so a residual-featurizer ckpt would silently lose every head past index 0.
    """
    indices = set()
    for key in state_dict:
        match = re.match(r"node_heads\.energy\.(\d+)\.", key)
        if match:
            indices.add(int(match.group(1)))
    if indices != {0}:
        raise ValueError(
            f"pet-jax implements num_readout_layers == 1 (a single readout from "
            f"the final GNN layer); checkpoint exposes readout-head indices "
            f"{sorted(indices)}. Only the feedforward featurizer is supported."
        )


def _extract_metadata(pet_ckpt):
    """Hypers, species mapping, scaler, composition shifts — all direct lookups."""
    model_data = pet_ckpt["model_data"]
    hypers = model_data["model_hypers"]

    for k, expected in REQUIRED_HYPERS.items():
        got = hypers.get(k)
        if got != expected:
            raise ValueError(
                f"pet-jax requires {k}={expected!r}; checkpoint has {got!r}. "
                f"pet-jax implements only the PET-MAD-shaped variant of PET."
            )
    if hypers.get("long_range", {}).get("enable", False):
        raise ValueError(
            "pet-jax does not implement long-range corrections; checkpoint has "
            "long_range.enable=True."
        )
    # metatrain ckpt v12 split the adaptive-selection algorithm into
    # "grid"/"solver"; all v11 checkpoints trained with grid, so an absent key
    # means grid (metatrain's own v11→v12 upgrade rule).
    method = hypers.get("adaptive_cutoff_method", "grid").lower()
    if method not in ("grid", "solver"):
        raise ValueError(
            f"unknown adaptive_cutoff_method {method!r} in checkpoint; "
            f"pet-jax implements 'grid' and 'solver'."
        )
    # metatrain ckpt v15 added charge/spin conditioning; pet-jax has no
    # equivalent embedding, so a conditioned model would be silently wrong.
    if hypers.get("system_conditioning", False):
        raise ValueError(
            "pet-jax does not implement system conditioning (charge/spin "
            "embeddings); checkpoint has system_conditioning=True."
        )

    config = {k: hypers[k] for k in CONFIG_KEYS}
    config["adaptive_cutoff_method"] = method
    # Hyper fallbacks mirror metatrain's own upgrade rules for checkpoints
    # predating each hyper: attention_temperature (v10→v11) and the
    # adaptive-selection taper width split off cutoff_width (v13→v14).
    config["attention_temperature"] = hypers.get("attention_temperature", 1.0)
    config["cutoff_width_adaptive"] = hypers.get(
        "cutoff_width_adaptive", hypers["cutoff_width"]
    )

    atomic_types = list(model_data["dataset_info"].atomic_types)
    config["max_atomic_number"] = max(int(z) for z in atomic_types)

    state_dict = pet_ckpt["best_model_state_dict"]

    energy_values = _scaler_values(state_dict, "energy")
    if energy_values is None:
        raise ValueError("checkpoint carries no energy scaler buffer.")
    energy_scale = float(energy_values.item())

    # Non-conservative scales (direct-capable checkpoints only): force is one
    # std per trained species, stress a scalar.
    force_scale = _scaler_values(state_dict, "non_conservative_forces")
    stress_values = _scaler_values(state_dict, "non_conservative_stress")
    stress_scale = float(stress_values.item()) if stress_values is not None else None

    comp = parse_metatensor_buffer(
        state_dict["additive_models.0.energy_composition_buffer"]
    )
    comp_samples = comp["blocks/0/samples.npy"]
    comp_values = comp["blocks/0/values.npy"].flatten()
    shifts = {int(s[0]): float(v) for s, v in zip(comp_samples, comp_values)}

    return {
        "config": config,
        "atomic_types": atomic_types,
        "energy_scale": energy_scale,
        "force_scale": force_scale,
        "stress_scale": stress_scale,
        "shifts": shifts,
    }


def _scaler_values(state_dict, target):
    """Scale values for ``target`` as a flat array, or None if absent.

    metatrain ckpt v13 split the single ``<target>_scaler_buffer`` into
    per-target × per-property factors whose product is the effective scale.
    Upgraded checkpoints carry all three buffers, fresh v13+ ones only the
    pair — prefer the pair, fall back to the old single buffer."""
    per_target_key = f"scaler.{target}_per_target_scaler_buffer"
    if per_target_key in state_dict:
        per_target = parse_metatensor_buffer(state_dict[per_target_key])
        per_property = parse_metatensor_buffer(
            state_dict[f"scaler.{target}_per_property_scaler_buffer"]
        )
        return (
            per_target["blocks/0/values.npy"].flatten()
            * per_property["blocks/0/values.npy"].flatten()
        )
    old_key = f"scaler.{target}_scaler_buffer"
    if old_key in state_dict:
        buf = parse_metatensor_buffer(state_dict[old_key])
        return buf["blocks/0/values.npy"].flatten()
    return None


def parse_metatensor_buffer(buf_tensor):
    """A metatensor TensorMap serialized to a uint8 torch buffer is a zip of .npy."""
    data = buf_tensor.numpy().tobytes()
    out = {}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            out[name] = np.load(io.BytesIO(zf.read(name)))
    return out


# -- state_dict -> Flax param tree --


_SKIP_PREFIXES = (
    "species_to_species_index",
    "scaler.",
    "additive_models.",
    "long_range_featurizer.",
)

# Readout-head params nest under energy_head/, the rest under backbone/
# (UPET = Backbone + Energy).
_HEAD_PREFIXES = ("node_heads_", "node_last_", "edge_heads_", "edge_last_")


def _convert_state_dict(state_dict):
    """Raw PyTorch state dict -> flat Flax param dict (keys scoped by module)."""
    import torch

    out = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith(_SKIP_PREFIXES):
            continue

        new_key = _rename_key(key)
        np_value = value.cpu().numpy()
        new_key, np_value = _finalize_key(new_key, np_value)
        out[_scope_key(new_key, key)] = jnp.array(np_value)
    return out


# Readout target name (in the source key) -> Flax head-module scope.
_HEAD_SCOPES = (
    ("non_conservative_forces", "forces_head"),
    ("non_conservative_stress", "stress_head"),
    ("energy", "energy_head"),
)


def _scope_key(key, orig_key):
    """Nest a flat key under its module scope: a per-target head module for
    readout heads (``energy_head`` / ``forces_head`` / ``stress_head``),
    ``backbone`` for everything else."""
    if not key.startswith(_HEAD_PREFIXES):
        return f"backbone.{key}"
    for tag, scope in _HEAD_SCOPES:
        if f".{tag}." in orig_key:
            return f"{scope}.{key}"
    return f"energy_head.{key}"


def _rename_key(key):
    """PyTorch state-dict key -> Flax parameter-tree key."""
    new_key = key

    new_key = new_key.replace("center_contraction", "center_contract")
    new_key = new_key.replace("center_expansion", "center_expand")
    new_key = new_key.replace("norm_attention", "norm_attn")
    new_key = new_key.replace("input_linear", "qkv")
    new_key = new_key.replace("output_linear", "out")
    new_key = new_key.replace("combination_norms", "comb_norms")
    new_key = new_key.replace("combination_mlps", "comb_mlps")

    if "gnn_layers" in new_key:
        new_key = new_key.replace("edge_embedder", "edge_embed")
        new_key = new_key.replace("neighbor_embedder", "neighbor_embed")

    if ".norm_center_features." in new_key:
        new_key = new_key.replace(".norm_center_features.", ".norm_center.")

    new_key = new_key.replace(".mlp.w_in.", ".mlp_in.")
    new_key = new_key.replace(".mlp.w_out.", ".mlp_out.")
    new_key = new_key.replace(".center_mlp.w_in.", ".center_mlp_in.")
    new_key = new_key.replace(".center_mlp.w_out.", ".center_mlp_out.")

    if new_key.endswith((".weight", ".bias")):
        parts = new_key.split(".")
        # Readout heads, any target name (energy / non_conservative_forces / ...):
        #   <head>.<target>.<readout>.<dense>.<w|b> -> <head>_<readout>.<dense>.<w|b>
        if parts[0] in ("node_heads", "edge_heads"):
            new_key = f"{parts[0]}_{parts[2]}." + ".".join(parts[3:])
        #   <head>_layers.<target>.<readout>.<target>___0.<w|b> -> <short>_<readout>.<w|b>
        elif parts[0] in ("node_last_layers", "edge_last_layers"):
            short = parts[0].replace("_layers", "")
            new_key = f"{short}_{parts[2]}." + ".".join(parts[4:])

    return new_key


def _finalize_key(new_key, np_value):
    """Apply Flax suffix rules (.weight -> .kernel/.embedding/.scale) and
    flatten layer indices to Flax's ``Module_<i>`` naming. Returns the final
    key and (possibly transposed) value."""
    if new_key.endswith(".weight"):
        is_true_embed = "embed" in new_key.lower() and ".edge_embed." not in new_key
        if is_true_embed:
            new_key = new_key.replace(".weight", ".embedding")
        elif "norms" in new_key or ".norm_" in new_key:
            new_key = new_key.replace(".weight", ".scale")
        else:
            if np_value.ndim == 2:
                np_value = np_value.T
            new_key = new_key.replace(".weight", ".kernel")

    new_key = re.sub(
        r"gnn_layers\.(\d+)\.trans\.layers\.(\d+)\.",
        r"gnn_layers_\1_trans_layers_\2.",
        new_key,
    )
    new_key = new_key.replace(".attention.", ".Attention_0.")
    new_key = re.sub(r"gnn_layers\.(\d+)\.", r"gnn_layers_\1_", new_key)
    new_key = re.sub(r"node_embedders\.(\d+)", r"node_embedders_\1", new_key)
    new_key = re.sub(r"comb_norms\.(\d+)", r"comb_norms_\1", new_key)
    new_key = re.sub(r"comb_mlps\.(\d+)\.", r"comb_mlps_\1.", new_key)

    for regex, sub in (
        (r"_compress\.0\.", r"_compress.Dense_0."),
        (r"_compress\.2\.", r"_compress.Dense_1."),
        (r"comb_mlps_(\d+)\.0\.", r"comb_mlps_\1.Dense_0."),
        (r"comb_mlps_(\d+)\.2\.", r"comb_mlps_\1.Dense_1."),
        (r"node_heads_(\d+)\.0\.", r"node_heads_\1.Dense_0."),
        (r"node_heads_(\d+)\.2\.", r"node_heads_\1.Dense_1."),
        (r"edge_heads_(\d+)\.0\.", r"edge_heads_\1.Dense_0."),
        (r"edge_heads_(\d+)\.2\.", r"edge_heads_\1.Dense_1."),
    ):
        new_key = re.sub(regex, sub, new_key)

    return new_key, np_value


def _scatter_species_embeddings(flat, atomic_types, n_rows):
    """Re-index embedding rows to atomic number: trained row ``i`` -> row
    ``Z = atomic_types[i]``. Untrained rows stay zero."""
    rows = np.asarray([int(z) for z in atomic_types])
    for key in list(flat):
        if key.endswith(".embedding"):
            table = np.asarray(flat[key])
            scattered = np.zeros((n_rows, table.shape[1]), dtype=table.dtype)
            scattered[rows] = table
            flat[key] = jnp.array(scattered)


def _unflatten(flat):
    """Dotted-key flat dict -> nested dict under ``{"params": ...}``."""
    nested = {}
    for key, value in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return {"params": nested}
