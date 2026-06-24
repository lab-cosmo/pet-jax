"""Convert metatrain PET ``.ckpt`` files to pet-jax's Flax msgpack layout.

Reads the Lightning-style ``.ckpt`` directly — no ``mtt export`` / TorchScript
intermediate. Expects the LLPR-wrapped PET-MAD v1.5.0 format as published on
Hugging Face (``lab-cosmo/upet``):

    outer:  architecture_name="llpr",  model_ckpt_version=3
    inner:  architecture_name="pet",   model_ckpt_version=11

Other versions fail hard — use ``mtt upgrade`` or fetch a newer release.

Writes:
    {output_dir}/model.msgpack    -- Flax parameter tree
    {output_dir}/metadata.yaml    -- config, energy_scale, shifts, species_to_index
"""

import numpy as np
import jax.numpy as jnp

import io
import re
import zipfile
from pathlib import Path

from marathon.io import write_msgpack, write_yaml

# -- expected checkpoint versions --

OUTER_ARCH = "llpr"
OUTER_VERSION = 3
INNER_ARCH = "pet"
INNER_VERSION = 11

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
    "attention_temperature",
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
    _check_single_readout(pet_ckpt["best_model_state_dict"])
    meta = _extract_metadata(pet_ckpt)

    params = _unflatten(_convert_state_dict(pet_ckpt["best_model_state_dict"]))

    write_msgpack(output_dir / "model.msgpack", params)
    metadata = {
        "config": meta["config"],
        "energy_scale": meta["energy_scale"],
        "shifts": meta["shifts"],
        "species_to_index": meta["species_to_index"],
    }
    write_yaml(output_dir / "metadata.yaml", metadata)

    print(f"Saved to {output_dir}/")
    print(f"  config: {meta['config']}")
    print(f"  energy_scale: {meta['energy_scale']:.6f}")
    print(f"  num atomic types: {len(meta['species_to_index'])}")
    return params, metadata


def load_checkpoint(checkpoint_dir):
    """Load a pet-jax checkpoint (``model.msgpack`` + ``metadata.yaml``)."""
    from marathon.io import read_msgpack, read_yaml

    checkpoint_dir = Path(checkpoint_dir)
    params = read_msgpack(checkpoint_dir / "model.msgpack")
    metadata = read_yaml(checkpoint_dir / "metadata.yaml")
    return params, metadata


# -- ckpt unwrap + metadata extraction --


def _unwrap_pet_checkpoint(ckpt):
    """Navigate the LLPR wrapper, validate versions, return the inner PET dict."""
    outer_arch = ckpt.get("architecture_name")
    outer_ver = ckpt.get("model_ckpt_version")
    if outer_arch != OUTER_ARCH or outer_ver != OUTER_VERSION:
        raise ValueError(
            f"pet-jax expects LLPR-wrapped PET-MAD v1.5.0 checkpoints "
            f"({OUTER_ARCH!r} v{OUTER_VERSION}); got {outer_arch!r} v{outer_ver}. "
            f"Run `uv run --with metatrain mtt upgrade` on the .ckpt or fetch a "
            f"compatible release."
        )

    inner = ckpt.get("wrapped_model_checkpoint")
    if not isinstance(inner, dict):
        raise ValueError(
            "missing 'wrapped_model_checkpoint' in outer ckpt — the LLPR wrapper "
            "is expected to carry the PET model nested inside."
        )

    inner_arch = inner.get("architecture_name")
    inner_ver = inner.get("model_ckpt_version")
    if inner_arch != INNER_ARCH or inner_ver != INNER_VERSION:
        raise ValueError(
            f"pet-jax expects inner {INNER_ARCH!r} v{INNER_VERSION}; got "
            f"{inner_arch!r} v{inner_ver}. Run `mtt upgrade` on the source ckpt."
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

    config = {k: hypers[k] for k in CONFIG_KEYS}

    atomic_types = list(model_data["dataset_info"].atomic_types)
    species_to_index = {int(z): i for i, z in enumerate(atomic_types)}
    config["num_species"] = len(atomic_types)

    state_dict = pet_ckpt["best_model_state_dict"]

    scaler = parse_metatensor_buffer(state_dict["scaler.energy_scaler_buffer"])
    energy_scale = float(scaler["blocks/0/values.npy"].item())

    comp = parse_metatensor_buffer(
        state_dict["additive_models.0.energy_composition_buffer"]
    )
    comp_samples = comp["blocks/0/samples.npy"]
    comp_values = comp["blocks/0/values.npy"].flatten()
    shifts = {int(s[0]): float(v) for s, v in zip(comp_samples, comp_values)}

    return {
        "config": config,
        "atomic_types": atomic_types,
        "species_to_index": species_to_index,
        "energy_scale": energy_scale,
        "shifts": shifts,
    }


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


def _convert_state_dict(state_dict):
    """Raw PyTorch state dict -> flat Flax param dict."""
    import torch

    out = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith(_SKIP_PREFIXES) or "non_conservative" in key:
            continue

        new_key = _rename_key(key)
        np_value = value.cpu().numpy()
        new_key, np_value = _finalize_key(new_key, np_value)
        out[new_key] = jnp.array(np_value)
    return out


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
        for prefix, replacement in (
            ("node_heads.energy.", "node_heads_"),
            ("edge_heads.energy.", "edge_heads_"),
        ):
            if new_key.startswith(prefix):
                parts = new_key.split(".")
                new_key = f"{replacement}{parts[2]}.{'.'.join(parts[3:])}"
                break
        for prefix, replacement in (
            ("node_last_layers.energy.", "node_last_"),
            ("edge_last_layers.energy.", "edge_last_"),
        ):
            if new_key.startswith(prefix):
                parts = new_key.split(".")
                new_key = f"{replacement}{parts[2]}.{'.'.join(parts[4:])}"
                break

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
