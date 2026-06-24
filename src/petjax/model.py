"""UPET: Clean JAX/Flax reimplementation of the PET (Point-Edge Transformer) model."""

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.linen import attention as flax_attention

from .utils import cutoff_bump, safe_norm


class UPET(nn.Module):
    """UPET: Point-Edge Transformer for interatomic potentials.

    A `Backbone` (node/edge features) followed by an `Energy` (readout).
    Hardcoded architecture choices from PET-MAD v1.5.0+: RMSNorm, SwiGLU,
    PreLN, feedforward featurizer.
    """

    d_pet: int = 128
    d_node: int = 512
    d_head: int = 128
    d_feedforward: int = 256
    num_heads: int = 8
    num_attention_layers: int = 1
    num_gnn_layers: int = 2
    cutoff: float = 7.5
    cutoff_width: float = 0.5
    num_neighbors_adaptive: int = 8
    attention_temperature: float = 1.0
    num_species: int = 103
    energy_scale: float = 1.0

    def get_probes(self):
        return jnp.arange(0.5, self.cutoff, self.cutoff_width / 4)

    @nn.compact
    def __call__(
        self,
        R_ij,
        centers,
        neighbors,
        species,
        reverse,
        pair_mask,
        atom_mask,
        pair_cutoffs=None,
    ):
        node, edge, cutoffs = Backbone(
            d_pet=self.d_pet,
            d_node=self.d_node,
            d_feedforward=self.d_feedforward,
            num_heads=self.num_heads,
            num_attention_layers=self.num_attention_layers,
            num_gnn_layers=self.num_gnn_layers,
            cutoff=self.cutoff,
            cutoff_width=self.cutoff_width,
            num_species=self.num_species,
            attention_temperature=self.attention_temperature,
            name="backbone",
        )(R_ij, centers, neighbors, species, reverse, pair_mask, atom_mask, pair_cutoffs)

        predictions = Energy(d_head=self.d_head, name="energy_head")(
            node, edge, cutoffs, pair_mask, atom_mask
        )

        return predictions * atom_mask * self.energy_scale


class Backbone(nn.Module):
    """PET core: embeddings + GNN/transformer layers -> (node, edge, cutoffs).

    Returns per-atom node features `[N, d_node]`, per-pair edge features
    (messages) `[P, d_pet]`, and per-pair cutoff factors `[P]`.
    """

    d_pet: int = 128
    d_node: int = 512
    d_feedforward: int = 256
    num_heads: int = 8
    num_attention_layers: int = 1
    num_gnn_layers: int = 2
    cutoff: float = 7.5
    cutoff_width: float = 0.5
    num_species: int = 103
    attention_temperature: float = 1.0

    @nn.compact
    def __call__(
        self,
        R_ij,
        centers,
        neighbors,
        species,
        reverse,
        pair_mask,
        atom_mask,
        pair_cutoffs=None,
    ):
        d_pet = self.d_pet
        d_node = self.d_node
        P = R_ij.shape[0]
        N = species.shape[0]
        n = P // N

        r_ij = safe_norm(R_ij, axis=-1)

        # Per-edge cutoff factor (used at readout, and at attention after
        # prepending the central-token slot)
        cutoff = pair_cutoffs if pair_cutoffs is not None else self.cutoff
        cutoffs = cutoff_bump(r_ij, cutoff, self.cutoff_width) * pair_mask

        # Token-level cutoffs + mask [N, 1+n], node slot prepended. Central
        # cutoff is atom_mask itself (1 for real atoms, 0 for padded).
        central = atom_mask[:, None].astype(cutoffs.dtype)
        cutoffs_tokens = jnp.concatenate([central, cutoffs.reshape(N, n)], axis=1)
        mask = jnp.concatenate([atom_mask[:, None], pair_mask.reshape(N, n)], axis=1)

        # Initial edge features
        edge_embed = nn.Embed(self.num_species, d_pet, name="edge_embedder")
        messages = edge_embed(species)[neighbors] * pair_mask[..., None]

        # Node embedding (feedforward: persists across layers)
        node_embed = nn.Embed(self.num_species, d_node, name="node_embedders_0")
        node = node_embed(species)[:, None, :] * atom_mask[:, None, None]

        for layer_idx in range(self.num_gnn_layers):
            # Geometric features
            geom = jnp.concatenate([R_ij, r_ij[..., None]], axis=-1)
            geom_proj = masked(
                nn.Dense(d_pet, name=f"gnn_layers_{layer_idx}_edge_embed"),
                geom,
                pair_mask,
            )

            if layer_idx == 0:
                tokens_flat = masked(
                    MLP(
                        (d_pet, d_pet),
                        name=f"gnn_layers_{layer_idx}_compress",
                    ),
                    jnp.concatenate([geom_proj, messages], axis=-1),
                    pair_mask,
                )
            else:
                neighbor_embed = nn.Embed(
                    self.num_species,
                    d_pet,
                    name=f"gnn_layers_{layer_idx}_neighbor_embed",
                )
                neighbor_feats = neighbor_embed(species)[neighbors] * pair_mask[..., None]
                tokens_flat = masked(
                    MLP(
                        (d_pet, d_pet),
                        name=f"gnn_layers_{layer_idx}_compress",
                    ),
                    jnp.concatenate([geom_proj, neighbor_feats, messages], axis=-1),
                    pair_mask,
                )

            edge = tokens_flat.reshape(N, n, d_pet)

            # Transformer layers
            for attn_idx in range(self.num_attention_layers):
                node, edge = TransformerLayer(
                    d_pet=d_pet,
                    d_node=d_node,
                    d_ff=self.d_feedforward,
                    num_heads=self.num_heads,
                    temperature=self.attention_temperature,
                    name=f"gnn_layers_{layer_idx}_trans_layers_{attn_idx}",
                )(node, edge, cutoffs_tokens, mask)

            # Message passing (feedforward mixing)
            edge_flat = edge.reshape(P, d_pet)
            reversed_flat = edge_flat[reverse]
            combined = jnp.concatenate([edge_flat, reversed_flat], axis=-1)
            combined = nn.LayerNorm(name=f"comb_norms_{layer_idx}")(combined)
            combined = MLP((2 * d_pet, d_pet), name=f"comb_mlps_{layer_idx}")(combined)
            combined = combined * pair_mask[..., None]
            messages = messages + edge_flat + combined

        node = node[:, 0, :] * atom_mask[:, None]
        return node, messages, cutoffs


class Energy(nn.Module):
    """Energy readout: node + cutoff-weighted edge contributions -> per-atom E.

    Single readout of final features, following current uPET hypers.
    """

    d_head: int = 128

    @nn.compact
    def __call__(self, node, edge, cutoffs, pair_mask, atom_mask):
        N = node.shape[0]
        n = edge.shape[0] // N

        # Node head
        node_h = masked(
            MLP2(self.d_head, self.d_head, name="node_heads_0"), node, atom_mask
        )
        node_contrib = masked(nn.Dense(1, name="node_last_0"), node_h, atom_mask)[:, 0]

        # Edge head
        edge_h = masked(
            MLP2(self.d_head, self.d_head, name="edge_heads_0"), edge, pair_mask
        )
        edge_out = masked(nn.Dense(1, name="edge_last_0"), edge_h, pair_mask)[:, 0]

        # Weighted edge sum per atom
        edge_contrib = (edge_out * cutoffs).reshape(N, n).sum(axis=1)
        return node_contrib + edge_contrib


# -- transformer block --


class TransformerLayer(nn.Module):
    """Transformer layer with PreLN, RMSNorm, and SwiGLU."""

    d_pet: int
    d_node: int
    d_ff: int
    num_heads: int
    temperature: float = 1.0

    @nn.compact
    def __call__(self, node, edge, cutoffs, mask):
        d_pet, d_node = self.d_pet, self.d_node
        d_ff = self.d_ff
        expanded = d_node != d_pet

        if expanded:
            proj = nn.Dense(d_pet, name="center_contract")(node)
        else:
            proj = node
        tokens = jnp.concatenate([proj, edge], axis=1)

        def edge_mlp(x):
            h = nn.Dense(2 * d_ff, name="mlp_in")(x)
            v, g = jnp.split(h, 2, axis=-1)
            return nn.Dense(d_pet, name="mlp_out")(v * jax.nn.sigmoid(g))

        def center_mlp(x):
            d_center_ff = 2 * d_node
            h = nn.Dense(2 * d_center_ff, name="center_mlp_in")(x)
            v, g = jnp.split(h, 2, axis=-1)
            return nn.Dense(d_node, name="center_mlp_out")(v * jax.nn.sigmoid(g))

        # PreLN transformer
        attn = Attention(self.num_heads, self.temperature)(
            nn.RMSNorm(name="norm_attn")(tokens), cutoffs, mask
        )
        out_node, out_edge = attn[:, :1], attn[:, 1:]

        if expanded:
            out_node = node + nn.Dense(d_node, name="center_expand")(out_node)
            out_node = out_node + center_mlp(nn.RMSNorm(name="norm_center")(out_node))
        else:
            out_node = node + out_node

        out_edge = edge + out_edge
        out_edge = out_edge + edge_mlp(nn.RMSNorm(name="norm_mlp")(out_edge))

        return out_node, out_edge


class Attention(nn.Module):
    """Multi-head scaled dot-product attention with cutoff weighting."""

    num_heads: int
    temperature: float = 1.0

    @nn.compact
    def __call__(self, x, cutoffs, mask):
        N, T, F = x.shape
        H = self.num_heads
        d = F // H

        qkv = nn.Dense(3 * F, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # flax scales q by 1/sqrt(d); fold in temperature for the full
        # 1/(sqrt(d) * temperature) scale.
        q = q.reshape(N, T, H, d) / self.temperature
        k = k.reshape(N, T, H, d)
        v = v.reshape(N, T, H, d)

        # cutoff * exp(qk) = exp(qk + log(cutoff)).
        bias = jnp.log(jnp.clip(cutoffs, 1e-15, None)).reshape(N, 1, 1, T)
        attn_mask = mask.reshape(N, 1, 1, T)

        # jax.nn hardcodes softmax to fp32 + masks with -0.7 * finfo(fp64).max,
        # which overflows to -inf in fp32 -> NaN softmax on fully-masked rows
        out = flax_attention.dot_product_attention(q, k, v, bias=bias, mask=attn_mask)
        out = out.reshape(N, T, F)

        return nn.Dense(F, name="out")(out)


# -- building blocks --


class MLP(nn.Module):
    """2-layer MLP with SiLU activation."""

    features: tuple

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features[0])(x)
        x = jax.nn.silu(x)
        x = nn.Dense(self.features[1])(x)
        return x


class MLP2(nn.Module):
    """2-layer MLP with SiLU after each layer (readout heads)."""

    hidden: int
    output: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = jax.nn.silu(x)
        x = nn.Dense(self.output)(x)
        x = jax.nn.silu(x)
        return x


def masked(fn, x, mask, fn_value=0.0, return_value=0.0):
    """Apply fn(x) where mask is True, otherwise return return_value."""
    if len(x.shape) == 1:
        m = mask
    else:
        m = mask[..., None]
    fn_value = jnp.array(fn_value, dtype=x.dtype)
    return_value = jnp.array(return_value, dtype=x.dtype)
    return jnp.where(m, fn(jnp.where(m, x, fn_value)), return_value)
