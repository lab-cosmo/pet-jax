"""UPET: Clean JAX/Flax reimplementation of the PET (Point-Edge Transformer) model."""

import jax
import jax.numpy as jnp

from collections.abc import Callable

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
    # Taper width inside the adaptive neighbor selection — distinct from the
    # final-taper cutoff_width since metatrain ckpt v14. Checkpoint configs
    # predating the split are filled in by load_checkpoint (adaptive :=
    # cutoff_width, upstream's own migration rule).
    cutoff_width_adaptive: float = 0.5
    # "grid" (probe grid + Gaussian weights) or "solver" (Newton-bisection
    # root find) — metatrain ckpt v12 splits these; pre-split checkpoints
    # trained with grid.
    adaptive_cutoff_method: str = "grid"
    num_neighbors_adaptive: int = 8
    attention_temperature: float = 1.0
    attention_cutoff: str = "bias"
    attention_gate_bias: float = 0.0
    max_atomic_number: int = 118
    system_conditioning: bool = False
    max_charge: int = 10
    max_spin_multiplicity: int = 10
    direct_forces: bool = False
    direct_stress: bool = False

    @nn.compact
    def __call__(
        self,
        R_ij,
        centers,
        neighbors,
        atomic_numbers,
        reverse,
        pair_mask,
        atom_mask,
        pair_cutoffs=None,
        charge=None,
        spin_multiplicity=None,
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
            max_atomic_number=self.max_atomic_number,
            attention_temperature=self.attention_temperature,
            attention_cutoff=self.attention_cutoff,
            attention_gate_bias=self.attention_gate_bias,
            system_conditioning=self.system_conditioning,
            max_charge=self.max_charge,
            max_spin_multiplicity=self.max_spin_multiplicity,
            name="backbone",
        )(
            R_ij,
            centers,
            neighbors,
            atomic_numbers,
            reverse,
            pair_mask,
            atom_mask,
            pair_cutoffs,
            charge,
            spin_multiplicity,
        )

        predictions = Energy(d_head=self.d_head, name="energy_head")(
            node, edge, cutoffs, pair_mask, atom_mask
        )

        energy_scale = self.param("energy_scale", nn.initializers.ones, (1,))
        energy = predictions * atom_mask * energy_scale

        if not (self.direct_forces or self.direct_stress):
            return energy

        # Non-conservative heads: raw per-atom force/stress, scaled by their
        # loaded scales (forces per-species by Z; stress scalar).
        out = {"energy": energy}
        if self.direct_forces:
            forces = DirectForces(d_head=self.d_head, name="forces_head")(
                node, edge, cutoffs, pair_mask, atom_mask
            )
            force_scale = self.param(
                "force_scale", nn.initializers.ones, (self.max_atomic_number + 1,)
            )
            out["forces"] = (
                forces * force_scale[atomic_numbers][:, None] * atom_mask[:, None]
            )
        if self.direct_stress:
            stress = DirectStress(d_head=self.d_head, name="stress_head")(
                node, edge, cutoffs, pair_mask, atom_mask
            )
            stress_scale = self.param("stress_scale", nn.initializers.ones, (1,))
            out["stress"] = stress * stress_scale * atom_mask[:, None, None]
        return out


class Backbone(nn.Module):
    """PET core: embeddings + GNN/transformer layers -> (node, edge, cutoffs).

    Returns per-atom node features `[N, d_node]`, per-pair edge features
    (messages) `[P, d_pet]`, and per-pair cutoff factors `[P]`.

    With ``system_conditioning``, ``charge`` and ``spin_multiplicity`` are
    per-atom integer-valued arrays ``[N]``, int or float (every atom carries
    its structure's value, so a batch of several structures needs no extra
    index); absent inputs fall back to charge 0 / multiplicity 1, as upstream
    does for systems without the data.

    ``final_message_passing=False`` skips the reverse-edge message passing
    after the last GNN layer. The returned edge messages then reach
    ``num_gnn_layers`` × cutoff from their center, like the node features,
    instead of one cutoff further (with one layer, strictly local). Message
    passing between earlier layers is unchanged, and the node features are not
    affected. A training-time option: metatrain's PET has no equivalent,
    so such a model cannot be exported. Its parameter tree lacks the last
    layer's ``comb_norms_*`` / ``comb_mlps_*``; loading a checkpoint that has
    them leaves those entries silently unused.

    ``attention_cutoff`` picks how the cutoff factors enter the attention and
    ``attention_gate_bias`` is the score offset of its sigmoid modes, see
    `Attention`. No mode adds parameters, so one checkpoint loads under all of
    them. Only "bias" and "smooth" exist in metatrain.

    Two hooks let architectures built on the PET backbone (FlashMD) reuse it.
    With ``species_embedding=False`` the backbone owns no species embedding
    table and the caller passes the initial node features ``[N, d_node]`` as
    ``node_features`` (e.g. FlashMD's species + momenta encoder); with the
    table present, ``node_features`` overrides it for that call. This is a
    construction-time switch rather than an input check so that the parameter
    tree is fixed by the config alone, not by what ``init`` happened to be
    called with. ``cutoff_function(r, cutoff, width)`` gives the per-pair
    cutoff factor (``cutoff_bump`` for PET-MAD, ``cutoff_cosine`` for FlashMD).
    """

    d_pet: int = 128
    d_node: int = 512
    d_feedforward: int = 256
    num_heads: int = 8
    num_attention_layers: int = 1
    num_gnn_layers: int = 2
    cutoff: float = 7.5
    cutoff_width: float = 0.5
    max_atomic_number: int = 118
    attention_temperature: float = 1.0
    attention_cutoff: str = "bias"
    attention_gate_bias: float = 0.0
    system_conditioning: bool = False
    max_charge: int = 10
    max_spin_multiplicity: int = 10
    final_message_passing: bool = True
    cutoff_function: Callable = cutoff_bump
    species_embedding: bool = True

    @nn.compact
    def __call__(
        self,
        R_ij,
        centers,
        neighbors,
        atomic_numbers,
        reverse,
        pair_mask,
        atom_mask,
        pair_cutoffs=None,
        charge=None,
        spin_multiplicity=None,
        node_features=None,
    ):
        d_pet = self.d_pet
        d_node = self.d_node
        P = R_ij.shape[0]
        N = atomic_numbers.shape[0]
        n = P // N

        # Computed once, added to the node features after every GNN layer
        # (upstream adds it to the layer's output node embeddings).
        conditioning = None
        if self.system_conditioning:
            if charge is None:
                charge = jnp.zeros(N, dtype=jnp.int32)
            if spin_multiplicity is None:
                spin_multiplicity = jnp.ones(N, dtype=jnp.int32)
            conditioning = SystemConditioning(
                d_out=d_node,
                max_charge=self.max_charge,
                max_spin_multiplicity=self.max_spin_multiplicity,
                name="system_conditioning",
            )(charge, spin_multiplicity)
            conditioning = conditioning * atom_mask[:, None]

        r_ij = safe_norm(R_ij, axis=-1)

        # Per-edge cutoff factor (used at readout, and at attention after
        # prepending the central-token slot)
        cutoff = pair_cutoffs if pair_cutoffs is not None else self.cutoff
        cutoffs = self.cutoff_function(r_ij, cutoff, self.cutoff_width) * pair_mask

        # Token-level cutoffs + mask [N, 1+n], node slot prepended. Central
        # cutoff is atom_mask itself (1 for real atoms, 0 for padded).
        central = atom_mask[:, None].astype(cutoffs.dtype)
        cutoffs_tokens = jnp.concatenate([central, cutoffs.reshape(N, n)], axis=1)
        mask = jnp.concatenate([atom_mask[:, None], pair_mask.reshape(N, n)], axis=1)

        # Initial edge features; species embeddings index by atomic number Z
        # (table size max_atomic_number + 1, row 0 unused).
        edge_embed = nn.Embed(self.max_atomic_number + 1, d_pet, name="edge_embedder")
        messages = edge_embed(atomic_numbers)[neighbors] * pair_mask[..., None]

        # Node embedding
        if self.species_embedding:
            embedded = nn.Embed(self.max_atomic_number + 1, d_node, name="node_embedders_0")
            node_features = (
                embedded(atomic_numbers) if node_features is None else node_features
            )
        assert node_features is not None, (
            "species_embedding=False needs node_features [N, d_node]"
        )
        node = node_features[:, None, :] * atom_mask[:, None, None]

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
                    self.max_atomic_number + 1,
                    d_pet,
                    name=f"gnn_layers_{layer_idx}_neighbor_embed",
                )
                neighbor_feats = (
                    neighbor_embed(atomic_numbers)[neighbors] * pair_mask[..., None]
                )
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
                    attention_cutoff=self.attention_cutoff,
                    gate_bias=self.attention_gate_bias,
                    name=f"gnn_layers_{layer_idx}_trans_layers_{attn_idx}",
                )(node, edge, cutoffs_tokens, mask)

            if conditioning is not None:
                node = node + conditioning[:, None, :]

            # Message passing (feedforward mixing)
            edge_flat = edge.reshape(P, d_pet) * pair_mask[..., None]
            messages = messages + edge_flat
            if layer_idx < self.num_gnn_layers - 1 or self.final_message_passing:
                reversed_flat = edge_flat[reverse]
                combined = jnp.concatenate([edge_flat, reversed_flat], axis=-1)
                combined = nn.LayerNorm(epsilon=1e-5, name=f"comb_norms_{layer_idx}")(
                    combined
                )
                combined = MLP((2 * d_pet, d_pet), name=f"comb_mlps_{layer_idx}")(combined)
                messages = messages + combined * pair_mask[..., None]

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


class DirectForces(nn.Module):
    """Non-conservative force readout: node + cutoff-weighted edge
    contributions -> per-atom ``[N, 3]``. Scale applied by ``UPET``."""

    d_head: int = 128

    @nn.compact
    def __call__(self, node, edge, cutoffs, pair_mask, atom_mask):
        N = node.shape[0]
        n = edge.shape[0] // N

        node_h = masked(
            MLP2(self.d_head, self.d_head, name="node_heads_0"), node, atom_mask
        )
        node_out = masked(nn.Dense(3, name="node_last_0"), node_h, atom_mask)

        edge_h = masked(
            MLP2(self.d_head, self.d_head, name="edge_heads_0"), edge, pair_mask
        )
        edge_out = masked(nn.Dense(3, name="edge_last_0"), edge_h, pair_mask)

        edge_contrib = (edge_out * cutoffs[:, None]).reshape(N, n, 3).sum(axis=1)
        return node_out + edge_contrib


class DirectStress(nn.Module):
    """Non-conservative stress readout: per-atom symmetrized ``[N, 3, 3]``.
    Scale, sum over atoms, and volume division happen downstream."""

    d_head: int = 128

    @nn.compact
    def __call__(self, node, edge, cutoffs, pair_mask, atom_mask):
        N = node.shape[0]
        n = edge.shape[0] // N

        node_h = masked(
            MLP2(self.d_head, self.d_head, name="node_heads_0"), node, atom_mask
        )
        node_out = masked(nn.Dense(9, name="node_last_0"), node_h, atom_mask)

        edge_h = masked(
            MLP2(self.d_head, self.d_head, name="edge_heads_0"), edge, pair_mask
        )
        edge_out = masked(nn.Dense(9, name="edge_last_0"), edge_h, pair_mask)

        edge_contrib = (edge_out * cutoffs[:, None]).reshape(N, n, 9).sum(axis=1)
        per_atom = (node_out + edge_contrib).reshape(N, 3, 3)
        return 0.5 * (per_atom + jnp.swapaxes(per_atom, 1, 2))


# -- system conditioning --


class SystemConditioning(nn.Module):
    """Charge / spin-multiplicity embedding, one vector per atom ``[N, d_out]``.

    Two lookup tables, concatenated and projected; upstream's
    ``SystemConditioningEmbedding``. Both are indexed by the value itself,
    like the species tables by Z: multiplicity ``m`` is row ``m`` (row 0
    unused), charge ``c`` is row ``c``, a negative charge reaching its row
    from the end of the table the way any negative index does. The
    converter lays the tables out that way; upstream stores them offset.
    """

    d_out: int
    max_charge: int = 10
    max_spin_multiplicity: int = 10

    @nn.compact
    def __call__(self, charge, spin_multiplicity):
        charge_embed = nn.Embed(
            2 * self.max_charge + 1, self.d_out, name="charge_embedding"
        )
        spin_embed = nn.Embed(
            self.max_spin_multiplicity + 1, self.d_out, name="spin_multiplicity_embedding"
        )
        c = charge_embed(charge.astype(jnp.int32))
        s = spin_embed(spin_multiplicity.astype(jnp.int32))
        return MLP((self.d_out, self.d_out), name="project")(
            jnp.concatenate([c, s], axis=-1)
        )


# -- transformer block --


class TransformerLayer(nn.Module):
    """Transformer layer with PreLN, RMSNorm, and SwiGLU."""

    d_pet: int
    d_node: int
    d_ff: int
    num_heads: int
    temperature: float = 1.0
    attention_cutoff: str = "bias"
    gate_bias: float = 0.0

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
        attn = Attention(
            self.num_heads, self.temperature, self.attention_cutoff, self.gate_bias
        )(rms_norm(tokens, name="norm_attn"), cutoffs, mask)
        out_node, out_edge = attn[:, :1], attn[:, 1:]

        if expanded:
            out_node = node + nn.Dense(d_node, name="center_expand")(out_node)
            out_node = out_node + center_mlp(rms_norm(out_node, name="norm_center"))
        else:
            out_node = node + out_node

        out_edge = edge + out_edge
        out_edge = out_edge + edge_mlp(rms_norm(out_edge, name="norm_mlp"))

        return out_node, out_edge


class Attention(nn.Module):
    """Multi-head scaled dot-product attention with cutoff weighting.

    A neighbor's cutoff factor ``f`` in [0, 1] has to remove it from the
    attention continuously as it leaves the cutoff. ``cutoff`` picks how,
    for query ``q``, key ``k`` and score ``s_qk``:

    - "bias": softmax weights proportional to ``f_k exp(s_qk)``, i.e. ``log f``
      added to the scores, for every query including the key's own token.
      The released models. A key that the scores prefer by a factor ``A``
      over the others together receives ``f A / (1 + f A)`` of the
      attention, so it switches on where ``f`` is about ``1 / A``, over a
      distance set by ``log f``, which diverges at the cutoff.
    - "self_exempt": as "bias", but a token attends to itself at full weight.
      The features of a token then no longer depend on its own ``f``; with
      "bias" they do, and since everything the token feeds is weighted by
      ``f`` again, the energy picks up powers of ``f``.
    - "smooth": "self_exempt", plus `smooth_cutoff_weights` for the central
      token's row. metatrain's ``smooth_attention_cutoff``.
    - "sigmoid": no normalisation over keys, the weights are
      ``f_k sigmoid(s_qk + gate_bias)`` with ``f = 1`` on the diagonal. Keys
      do not compete, so the output is linear in every ``f_k`` with a slope
      the scores cannot increase. The output grows with the number of keys;
      ``gate_bias`` near ``-log`` of that number keeps it at the softmax's
      scale at initialisation.
    - "sigmoid_mean": "sigmoid" divided by the sum of the ``f_k`` of the row,
      a normalisation that does not depend on the scores.
    """

    num_heads: int
    temperature: float = 1.0
    cutoff: str = "bias"
    gate_bias: float = 0.0

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

        attn_mask = mask.reshape(N, 1, 1, T)

        if self.cutoff == "bias":
            factors = cutoffs.reshape(N, 1, 1, T)
        elif self.cutoff in ("self_exempt", "smooth", "sigmoid", "sigmoid_mean"):
            # [N, 1, query, key]; padded tokens keep their zero factor
            is_self = jnp.eye(T, dtype=bool) & (mask[:, :, None] > 0)
            factors = jnp.where(is_self, 1.0, cutoffs[:, None, :])[:, None]
        else:
            raise ValueError(f"unknown attention cutoff {self.cutoff!r}")

        if self.cutoff in ("bias", "self_exempt"):
            out = _softmax_attention(q, k, v, factors, attn_mask)
        elif self.cutoff == "smooth":
            central_scores = _scores(q[:, :1], k)
            weights = smooth_cutoff_weights(central_scores, factors[:, :, :1])
            central = jnp.einsum("nhqk,nkhd->nqhd", weights, v)
            others = _softmax_attention(q[:, 1:], k, v, factors[:, :, 1:], attn_mask)
            out = jnp.concatenate([central, others], axis=1)
        else:
            weights = factors * jax.nn.sigmoid(_scores(q, k) + self.gate_bias)
            if self.cutoff == "sigmoid_mean":
                # at least 1 for a real query, which counts itself
                total = jnp.sum(factors, axis=-1, keepdims=True)
                weights = weights / jnp.maximum(total, 1.0)
            out = jnp.einsum("nhqk,nkhd->nqhd", weights, v)

        return nn.Dense(F, name="out")(out.reshape(N, T, F))


def smooth_cutoff_weights(scores, factors, established_power=8.0):
    """Attention weights that grow in proportion to the keys' cutoff factors.

    metatrain's ``smooth_cutoff_attention``. Before normalisation over the
    keys, a key weighs

        f exp(score) / (1 + (1 - f) exp(score) / established)

    where ``established`` is the summed weight of the other keys, each
    counted with ``f**established_power``, i.e. once it is well inside the
    cutoff. With a single key in the cutoff tail, that key receives exactly
    ``f`` times the attention it would receive at ``f = 1``. Keys at
    ``f = 1`` keep their softmax weight.

    ``scores`` is ``[..., queries, keys]`` and ``factors`` broadcasts to it;
    keys with a zero factor are excluded.
    """
    # a finite fill, so that rows without any key (padded atoms) stay finite
    scores = jnp.where(factors > 0, scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)

    # Summed counted weight of the keys other than each one. Subtracting each
    # term from the total would cancel for a key that dominates it, so start
    # from the sum without the largest term.
    counted = weights * factors**established_power
    index = jnp.argmax(counted, axis=-1, keepdims=True)
    largest = jnp.take_along_axis(counted, index, axis=-1)
    is_largest = jnp.arange(counted.shape[-1]) == index
    rest = jnp.sum(jnp.where(is_largest, 0.0, counted), axis=-1, keepdims=True)
    established = jnp.maximum(rest + (largest - counted), 1e-12)

    weights = factors * weights * established / (established + (1.0 - factors) * weights)
    total = jnp.sum(weights, axis=-1, keepdims=True)
    return weights / jnp.where(total > 0, total, 1.0)


def _scores(q, k):
    """Attention scores ``[N, H, queries, keys]`` from ``[N, tokens, H, d]``."""
    return jnp.einsum("nqhd,nkhd->nhqk", q, k) / jnp.sqrt(q.shape[-1]).astype(q.dtype)


def _softmax_attention(q, k, v, factors, attn_mask):
    # cutoff * exp(qk) = exp(qk + log(cutoff)).
    bias = jnp.log(jnp.clip(factors, 1e-15, None))
    # jax.nn hardcodes softmax to fp32 + masks with -0.7 * finfo(fp64).max,
    # which overflows to -inf in fp32 -> NaN softmax on fully-masked rows
    return flax_attention.dot_product_attention(q, k, v, bias=bias, mask=attn_mask)


# -- building blocks --


class MLP(nn.Module):
    """N-layer MLP: SiLU after each hidden layer, no activation on the last."""

    features: tuple

    @nn.compact
    def __call__(self, x):
        for f in self.features[:-1]:
            x = nn.Dense(f)(x)
            x = jax.nn.silu(x)
        x = nn.Dense(self.features[-1])(x)
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


def rms_norm(x, name):
    """RMSNorm with torch's default epsilon, ``finfo(dtype).eps`` (flax: 1e-6)."""
    return nn.RMSNorm(epsilon=float(jnp.finfo(x.dtype).eps), name=name)(x)


def masked(fn, x, mask, fn_value=0.0, return_value=0.0):
    """Apply fn(x) where mask is True, otherwise return return_value."""
    if len(x.shape) == 1:
        m = mask
    else:
        m = mask[..., None]
    fn_value = jnp.array(fn_value, dtype=x.dtype)
    return_value = jnp.array(return_value, dtype=x.dtype)
    return jnp.where(m, fn(jnp.where(m, x, fn_value)), return_value)
