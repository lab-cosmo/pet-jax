"""UPET: Clean JAX/Flax reimplementation of the PET (Point-Edge Transformer) model."""

import jax
import jax.numpy as jnp

from collections.abc import Callable

import flax.linen as nn
from flax.linen import attention as flax_attention

from typing import NamedTuple

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
    max_atomic_number: int = 118
    system_conditioning: bool = False
    max_charge: int = 10
    max_spin_multiplicity: int = 10
    direct_forces: bool = False
    direct_stress: bool = False
    # Token layout and attention tiling (``express.Express``); see ``Backbone``.
    tokens: str = "padded"
    attention_tile: int | None = None
    fold_extra_block: bool = False
    reverse_unique: bool = False

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
        tile_tokens=None,
        extra_atoms=None,
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
            system_conditioning=self.system_conditioning,
            max_charge=self.max_charge,
            max_spin_multiplicity=self.max_spin_multiplicity,
            tokens=self.tokens,
            attention_tile=self.attention_tile,
            fold_extra_block=self.fold_extra_block,
            reverse_unique=self.reverse_unique,
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
            tile_tokens=tile_tokens,
            extra_atoms=extra_atoms,
        )
        # The readouts sum each atom's edges: a reshape on the padded layout,
        # a sorted segment sum by center on the flat one.
        segments = centers if self.tokens == "flat" else None

        predictions = Energy(d_head=self.d_head, name="energy_head")(
            node, edge, cutoffs, pair_mask, atom_mask, segments
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
                node, edge, cutoffs, pair_mask, atom_mask, segments
            )
            force_scale = self.param(
                "force_scale", nn.initializers.ones, (self.max_atomic_number + 1,)
            )
            out["forces"] = (
                forces * force_scale[atomic_numbers][:, None] * atom_mask[:, None]
            )
        if self.direct_stress:
            stress = DirectStress(d_head=self.d_head, name="stress_head")(
                node, edge, cutoffs, pair_mask, atom_mask, segments
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

    Two hooks let architectures built on the PET backbone (FlashMD) reuse it.
    With ``species_embedding=False`` the backbone owns no species embedding
    table and the caller passes the initial node features ``[N, d_node]`` as
    ``node_features`` (e.g. FlashMD's species + momenta encoder); with the
    table present, ``node_features`` overrides it for that call. This is a
    construction-time switch rather than an input check so that the parameter
    tree is fixed by the config alone, not by what ``init`` happened to be
    called with. ``cutoff_function(r, cutoff, width)`` gives the per-pair
    cutoff factor (``cutoff_bump`` for PET-MAD, ``cutoff_cosine`` for FlashMD).

    Token layout. ``tokens="padded"`` (stock) takes the pairs in the
    rectangular ``[N * n]`` layout. ``tokens="flat"`` takes one compacted,
    center-sorted token list ``[P]`` of any static length plus the attention
    tile maps ``tile_tokens`` / ``extra_atoms`` (``select._token_tiles``):
    every per-token layer and the reverse gather run on the flat list, and only
    the attention core gathers tiles of width ``W = tile_tokens.shape[1] + 1``
    (centre in slot 0). Without ``attention_tile`` that is one tile of
    ``k_sel + 1``; with it, an atom spilling past ``W - 1`` neighbours has one
    extra tile and its softmax runs across both (``_attention_tiles``).
    Either way the result is the padded layout's to
    summation order, with the same parameter tree.
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
    system_conditioning: bool = False
    max_charge: int = 10
    max_spin_multiplicity: int = 10
    final_message_passing: bool = True
    cutoff_function: Callable = cutoff_bump
    species_embedding: bool = True
    tokens: str = "padded"
    attention_tile: int | None = None
    fold_extra_block: bool = False
    reverse_unique: bool = False

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
        tile_tokens=None,
        extra_atoms=None,
    ):
        d_pet = self.d_pet
        d_node = self.d_node
        P = R_ij.shape[0]
        N = atomic_numbers.shape[0]
        n = P // N
        flat = self.tokens == "flat"
        assert flat == (tile_tokens is not None), "tokens='flat' needs the tile maps"
        if self.attention_tile is not None:
            assert flat and tile_tokens.shape[1] + 1 == self.attention_tile

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
        if flat:
            tiles = token_tiles(tile_tokens, extra_atoms, atom_mask, P)
            cutoffs_tokens = tiles.gather(jnp.concatenate([central[:, 0], cutoffs]))
            mask = tiles.mask
        else:
            tiles = None
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

            edge = tokens_flat if flat else tokens_flat.reshape(N, n, d_pet)

            # Transformer layers
            for attn_idx in range(self.num_attention_layers):
                node, edge = TransformerLayer(
                    d_pet=d_pet,
                    d_node=d_node,
                    d_ff=self.d_feedforward,
                    num_heads=self.num_heads,
                    temperature=self.attention_temperature,
                    name=f"gnn_layers_{layer_idx}_trans_layers_{attn_idx}",
                )(node, edge, cutoffs_tokens, mask, tiles)

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
    ``segments`` (the centers) switches the per-atom edge sum from the
    rectangular reshape to a sorted segment sum, for the flat token layout.
    """

    d_head: int = 128

    @nn.compact
    def __call__(self, node, edge, cutoffs, pair_mask, atom_mask, segments=None):
        N = node.shape[0]

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
        edge_contrib = per_atom_sum(edge_out * cutoffs, N, segments)
        return node_contrib + edge_contrib


class DirectForces(nn.Module):
    """Non-conservative force readout: node + cutoff-weighted edge
    contributions -> per-atom ``[N, 3]``. Scale applied by ``UPET``."""

    d_head: int = 128

    @nn.compact
    def __call__(self, node, edge, cutoffs, pair_mask, atom_mask, segments=None):
        N = node.shape[0]

        node_h = masked(
            MLP2(self.d_head, self.d_head, name="node_heads_0"), node, atom_mask
        )
        node_out = masked(nn.Dense(3, name="node_last_0"), node_h, atom_mask)

        edge_h = masked(
            MLP2(self.d_head, self.d_head, name="edge_heads_0"), edge, pair_mask
        )
        edge_out = masked(nn.Dense(3, name="edge_last_0"), edge_h, pair_mask)

        edge_contrib = per_atom_sum(edge_out * cutoffs[:, None], N, segments)
        return node_out + edge_contrib


class DirectStress(nn.Module):
    """Non-conservative stress readout: per-atom symmetrized ``[N, 3, 3]``.
    Scale, sum over atoms, and volume division happen downstream."""

    d_head: int = 128

    @nn.compact
    def __call__(self, node, edge, cutoffs, pair_mask, atom_mask, segments=None):
        N = node.shape[0]

        node_h = masked(
            MLP2(self.d_head, self.d_head, name="node_heads_0"), node, atom_mask
        )
        node_out = masked(nn.Dense(9, name="node_last_0"), node_h, atom_mask)

        edge_h = masked(
            MLP2(self.d_head, self.d_head, name="edge_heads_0"), edge, pair_mask
        )
        edge_out = masked(nn.Dense(9, name="edge_last_0"), edge_h, pair_mask)

        edge_contrib = per_atom_sum(edge_out * cutoffs[:, None], N, segments)
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

    @nn.compact
    def __call__(self, node, edge, cutoffs, mask, tiles=None):
        """``edge`` is ``[N, n, d_pet]``, or the flat ``[P, d_pet]`` with
        ``tiles`` (``TokenTiles``); then the centre and edge tokens run as one
        ``[N + P]`` list through the same row-wise layers."""
        d_pet, d_node = self.d_pet, self.d_node
        d_ff = self.d_ff
        expanded = d_node != d_pet

        if expanded:
            proj = nn.Dense(d_pet, name="center_contract")(node)
        else:
            proj = node
        if tiles is None:
            tokens = jnp.concatenate([proj, edge], axis=1)
        else:
            tokens = jnp.concatenate([proj[:, 0], edge], axis=0)

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
            rms_norm(tokens, name="norm_attn"), cutoffs, mask, tiles
        )
        if tiles is None:
            out_node, out_edge = attn[:, :1], attn[:, 1:]
        else:
            N = node.shape[0]
            out_node, out_edge = attn[:N, None], attn[N:]

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

    ``x`` is ``[N, T, F]``, one row of tokens per atom; or, with ``tiles``,
    the flat token list ``[N + P, F]`` whose projections run unpadded and only
    the attention core sees the tiles (``cutoffs`` / ``mask`` then in the
    tile frame).
    """

    num_heads: int
    temperature: float = 1.0

    @nn.compact
    def __call__(self, x, cutoffs, mask, tiles=None):
        if tiles is not None:
            return self._tiled(x, cutoffs, mask, tiles)
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

    def _tiled(self, x, cutoffs, mask, tiles):
        _, F = x.shape
        H = self.num_heads
        d = F // H

        qkv = tiles.gather(nn.Dense(3 * F, name="qkv")(x))
        T, W, _ = qkv.shape
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(T, W, H, d) / self.temperature
        k = k.reshape(T, W, H, d)
        v = v.reshape(T, W, H, d)
        bias = jnp.log(jnp.clip(cutoffs, 1e-15, None))

        if tiles.extra_atoms.shape[0] == 0:
            # One tile per atom: the padded layout's attention at T = W.
            out = flax_attention.dot_product_attention(
                q, k, v, bias=bias[:, None, None, :], mask=mask[:, None, None, :]
            )
        else:
            out = _attention_tiles(q, k, v, bias, mask, tiles)

        # Back to the flat list before the output projection, which runs unpadded.
        return nn.Dense(F, name="out")(tiles.scatter(out.reshape(T, W, F)))


# -- flat tokens: attention tiles --


class TokenTiles(NamedTuple):
    """Index maps between the flat token list ``[N + P]`` (the ``N`` centre
    tokens, then the ``P`` edge tokens) and the attention tiles ``[T, W]``,
    ``T = N + E``: tile ``i < N`` is atom ``i``'s first, tile ``N + x`` the
    ``x``-th extra one. Every map has unique indices, empty entries being
    distinct out-of-range ones, so each gather's transpose is a scatter
    without atomics."""

    rows: jax.Array  # [T, W] token behind each tile slot
    slots: jax.Array  # [N + P] tile slot (in the flattened [T * W]) of each token
    mask: jax.Array  # [T, W] bool, the slot holds a real token
    extra_atoms: jax.Array  # [E] atom owning each extra tile
    extra_of: jax.Array  # [N] extra tile of each atom

    def gather(self, x):
        """``[N + P, ...]`` -> ``[T, W, ...]``, zero in the empty slots."""
        return x.at[self.rows].get(mode="fill", fill_value=0, unique_indices=True)

    def scatter(self, y):
        """``[T, W, ...]`` -> ``[N + P, ...]``, the inverse; zero for a token
        in no tile."""
        y = y.reshape((-1,) + y.shape[2:])
        return y.at[self.slots].get(mode="fill", fill_value=0, unique_indices=True)


def token_tiles(tile_tokens, extra_atoms, atom_mask, P):
    """``TokenTiles`` from the pack's maps (``select._token_tiles``): an entry
    ``>= P`` of ``tile_tokens`` is an empty slot, one ``>= N`` of
    ``extra_atoms`` an unused extra tile."""
    N = atom_mask.shape[0]
    T, w = tile_tokens.shape
    W = w + 1
    E = T - N
    idx = tile_tokens.dtype
    tile = jnp.arange(T, dtype=idx)

    centre = jnp.where(tile < N, tile, N + P)[:, None]
    rows = jnp.concatenate([centre, N + tile_tokens], axis=1)
    live = rows < N + P
    rows = jnp.where(live, rows, N + P + jnp.arange(T * W, dtype=idx).reshape(T, W))
    mask = live & jnp.concatenate(
        [jnp.pad(atom_mask, (0, E))[:, None], jnp.ones((T, w), bool)], axis=1
    )
    slots = jnp.arange(T * W, T * W + N + P, dtype=idx)
    slots = slots.at[rows.reshape(-1)].set(
        jnp.arange(T * W, dtype=idx), mode="drop", unique_indices=True
    )

    x = jnp.arange(E, dtype=idx)
    extra_atoms = jnp.where(extra_atoms < N, extra_atoms, T + x)
    extra_of = jnp.arange(E, E + N, dtype=idx)
    extra_of = extra_of.at[extra_atoms].set(x, mode="drop", unique_indices=True)
    return TokenTiles(rows, slots, mask, extra_atoms, extra_of)


def _attention_tiles(q, k, v, bias, mask, tiles):
    """Attention over tiles where an atom may own two, ``q, k, v`` ``[T, W, H, d]``
    (``q`` already over the temperature), ``bias`` / ``mask`` ``[T, W]`` of
    the key slots; returns ``[T, W, H, d]``.

    A query sees all of its atom's keys: the diagonal blocks (first tile
    against itself) are one rectangular pass over the first ``N`` tiles; an
    atom with an extra tile adds the blocks (first, extra), (extra, first) and
    (extra, extra), computed on the ``E`` extra tiles only.

    Each query row's softmax spans its blocks in two passes: block maxima,
    then with the final maximum ``m`` (``stop_gradient``, as in
    ``jax.nn.softmax``) ``out = sum exp(s - m) v / sum exp(s - m)``. Scores
    get the bias and the ``finfo.min`` mask as in ``flax``'s
    ``dot_product_attention``, so a fully masked row (padding) comes out
    uniform and finite like there, and the result is the one-tile softmax to
    summation order.
    """
    T, W, H, d = q.shape
    E = tiles.extra_atoms.shape[0]
    N = T - E
    q = q / jnp.sqrt(d).astype(q.dtype)
    big_neg = jnp.finfo(q.dtype).min

    def scores(qb, kb, bias_k, mask_k):
        s = jnp.einsum("...qhd,...khd->...hqk", qb, kb) + bias_k[..., None, None, :]
        return jnp.where(mask_k[..., None, None, :], s, big_neg)

    def first_of_extra(x, fill=0):
        # [T, ...] -> [E, ...]: the first tile of each extra tile's atom
        return x.at[tiles.extra_atoms].get(
            mode="fill", fill_value=fill, unique_indices=True
        )

    def extra_of_atom(x, fill):
        # [E, ...] -> [N, ...]: the extra tile of each atom
        return x.at[tiles.extra_of].get(mode="fill", fill_value=fill, unique_indices=True)

    def accumulate(s, m, vb):
        p = jnp.exp(s - m[..., None])
        return p.sum(axis=-1), jnp.einsum("...hqk,...khd->...qhd", p, vb)

    q_a, k_a, v_a = first_of_extra(q), first_of_extra(k), first_of_extra(v)
    bias_a, mask_a = first_of_extra(bias), first_of_extra(mask, False)
    q_x, k_x, v_x, bias_x, mask_x = q[N:], k[N:], v[N:], bias[N:], mask[N:]

    s_d = scores(q[:N], k[:N], bias[:N], mask[:N])  # [N, H, W, W]
    s_ax = scores(q_a, k_x, bias_x, mask_x)  # (first, extra), [E, H, W, W]
    s_xa = scores(q_x, k_a, bias_a, mask_a)  # (extra, first)
    s_xx = scores(q_x, k_x, bias_x, mask_x)  # (extra, extra)
    m_d, m_ax, m_xa = s_d.max(axis=-1), s_ax.max(axis=-1), s_xa.max(axis=-1)
    m_xx = s_xx.max(axis=-1)

    m_first = jax.lax.stop_gradient(jnp.maximum(m_d[:N], extra_of_atom(m_ax, big_neg)))
    m_extra = jax.lax.stop_gradient(jnp.maximum(m_xa, m_xx))

    l_d, o_d = accumulate(s_d, m_first, v[:N])
    l_ax, o_ax = accumulate(s_ax, first_of_extra(m_first), v_x)
    l_xa, o_xa = accumulate(s_xa, m_extra, v_a)
    l_xx, o_xx = accumulate(s_xx, m_extra, v_x)

    l = jnp.concatenate([l_d[:N] + extra_of_atom(l_ax, 0), l_xa + l_xx])  # [T, H, W]
    o = jnp.concatenate([o_d[:N] + extra_of_atom(o_ax, 0), o_xa + o_xx])  # [T, W, H, d]
    return o / jnp.swapaxes(l, 1, 2)[..., None]


def per_atom_sum(x, N, segments=None):
    """Sum per-pair ``x`` ``[P, ...]`` over each atom's pairs: a reshape to
    ``[N, P // N]`` on the padded layout, or with ``segments`` (the centers,
    sorted) a segment sum on the flat one."""
    if segments is None:
        return x.reshape((N, -1) + x.shape[1:]).sum(axis=1)
    return jax.ops.segment_sum(x, segments, num_segments=N, indices_are_sorted=True)


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
