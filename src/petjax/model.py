"""UPET: Clean JAX/Flax reimplementation of the PET (Point-Edge Transformer) model."""

import jax
import jax.numpy as jnp

import functools
from collections.abc import Callable

import flax.linen as nn
from flax.linen import attention as flax_attention

from typing import NamedTuple

from .structure import atom_blocks
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
    # Token layout, attention tiling, the reverse gather, rematerialisation and
    # atom blocks (``express.Express``); see ``Backbone``.
    tokens: str = "padded"
    attention_tile: int | None = None
    fold_extra_block: bool = False
    reverse_unique: bool = False
    attention: str = "xla"
    fold_compress: bool = False
    remat: bool = False
    atom_block: int | None = None

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
            attention=self.attention,
            fold_compress=self.fold_compress,
            remat=self.remat,
            atom_block=self.atom_block,
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

        def readout(module, node, edge, cutoffs, pair_mask, atom_mask):
            return _readout(module, node, edge, cutoffs, pair_mask, atom_mask, segments)

        predictions, forces, stress = staged(self, readout, atomic_numbers.shape[0])(
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
            force_scale = self.param(
                "force_scale", nn.initializers.ones, (self.max_atomic_number + 1,)
            )
            out["forces"] = (
                forces * force_scale[atomic_numbers][:, None] * atom_mask[:, None]
            )
        if self.direct_stress:
            stress_scale = self.param("stress_scale", nn.initializers.ones, (1,))
            out["stress"] = stress * stress_scale * atom_mask[:, None, None]
        return out


def _readout(upet, node, edge, cutoffs, pair_mask, atom_mask, segments):
    """The heads of ``upet`` on (a block of) the backbone's features: per-atom
    energy, and the raw direct forces and stress (``None`` without heads)."""
    args = (node, edge, cutoffs, pair_mask, atom_mask, segments)
    energy = Energy(d_head=upet.d_head, name="energy_head")(*args)
    forces = stress = None
    if upet.direct_forces:
        forces = DirectForces(d_head=upet.d_head, name="forces_head")(*args)
    if upet.direct_stress:
        stress = DirectStress(d_head=upet.d_head, name="stress_head")(*args)
    return energy, forces, stress


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

    ``reverse_unique=True`` gathers the reverse messages through an involution
    (mutual real pairs swap, every other slot maps to itself and gets a zero
    partner message, as the masked sentinel gives it otherwise), so the gather
    has unique indices and its transpose needs no scatter-add.

    ``attention="cute"`` runs each tile's attention core (the tile gather,
    scores, softmax, contraction with ``v``, the scatter back, and their
    backward) as one CuTe DSL kernel per direction (``kernels.attention``)
    instead of ``flax``'s ``dot_product_attention``; single tiles only, same
    result to round-off.

    ``fold_compress=True`` folds the inputs of each layer's compress MLP into
    its first ``Dense``: the geometry projection, the neighbour species table
    and the biases are linear, so ``concat([geom_proj, table[Z_j], messages])
    @ W1`` becomes ``messages @ W1_m + (table @ W1_t)[Z_j] + geom @ (W_e W1_g)
    + b'`` with the three small weight products formed at trace time; at layer
    0 the messages are the species embedding itself and the GEMM becomes a
    table lookup. One GEMM of width ``d_pet`` replaces the width-4 GEMM, the
    concat and the width-``3 d_pet`` GEMM; the parameter tree is unchanged and
    the result is the stock one to round-off. Applies only in ``apply``: the
    stock layers run during ``init`` so the parameters exist.

    Token layout. ``tokens="padded"`` (stock) takes the pairs in the
    rectangular ``[N * n]`` layout. ``tokens="flat"`` takes one compacted,
    center-sorted token list ``[P]`` of any static length plus the attention
    tile maps ``tile_tokens`` / ``extra_atoms`` (``select._token_tiles``):
    every per-token layer and the reverse gather run on the flat list, and only
    the attention core gathers tiles of width ``W = tile_tokens.shape[1] + 1``
    (centre in slot 0). Without ``attention_tile`` that is one tile of
    ``k_sel + 1``; with it, an atom spilling past ``W - 1`` neighbours has one
    extra tile and its softmax runs across both (``_attention_tiles``);
    ``fold_extra_block`` computes the extra tiles' own block in the
    rectangular pass. Either way the result is the padded layout's to summation
    order, with the same parameter tree.

    Peak memory. Each GNN layer runs in two stages: the tokens (geometry,
    compress, transformer layers) give the node features and the edge
    messages, and the message update (reverse gather, comb MLP) gives the next
    messages; the readout in ``UPET`` is a third. ``remat=True``
    rematerialises each stage in the backward pass. ``atom_block=B`` maps each
    stage over contiguous blocks of ``B`` atoms (``staged``) with
    rematerialised bodies: on the padded layout a block of atoms is a
    contiguous slice of ``B * n`` pairs, and the only couplings across atoms,
    the reverse gather and the species tables gathered by neighbour, read from
    full arrays that the blocks close over. Only the streams between the
    stages (messages ``[P, d_pet]``, node features ``[N, d_node]``) are
    materialised. ``N`` is cut into ``ceil(N / B)`` equal blocks
    (``structure.atom_blocks``) and must be a multiple of that; extra padded atoms
    are masked like the dummy atom; padded layout only. Neither switch changes
    the parameter tree, and the result is the stock one to summation order.
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
    attention: str = "xla"
    fold_compress: bool = False
    remat: bool = False
    atom_block: int | None = None

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
        assert not (flat and self.atom_block is not None), "atom_block needs padded tokens"

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

        paired = None
        if self.reverse_unique:
            idx = jnp.arange(P)
            paired = pair_mask & pair_mask[reverse] & (reverse[reverse] == idx)
            reverse = jnp.where(paired, reverse, idx)

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
            # the species tables the compress MLP reads: the neighbour embedding, and at
            # layer 0 the edge embedding behind the messages (for the fold)
            neighbor_table = None
            if layer_idx > 0:
                neighbor_table = nn.Embed(
                    self.max_atomic_number + 1,
                    d_pet,
                    name=f"gnn_layers_{layer_idx}_neighbor_embed",
                )(atomic_numbers)
            message_table = edge_embed(atomic_numbers) if layer_idx == 0 else None

            def tokens_stage(
                module,
                *args,
                layer_idx=layer_idx,
                table=neighbor_table,
                message_table=message_table,
            ):
                return _layer_tokens(module, layer_idx, table, message_table, tiles, *args)

            node, edge_flat = staged(self, tokens_stage, N)(
                R_ij,
                r_ij,
                messages,
                neighbors,
                pair_mask,
                node,
                cutoffs_tokens,
                mask,
                conditioning,
            )

            def messages_stage(module, *args, layer_idx=layer_idx, edge_full=edge_flat):
                return _layer_messages(module, layer_idx, edge_full, *args)

            messages = staged(self, messages_stage, N)(
                edge_flat, reverse, paired, messages, pair_mask
            )

        node = node[:, 0, :] * atom_mask[:, None]
        return node, messages, cutoffs


def _layer_tokens(
    backbone,
    layer_idx,
    neighbor_table,
    message_table,
    tiles,
    R_ij,
    r_ij,
    messages,
    neighbors,
    pair_mask,
    node,
    cutoffs_tokens,
    mask,
    conditioning,
):
    """First stage of GNN layer ``layer_idx`` of ``backbone`` on (a block of)
    the atoms: geometry and compress MLP per pair, then the transformer layers
    per atom. Returns the node features ``[N, 1, d_node]`` and the masked edge
    messages ``[P, d_pet]``. ``neighbor_table`` / ``message_table`` are the
    species tables ``[N, d_pet]`` behind the neighbour features and (at layer
    0) the messages."""
    d_pet = backbone.d_pet
    geom = jnp.concatenate([R_ij, r_ij[..., None]], axis=-1)
    if backbone.fold_compress and not backbone.is_initializing():
        tokens_flat = _compress_folded(
            backbone,
            layer_idx,
            neighbor_table,
            message_table,
            geom,
            messages,
            neighbors,
            pair_mask,
        )
    else:
        geom_proj = masked(
            nn.Dense(d_pet, name=f"gnn_layers_{layer_idx}_edge_embed"), geom, pair_mask
        )
        if neighbor_table is None:
            features = [geom_proj, messages]
        else:
            neighbor_feats = neighbor_table[neighbors] * pair_mask[..., None]
            features = [geom_proj, neighbor_feats, messages]
        tokens_flat = masked(
            MLP((d_pet, d_pet), name=f"gnn_layers_{layer_idx}_compress"),
            jnp.concatenate(features, axis=-1),
            pair_mask,
        )

    N = node.shape[0]
    edge = tokens_flat if tiles is not None else tokens_flat.reshape(N, -1, d_pet)
    for attn_idx in range(backbone.num_attention_layers):
        node, edge = TransformerLayer(
            d_pet=d_pet,
            d_node=backbone.d_node,
            d_ff=backbone.d_feedforward,
            num_heads=backbone.num_heads,
            temperature=backbone.attention_temperature,
            fold_extra_block=backbone.fold_extra_block,
            attention_kernel=backbone.attention,
            name=f"gnn_layers_{layer_idx}_trans_layers_{attn_idx}",
        )(node, edge, cutoffs_tokens, mask, tiles)

    if conditioning is not None:
        node = node + conditioning[:, None, :]
    return node, edge.reshape(-1, d_pet) * pair_mask[..., None]


def _compress_folded(
    backbone, layer_idx, neighbor_table, message_table, geom, messages, neighbors, pair_mask
):
    """The compress stage of ``_layer_tokens`` with its inputs folded into the
    first ``Dense`` (see ``Backbone``): the layer's parameters are read directly,
    so this runs in ``apply`` only."""
    d = backbone.d_pet
    params = backbone.variables["params"]
    edge_embed = params[f"gnn_layers_{layer_idx}_edge_embed"]
    compress = params[f"gnn_layers_{layer_idx}_compress"]
    W1, b1 = compress["Dense_0"]["kernel"], compress["Dense_0"]["bias"]
    W2, b2 = compress["Dense_1"]["kernel"], compress["Dense_1"]["bias"]
    W1_geom = W1[:d]
    h = geom @ (edge_embed["kernel"] @ W1_geom) + (edge_embed["bias"] @ W1_geom + b1)
    if neighbor_table is None:
        # layer 0: the messages are the edge embedding of the neighbour's species
        h = h + (message_table @ W1[d:])[neighbors]
    else:
        h = h + (neighbor_table @ W1[d : 2 * d])[neighbors] + messages @ W1[2 * d :]
    out = jax.nn.silu(h) @ W2 + b2
    return jnp.where(pair_mask[..., None], out, jnp.zeros((), out.dtype))


def _layer_messages(
    backbone, layer_idx, edge_full, edge_flat, reverse, paired, messages, pair_mask
):
    """Second stage of GNN layer ``layer_idx``: add the edge messages
    ``edge_flat`` of (a block of) the pairs and, with message passing after
    this layer, their mix with the reverse messages, gathered from all pairs'
    ``edge_full``."""
    d_pet = backbone.d_pet
    messages = messages + edge_flat
    if layer_idx < backbone.num_gnn_layers - 1 or backbone.final_message_passing:
        if backbone.reverse_unique:
            reversed_flat = edge_full.at[reverse].get(unique_indices=True)
            reversed_flat = reversed_flat * paired[..., None]
        else:
            reversed_flat = edge_full[reverse]
        combined = jnp.concatenate([edge_flat, reversed_flat], axis=-1)
        combined = nn.LayerNorm(epsilon=1e-5, name=f"comb_norms_{layer_idx}")(combined)
        combined = MLP((2 * d_pet, d_pet), name=f"comb_mlps_{layer_idx}")(combined)
        messages = messages + combined * pair_mask[..., None]
    return messages


def staged(module, fn, num_atoms):
    """``fn(module, *args)`` as a stage of the forward, run as ``module``'s
    ``remat`` / ``atom_block`` say: as is (stock), rematerialised, or mapped
    over blocks of atoms. Every leaf of ``args`` and of the result is per
    atom or per pair, its leading axis ``num_atoms`` or ``num_atoms * n`` in
    center-major order; a block is the same contiguous fraction of each.

    The Flax lifted transforms take ``fn`` with the module as first argument,
    so the submodules ``fn`` creates keep their names in ``module``'s scope
    and the parameter tree is the stock one; ``variable_broadcast="params"``
    hands every block the same parameters. Arrays ``fn`` closes over are seen
    whole by each block."""
    if module.atom_block is None:
        return functools.partial(nn.remat(fn) if module.remat else fn, module)
    num_blocks, _ = atom_blocks(num_atoms, module.atom_block)

    def body(mod, carry, args):
        return carry, fn(mod, *args)

    scan = nn.scan(
        nn.remat(body, prevent_cse=False),
        variable_broadcast="params",
        split_rngs={"params": False},
        length=num_blocks,
    )

    def run(*args):
        blocks = jax.tree.map(lambda x: x.reshape((num_blocks, -1) + x.shape[1:]), args)
        _, out = scan(module, None, blocks)
        return jax.tree.map(lambda y: y.reshape((-1,) + y.shape[2:]), out)

    return run


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
    fold_extra_block: bool = False
    attention_kernel: str = "xla"

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
        attn = Attention(
            self.num_heads, self.temperature, self.fold_extra_block, self.attention_kernel
        )(rms_norm(tokens, name="norm_attn"), cutoffs, mask, tiles)
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
    fold_extra_block: bool = False
    kernel: str = "xla"

    @nn.compact
    def __call__(self, x, cutoffs, mask, tiles=None):
        if tiles is not None:
            return self._tiled(x, cutoffs, mask, tiles)
        N, T, F = x.shape
        H = self.num_heads
        d = F // H

        qkv = nn.Dense(3 * F, name="qkv")(x)
        if self.kernel == "cute":
            # the padded layout is the flat list [N * T] with the identity tile map
            rows = jnp.arange(N * T, dtype=jnp.int32).reshape(N, T)
            out = self._fused(qkv.reshape(N * T, 3 * F), rows, cutoffs, mask)
            return nn.Dense(F, name="out")(out.reshape(N, T, F))
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

        if self.kernel == "cute":
            # the kernel gathers the tiles itself and writes the flat list back
            assert tiles.extra_atoms.shape[0] == 0, "the cute kernel runs single tiles"
            out = self._fused(nn.Dense(3 * F, name="qkv")(x), tiles.rows, cutoffs, mask)
            return nn.Dense(F, name="out")(out)
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
            out = _attention_tiles(q, k, v, bias, mask, tiles, self.fold_extra_block)

        # Back to the flat list before the output projection, which runs unpadded.
        return nn.Dense(F, name="out")(tiles.scatter(out.reshape(T, W, F)))

    def _fused(self, qkv, rows, cutoffs, mask):
        from .kernels.attention import fused_attention

        bias = jnp.log(jnp.clip(cutoffs, 1e-15, None))
        return fused_attention(qkv, rows, bias, mask, self.num_heads, self.temperature)


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


def _attention_tiles(q, k, v, bias, mask, tiles, fold):
    """Attention over tiles where an atom may own two, ``q, k, v`` ``[T, W, H, d]``
    (``q`` already over the temperature), ``bias`` / ``mask`` ``[T, W]`` of
    the key slots; returns ``[T, W, H, d]``.

    A query sees all of its atom's keys: the diagonal blocks (first tile
    against itself) are one rectangular pass over the first ``N`` tiles; an
    atom with an extra tile adds the blocks (first, extra), (extra, first) and
    (extra, extra), computed on the ``E`` extra tiles only. ``fold`` runs the
    (extra, extra) block inside the rectangular pass instead, over all ``T``.

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
    D = T if fold else N
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

    s_d = scores(q[:D], k[:D], bias[:D], mask[:D])  # [D, H, W, W]
    s_ax = scores(q_a, k_x, bias_x, mask_x)  # (first, extra), [E, H, W, W]
    s_xa = scores(q_x, k_a, bias_a, mask_a)  # (extra, first)
    m_d, m_ax, m_xa = s_d.max(axis=-1), s_ax.max(axis=-1), s_xa.max(axis=-1)
    if fold:
        m_xx = m_d[N:]
    else:
        s_xx = scores(q_x, k_x, bias_x, mask_x)  # (extra, extra)
        m_xx = s_xx.max(axis=-1)

    m_first = jax.lax.stop_gradient(jnp.maximum(m_d[:N], extra_of_atom(m_ax, big_neg)))
    m_extra = jax.lax.stop_gradient(jnp.maximum(m_xa, m_xx))

    m_rows = jnp.concatenate([m_first, m_extra]) if fold else m_first
    l_d, o_d = accumulate(s_d, m_rows, v[:D])
    l_ax, o_ax = accumulate(s_ax, first_of_extra(m_first), v_x)
    l_xa, o_xa = accumulate(s_xa, m_extra, v_a)
    if fold:
        l_xx, o_xx = l_d[N:], o_d[N:]
    else:
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
