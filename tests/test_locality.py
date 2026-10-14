"""`Backbone(final_message_passing=False)`: checkpoint-free checks on a tiny
randomly initialised backbone.

With one GNN layer the stock backbone's per-atom energy reaches 2 × cutoff (the
reverse edge k → i has attended over k's neighbours), the local one only 1 × cutoff.
A chain i – k – j with j between cutoff and 2 × cutoff from i separates the two:
moving j must change E_i for the stock backbone and leave it bit-identical for the
local one.
"""

import numpy as np
import jax
import jax.numpy as jnp

import flax.linen as nn

from petjax import Backbone, Energy

CUTOFF = 3.0
CONFIG = dict(
    d_pet=8,
    d_node=16,
    d_feedforward=8,
    num_heads=2,
    cutoff=CUTOFF,
    cutoff_width=1.0,
    max_atomic_number=8,
)


def _inputs(x_j):
    """Chain along x: i at 0, k at 2.0, j at ``x_j``. Pairs within the cutoff only,
    padded to two slots per atom (the padded slot is masked and self-reverse)."""
    positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [x_j, 0.0, 0.0]])
    assert CUTOFF < x_j < 2 * CUTOFF
    assert x_j - 2.0 < CUTOFF
    # Slot layout: (center, neighbor) per padded pair, -1 for padding
    pairs = [(0, 1), (0, -1), (1, 0), (1, 2), (2, 1), (2, -1)]
    centers = jnp.array([i for i, _ in pairs])
    neighbors = jnp.array([max(j, 0) for _, j in pairs])
    pair_mask = jnp.array([j >= 0 for _, j in pairs])
    reverse = jnp.array(
        [pairs.index((j, i)) if j >= 0 else idx for idx, (i, j) in enumerate(pairs)]
    )
    return {
        "R_ij": (positions[neighbors] - positions[centers]) * pair_mask[:, None],
        "centers": centers,
        "neighbors": neighbors,
        "atomic_numbers": jnp.array([8, 1, 8]),
        "reverse": reverse,
        "pair_mask": pair_mask,
        "atom_mask": jnp.ones(3, dtype=bool),
    }


class _Model(nn.Module):
    """Backbone plus energy readout, so per-atom energies are the observable."""

    num_gnn_layers: int = 1
    final_message_passing: bool = True

    @nn.compact
    def __call__(self, **inputs):
        node, edge, cutoffs = Backbone(
            **CONFIG,
            num_gnn_layers=self.num_gnn_layers,
            final_message_passing=self.final_message_passing,
            name="backbone",
        )(**inputs)
        return Energy(d_head=8, name="energy_head")(
            node, edge, cutoffs, inputs["pair_mask"], inputs["atom_mask"]
        )


def _energy_of_i(model, params, x_j):
    return np.asarray(model.apply(params, **_inputs(x_j)))[0]


def test_local_backbone_ignores_atoms_beyond_cutoff():
    stock = _Model()
    local = _Model(final_message_passing=False)
    for model, expect_change in ((stock, True), (local, False)):
        params = model.init(jax.random.key(0), **_inputs(4.0))
        e_before = _energy_of_i(model, params, 4.0)
        e_after = _energy_of_i(model, params, 4.3)
        assert (e_before != e_after) == expect_change, model.final_message_passing
    # Not a dead trunk: a neighbour inside the cutoff still reaches E_i
    params = local.init(jax.random.key(0), **_inputs(4.0))
    inputs = _inputs(4.0)
    inputs["R_ij"] = inputs["R_ij"].at[0].add(0.1)
    assert _energy_of_i(local, params, 4.0) != np.asarray(local.apply(params, **inputs))[0]


def test_local_backbone_parameter_tree():
    """The last layer's combine parameters are absent; a stock parameter tree still
    applies, its extra entries unused."""
    stock = _Model(num_gnn_layers=2)
    local = _Model(num_gnn_layers=2, final_message_passing=False)
    inputs = _inputs(4.0)
    stock_params = stock.init(jax.random.key(0), **inputs)["params"]
    local_params = local.init(jax.random.key(0), **inputs)["params"]
    stock_keys, local_keys = set(stock_params["backbone"]), set(local_params["backbone"])
    assert {"comb_norms_0", "comb_mlps_0"} <= local_keys
    assert stock_keys - local_keys == {"comb_norms_1", "comb_mlps_1"}

    from_stock = local.apply({"params": stock_params}, **inputs)
    from_local = local.apply({"params": local_params}, **inputs)
    np.testing.assert_array_equal(from_stock, from_local)
