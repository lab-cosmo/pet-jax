"""Performance switches for the calculator and the forward ("express").

Every switch defaults to the stock path and is exact: switching it on changes
how a quantity is computed, never what it is, up to summation order. The
equivalence test in ``tests/test_express.py`` holds every non-default setting
to that in fp64 against the stock path. The calculator records the active
settings in ``debug_stats["express"]`` and the benchmark driver in its JSON,
so a timing is never separated from the switches that produced it.

Values are trace-time constants: ``get_predict_fn`` closes over the instance,
and the model-level switches reach the Flax modules as attributes via
``model_updates``.
"""

from dataclasses import asdict, dataclass, fields

SOLVERS = ("segment_sum", "sorted_segment_sum", "blocked_prefix_sum", "sorted_window")
NEIGHBOR_LISTS = ("host", "device")
TOKEN_LAYOUTS = ("padded", "flat")
ATTENTION_KERNELS = ("xla", "cute")
SOLVER_KERNELS = ("xla", "cute")


@dataclass(frozen=True)
class Express:
    """The switches, grouped by what they change.

    Adaptive-cutoff solver (``select.py``):
      solver: how the per-center sums of the Newton-bisection are done.
        ``segment_sum`` is the stock scatter; ``sorted_segment_sum`` the same
        with ``indices_are_sorted``; ``blocked_prefix_sum`` a two-level prefix
        sum on the center-sorted list; ``sorted_window`` the solver on each
        atom's sorted distances, where only the pairs inside the bracket's
        window are touched and the selection packs from the sorted candidate
        prefix.
      solver_iterations: Newton-bisection steps; ``None`` means the variant's
        own default (10 from ``[0, cutoff]``, 4 from the window bracket). With
        ``seed_cutoffs``, the steps of a seeded solve.
      window_capacity: pairs per atom the window variant can hold.
      candidate_extra: sorted candidates per atom beyond ``k_sel`` that the
        window variant considers for selection.
      seed_cutoffs: start the solve from the previous call's cutoffs with the
        Verlet-bound bracket ``[r - 2 delta, r + 2 delta]``. The calculator
        supplies the seed; without a usable one (first call, other atoms, a
        step wider than the bump) the solve takes the stock bracket and at
        least 10 steps. A no-op for ``sorted_window``, which brackets from the
        sorted distances.
      solver_kernel: ``xla`` or ``cute``: run the ``sorted_window`` solve and
        its implicit-function derivative as one CuTe DSL kernel per direction
        with a warp per atom (``kernels.solver``; fp32 only, exact to
        round-off, needs ``pet-jax[cute]`` and a GPU). The atom's raw pairs
        are sorted inside the warp, so the global sorts go; on the padded
        layout the rows then pack in list order, not by distance.
      solver_capacity: raw pairs per atom the kernel holds (a power of two up to
        256, above ``window_capacity``); more raises ``overflow``.

    Raw neighbour list (``structure.py`` / ``calculator.py``):
      neighbor_list: ``host`` (vesin at ``cutoff + skin``, Verlet-cached) or
        ``device`` (built inside the forward every step, no skin).
      cutoff_guard: with ``cutoff_override``, treat a per-atom cutoff above the
        raw radius as an event (discard the step, widen, rebuild) instead of a
        warning.

    Tokens and attention (``model.py``):
      tokens: ``padded`` (every atom ``k_sel`` slots) or ``flat`` (one
        compacted list of static capacity, sized with ``token_margin``).
      attention_tile: width of the fixed attention tile (centre plus
        ``attention_tile - 1`` neighbours; an atom with more spills into one
        extra tile, sized with ``tile_margin`` and ``tile_floor``); ``None``
        keeps the single ``k_sel + 1`` tile. Needs ``tokens="flat"``.
      fold_extra_block: run the extra tiles' diagonal block inside the
        rectangular pass instead of the compacted extra path.
      reverse_unique: gather the reverse messages through an involution with
        unique indices.
      attention: ``xla`` (``flax``'s ``dot_product_attention``) or ``cute``
        (``kernels.attention``: one CuTe DSL kernel per tile and direction,
        fp32 only, exact to round-off; needs ``pet-jax[cute]`` and a GPU).
        Single-tile attention only, so not with ``attention_tile``.

    The compress MLP (``model.py``):
      fold_compress: fold the geometry projection, the neighbour species table
        and the biases into the compress MLP's first ``Dense`` (exact; one
        GEMM of width ``d_pet`` instead of two GEMMs and a concat, a table
        lookup at layer 0). The parameter tree is unchanged.

    Peak memory (``model.py``):
      remat: rematerialise each GNN layer's two stages (geometry, compress and
        transformer; message update) and the readout in the backward pass
        instead of keeping their activations.
      atom_block: run those stages over contiguous blocks of that many atoms
        (``lax.scan``), block bodies rematerialised, so only the per-layer
        streams ``[P, d_pet]`` and ``[N, d_node]`` and one block's activations
        are live. It is an upper bound: the atoms are cut into the fewest
        equal blocks of at most that size, and the calculator pads ``N_padded``
        to a multiple of their number (below ``N_padded / atom_block + 1``
        atoms of padding). Needs ``tokens="padded"``.
    """

    solver: str = "segment_sum"
    solver_iterations: int | None = None
    window_capacity: int = 64
    candidate_extra: int = 8
    seed_cutoffs: bool = False
    solver_kernel: str = "xla"
    solver_capacity: int = 256
    neighbor_list: str = "host"
    cutoff_guard: bool = False
    tokens: str = "padded"
    token_margin: float = 1.05
    attention_tile: int | None = None
    tile_margin: float = 1.5
    tile_floor: int = 8
    fold_extra_block: bool = False
    reverse_unique: bool = False
    attention: str = "xla"
    fold_compress: bool = False
    remat: bool = False
    atom_block: int | None = None

    def __post_init__(self):
        _check_choice("solver", self.solver, SOLVERS)
        _check_choice("neighbor_list", self.neighbor_list, NEIGHBOR_LISTS)
        _check_choice("tokens", self.tokens, TOKEN_LAYOUTS)
        _check_choice("attention", self.attention, ATTENTION_KERNELS)
        _check_choice("solver_kernel", self.solver_kernel, SOLVER_KERNELS)
        if self.solver_kernel == "cute" and self.solver != "sorted_window":
            raise ValueError("solver_kernel='cute' runs the sorted_window solver only")
        if self.attention == "cute" and self.attention_tile is not None:
            raise ValueError(
                "attention='cute' runs single tiles only, not with attention_tile"
            )
        if self.attention_tile is not None and self.tokens != "flat":
            raise ValueError("attention_tile needs tokens='flat'")
        if self.fold_extra_block and self.attention_tile is None:
            raise ValueError("fold_extra_block needs attention_tile")
        if self.attention_tile is not None and self.attention_tile < 2:
            raise ValueError("attention_tile must hold the centre and one neighbour")
        if self.atom_block is not None and self.tokens != "padded":
            raise ValueError("atom_block needs tokens='padded'")
        if self.atom_block is not None and self.atom_block < 1:
            raise ValueError("atom_block must be at least 1")

    @classmethod
    def from_any(cls, value):
        """``None`` (stock), a dict of fields, or an instance."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**dict(value))

    def as_dict(self):
        return asdict(self)

    def is_stock(self):
        return self == Express()

    def model_updates(self):
        """The switches the Flax modules take as attributes (for
        ``model.clone``), only where they differ from the stock model."""
        updates = {
            "tokens": self.tokens,
            "attention_tile": self.attention_tile,
            "fold_extra_block": self.fold_extra_block,
            "reverse_unique": self.reverse_unique,
            "attention": self.attention,
            "fold_compress": self.fold_compress,
            "remat": self.remat,
            "atom_block": self.atom_block,
        }
        stock = Express()
        return {k: v for k, v in updates.items() if v != getattr(stock, k)}


def _check_choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


EXPRESS_FIELDS = tuple(f.name for f in fields(Express))
