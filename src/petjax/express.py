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
        own default (10 from ``[0, cutoff]``, 4 from the window bracket).
      window_capacity: pairs per atom the window variant can hold.
      candidate_extra: sorted candidates per atom beyond ``k_sel`` that the
        window variant considers for selection.
      seed_cutoffs: start the solve from the previous call's cutoffs with the
        Verlet-bound bracket ``[r - 2 delta, r + 2 delta]``.

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
    """

    solver: str = "segment_sum"
    solver_iterations: int | None = None
    window_capacity: int = 64
    candidate_extra: int = 8
    seed_cutoffs: bool = False
    neighbor_list: str = "host"
    cutoff_guard: bool = False
    tokens: str = "padded"
    token_margin: float = 1.05
    attention_tile: int | None = None
    tile_margin: float = 1.5
    tile_floor: int = 8
    fold_extra_block: bool = False
    reverse_unique: bool = False

    def __post_init__(self):
        _check_choice("solver", self.solver, SOLVERS)
        _check_choice("neighbor_list", self.neighbor_list, NEIGHBOR_LISTS)
        _check_choice("tokens", self.tokens, TOKEN_LAYOUTS)
        if self.attention_tile is not None and self.tokens != "flat":
            raise ValueError("attention_tile needs tokens='flat'")
        if self.fold_extra_block and self.attention_tile is None:
            raise ValueError("fold_extra_block needs attention_tile")
        if self.attention_tile is not None and self.attention_tile < 2:
            raise ValueError("attention_tile must hold the centre and one neighbour")

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
        }
        stock = Express()
        return {k: v for k, v in updates.items() if v != getattr(stock, k)}


def _check_choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


EXPRESS_FIELDS = tuple(f.name for f in fields(Express))
