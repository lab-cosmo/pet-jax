"""Host-side structure-dict construction for UPETCalculator — pure numpy.

Build the sorted+padded flat NL with vesin and the vendored metatrain
corresponding-edges map, emit the result as a plain dict (the "structure";
no labels, since pet-jax doesn't train here). The in-JIT adaptive-cutoff
machine + model forward live in ``select.py`` + ``predict.py``.
"""

import numpy as np

from marathon.utils import next_size


def to_structure(
    atoms,
    cutoff,
    species_to_index,
    skin=0.5,
    bucket_strategy="multiples",
    n_atoms_bucket_strategy=None,
    n_pair_bucket_strategy=None,
    int_dtype=np.int64,
    float_dtype=np.float64,
):
    """Build the structure dict (the raw padded flat NL) from ASE ``atoms``.

    ``bucket_strategy`` is the default bucket strategy for the pair axis.
    ``n_pair_bucket_strategy`` optionally overrides it for that axis.

    ``n_atoms_bucket_strategy`` controls bucketing of ``N_padded``. Defaults
    to ``None`` — no bucketing — because ``n_atoms`` is constant across an
    MD trajectory, so any padding above ``n_atoms + 1`` is dead per-atom
    work in the model body. Set explicitly only if a single Calculator is
    reused across structures of different sizes.

    Pairs come back grouped by center (``sorted=True``): ``_pack_selected_to_flat``
    and the segment sums require ``i`` non-decreasing.

    Returns the structure dict. ``N_padded`` / ``n_pair_padded`` are read off
    ``structure["positions"].shape[0]`` / ``structure["centers"].shape[0]``;
    ``k_sel_sizer`` is put in by the calculator after ``determine_k_sel``.
    """
    if n_pair_bucket_strategy is None:
        n_pair_bucket_strategy = bucket_strategy

    n_atoms = len(atoms)
    effective_cutoff = cutoff + skin

    from vesin import NeighborList

    nl = NeighborList(cutoff=effective_cutoff, full_list=True, sorted=True)
    box = atoms.get_cell().array
    points = atoms.get_positions()
    if atoms.pbc.any():
        i, j, _, S = nl.compute(
            points=points, box=box, periodic=atoms.pbc.tolist(), quantities="ijDS"
        )
    else:
        i, j, _ = nl.compute(
            points=points, box=box, periodic=[False, False, False], quantities="ijD"
        )
        S = np.zeros((len(i), 3), dtype=float_dtype)

    N_padded = _bucket_or(n_atoms + 1, n_atoms_bucket_strategy)
    dummy_atom = N_padded - 1

    n_pair_raw = len(i)
    # +1 reserves a guaranteed-empty sentinel slot at index n_pair_padded - 1.
    # The flat-pack kernel uses this slot as the "masked-out" target both for
    # `slot[non-fitting pair] = P_sel - 1` and for `sel_to_pair[unused] =
    # n_pair_padded - 1`. Without the +1, when n_pair_raw exactly hits a bucket
    # boundary the sentinel would coincide with a real pair.
    n_pair_padded = _bucket_or(n_pair_raw + 1, n_pair_bucket_strategy)

    if n_pair_raw > 0:
        i = i.astype(int_dtype)
        j = j.astype(int_dtype)
        S = np.asarray(S, dtype=float_dtype)
        # Reverse map operates on the sparse arrays in their center-sorted
        # layout — the output indexes the same flat array.
        reverse_sparse = _get_corresponding_edges(i, j, S)
    else:
        # Empty raw NL (isolated atom / cutoff smaller than min image distance).
        # Allocate zero-length sparse arrays; the padded layout below still
        # has shape ≥ 1 thanks to the +1 in n_pair_padded, so JIT shapes are
        # always non-zero on the pair axis.
        i = np.zeros(0, dtype=int_dtype)
        j = np.zeros(0, dtype=int_dtype)
        S = np.zeros((0, 3), dtype=float_dtype)
        reverse_sparse = np.zeros(0, dtype=int_dtype)

    centers = np.full(n_pair_padded, dummy_atom, dtype=int_dtype)
    others = np.full(n_pair_padded, dummy_atom, dtype=int_dtype)
    cell_shifts = np.zeros((n_pair_padded, 3), dtype=float_dtype)
    pair_mask = np.zeros(n_pair_padded, dtype=bool)
    # Padded pairs are self-referential under reverse so the gather
    # `reverse_pair[sel_to_pair]` is well-defined for slots that map to a
    # padded pair (sel_to_pair = n_pair_padded - 1 for unused P_sel slots).
    reverse = np.arange(n_pair_padded, dtype=int_dtype)

    centers[:n_pair_raw] = i
    others[:n_pair_raw] = j
    cell_shifts[:n_pair_raw] = S
    pair_mask[:n_pair_raw] = True
    reverse[:n_pair_raw] = reverse_sparse.astype(int_dtype)

    species = np.zeros(N_padded, dtype=int_dtype)
    species[:n_atoms] = [species_to_index[int(z)] for z in atoms.get_atomic_numbers()]

    atom_mask = np.zeros(N_padded, dtype=bool)
    atom_mask[:n_atoms] = True

    cell = np.array(atoms.get_cell()[:], dtype=float_dtype)
    if not atoms.pbc.any() and (cell == 0).all():
        cell = np.eye(3, dtype=float_dtype)

    positions = np.zeros((N_padded, 3), dtype=float_dtype)
    positions[:n_atoms] = atoms.get_positions()

    return {
        "positions": positions,
        "cell": cell,
        "species": species,
        "atom_mask": atom_mask,
        "centers": centers,
        "others": others,
        "cell_shifts": cell_shifts,
        "reverse": reverse,
        "pair_mask": pair_mask,
    }


# -- helpers --


def _bucket_or(value, strategy):
    """``next_size`` with a ``None`` escape hatch for callers that prefer
    tight sizes over JIT-shape stability."""
    return int(next_size(value, strategy=strategy)) if strategy else int(value)


# -- vendored from metatrain (pet/modules/nef.py), translated to numpy --


def _get_corresponding_edges(centers, neighbors, cell_shifts):
    """For each edge ``(i, j, S)``, return the index of ``(j, i, -S)``.

    Encodes both as unique int64 via base multiplication, then two argsorts
    match the permutations. No dicts, no Python loops.
    """
    if len(centers) == 0:
        return np.empty(0, dtype=np.int64)

    centers = centers.astype(np.int64)
    neighbors = neighbors.astype(np.int64)
    sx = cell_shifts[:, 0].astype(np.int64)
    sy = cell_shifts[:, 1].astype(np.int64)
    sz = cell_shifts[:, 2].astype(np.int64)

    nsx = -sx
    nsy = -sy
    nsz = -sz

    # Shift to non-negative range so the encoding fits
    min_sx = sx.min()
    sx = sx - min_sx
    nsx = nsx - min_sx
    min_sy = sy.min()
    sy = sy - min_sy
    nsy = nsy - min_sy
    min_sz = sz.min()
    sz = sz - min_sz
    nsz = nsz - min_sz

    max_cn = int(max(centers.max(), neighbors.max())) + 1
    max_sx = int(sx.max()) + 1
    max_sy = int(sy.max()) + 1
    max_sz = int(sz.max()) + 1

    size_1 = max_sz
    size_2 = max_sy * size_1
    size_3 = max_sx * size_2
    size_4 = max_cn * size_3

    unique_id = centers * size_4 + neighbors * size_3 + sx * size_2 + sy * size_1 + sz
    unique_id_inverse = (
        neighbors * size_4 + centers * size_3 + nsx * size_2 + nsy * size_1 + nsz
    )

    arg = np.argsort(unique_id, kind="stable")
    arg_inv = np.argsort(unique_id_inverse, kind="stable")

    corresponding = np.empty_like(centers)
    corresponding[arg] = arg_inv
    return corresponding
