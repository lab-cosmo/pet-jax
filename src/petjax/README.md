# `pet-jax` internals

Developer-facing notes for the `petjax` package. See the top-level [README.md](../../README.md) for usage, the high-level 2-phase architecture, performance knobs, and project conventions; this file documents the design intent behind the code and the rules contributors should keep.

## Design intent

The top-level README explains the 2-phase split (outside JIT = build + size; inside JIT = re-run + forward + grad). The rationales behind that split, grouped by which side of the JIT boundary they live on:

**Outside JIT** — `structure.to_structure` (host-side build) + `select.determine_k_sel` (CPU-pinned sizing kernel)

- The structure dict stores **cell shifts**, not pre-computed displacements. `R_ij` is re-derived from `(positions, cell)` inside JIT so positions and cell are the only gradient inputs. Clean autodiff w.r.t. both, including the strain derivative for stress.
- `N_padded` / `n_pair_padded` / `k_sel` are snapped to a bucket strategy via `marathon.utils.next_size`. Small fluctuations in atom or pair count must not retrigger XLA compilation.
- Sizing runs on a CPU-pinned JIT to avoid GPU contention with the forward and to read the int back without a device→host sync.

**Inside JIT** — `select.truncate` + `model.UPET`, composed by `predict.get_predict_fn`

- Adaptive per-atom cutoffs are recomputed inside the autograd graph every step. They depend on neighbour positions, so they must contribute to `dE/dpositions` — otherwise forces are wrong.
- Selection uses `vmap(boolean_mask_indices)`, a JIT-compatible `array[mask]` adapted from [GLP](https://github.com/sirmarcel/glp), so shapes stay fixed across selections.
- The reverse-pair map is built via the index chain `raw_to_sel[reverse_raw[sel_to_raw]]`, not by search-by-value. The latter is ambiguous when an atom is its own neighbour through multiple periodic images.
- `value_and_grad(..., argnums=(positions, strain))` gives forces and the virial stress in one pass. Composition shifts (linear in atomic numbers) are added post-JIT in Python fp64 so they contribute zero to forces with no precision loss.

**Per-step orchestration** — `UPETCalculator.calculate`

- The Verlet skin check skips the raw-NL rebuild while displacements stay below the skin. Position-only updates re-derive `R_ij` inside JIT from cached cell shifts and avoid all host-side work.
- If any atom selects more than `k_sel` neighbours, `predict_fn` returns `overflow=True`. The calculator rebuilds with `k_sel_actual + extra_neighbors` and retries. One retry covers all but pathological cases.

## JIT placement rule

The only `@jax.jit` sites in the package are `select._k_sel_kernel` (CPU-pinned, sizing) and `predict.predict_fn` (forward, default device). Every other helper is undecorated and traced into whichever entry point calls it. **Do not add `@jax.jit` elsewhere** — nesting jit inside `predict_fn` defeats the trace-once-then-execute model and silently inflates compile times.

## Key invariant

The `UPET` Flax module in `model.py` is unchanged from the upstream `metatrain` design — all the new machinery (selection, packing, overflow, calculator) wraps it from the outside. Anything that touches `src/petjax/model.py` should be a parity bug fix, not a feature. Features belong in the wrapping code.

Shape stability is best-effort, not enforced: shapes tend to be stable or grow within a relaxation/MD run (bucket granularity absorbs small fluctuations), but a rebuild can land on a smaller bucket if local density drops, triggering a re-JIT at the new size.

## Internal conventions

- **Rectangular NL layout**: shape `[N_padded * k, ...]` where `k = k_raw` (raw NL) or `k = k_sel` (post-selection). A single dummy atom at index `N_padded - 1` absorbs all padded neighbours; pair masks zero out their contributions inside the model.
- **Attention uses `flax.linen.attention.dot_product_attention`**, not `jax.nn.dot_product_attention`. The JAX builtin hardcodes its softmax to fp32 and fills masked slots with `-0.7 * finfo(logits_dtype).max`. That overflows to `-inf` when fp64 logits are cast to fp32, which gives NaN softmax on fully-masked padded-atom rows (only the backward path bites; forward survives because NaN×0 for masked slots is still NaN, but only in the gradient). Flax runs softmax in the input dtype and masks with `finfo(dtype).min` in that same dtype, so both fp32 and fp64 work.

## Code style

- `ruff` with line-length 92 and the suppressed rules in `pyproject.toml` (`E741`, `E731`, `F722`, `E402`, `E501`).
- Import order: `numpy` / `jax` before other third-party (`isort` sections in `pyproject.toml`).
- Factory functions that return JIT'd closures use the `get_*_fn` pattern (e.g. `get_predict_fn`). Data builders use `make_*` or `to_*` (e.g. `to_structure`).
- Input-producing functions take configurable `int_dtype` / `float_dtype` kwargs.
- `# -- section --` separators for intra-file grouping.
- `__all__` in `src/petjax/__init__.py` defines the public API.
- Private functions use a `_` prefix.
- Top-down file order: public API / entry points at the top, supporting components below. Semantic grouping (e.g. a jitted kernel paired with its caller) beats strict caller-before-callee when the two read as one unit. For classes: `__init__`, then classmethods, then the main public method, then private helpers.
- Docstrings only where behaviour isn't obvious from the signature. Comments explain *why*, not *what*: terse, self-contained, written for an outside reader.
- Lazy-import optional deps (e.g. `torch`, `metatomic.torch` inside `convert.py`).

## Markdown

- Do **not** hard-wrap lines in `.md` files. Each paragraph (and each list item, however long) is one physical line; blank lines separate paragraphs/items as usual. Code fences, tables, and headers are unaffected. Let the renderer / editor soft-wrap.
- Wrap package and tool names in inline backticks in prose: `pet-jax`, `tox`, `ruff`, `pytest`, `vesin`, `metatrain`, `numpy`, `jax`, `flax`, `isort`, etc. **Doesn't apply inside fenced code blocks, inline code spans, Python docstrings, or code comments** — those are already code contexts. Skip uppercase framework references (JAX, Flax, PyTorch, ASE) and model identifiers (PET, PET-MAD, LLPR); they aren't package names in the same sense.
