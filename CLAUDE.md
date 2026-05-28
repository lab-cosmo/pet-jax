# `pet-jax` — notes for AI agents

There are two README files. Read both before editing the code.

- [`README.md`](README.md) — user-facing. Quick start, public API, high-level 2-phase architecture sketch, performance knobs, checkpoint format, dev workflow (`tox` / `uv`), dependencies, status.
- [`src/petjax/README.md`](src/petjax/README.md) — developer-facing. Design intent behind each side of the JIT boundary, the JIT placement rule (`@jax.jit` lives only at two sites), the upstream-faithful `UPET` invariant, internal NL/padding/attention conventions, code style, Markdown soft-wrap rule.

Section pointers, so you don't have to grep:

- Usage examples + public API → top-level README §Quick start
- High-level 2-phase architecture → top-level README §Architecture
- User-facing knobs (`skin`, `default_dtype`, `bucket_strategy`, `cutoff_override`, …) → top-level README §Performance knobs
- Calculator-level invariants (energy/stress conventions, scaling, adaptive cutoff) → top-level README §Conventions
- Dev workflow (`tox`, `uv`, lint, tests, fetching the reference checkpoint) → top-level README §Development
- **Design rationale by JIT phase, the `flax.linen.attention` softmax/dtype caveat, the rectangular-NL layout** → `src/petjax/README.md`
- **Code conventions (`ruff`, naming, file layout, comment policy, Markdown)** → `src/petjax/README.md`

When you learn or decide something that other contributors will need, add it to the relevant section in whichever README is appropriate — do not duplicate it into this file. Keep this file short.
