Create a new release of pet-jax on PyPI. Optional argument: version bump level (`patch`, `minor`, or `major`).

## Steps

### 1. Pre-flight checks

- Work from a clean checkout of `main` at `origin/main`. If the current checkout is on another branch or has changes, do not switch it; create a worktree instead: `git fetch origin main && git worktree add <scratch>/release origin/main`, and run everything below in there. Remove the worktree at the end.
- Check that CI is passing on the latest commit of `main`: `gh run list --repo lab-cosmo/pet-jax --branch main --limit 3`
- If CI is not green, do NOT proceed — investigate and fix first
- Check that `pyproject.toml` has no direct-URL dependencies (`@ git+...`), including in the extras. PyPI rejects them, and the publish job would fail after the tag is already pushed.
- Check that the README's Installation section matches reality: once the package is on PyPI it should say `pip install pet-jax` (and `pip install "pet-jax[convert]"`). PyPI renders the README of the tagged commit, so fix it before tagging.

### 2. Run full local verification

- Run `uvx --with tox-uv tox -e lint` — must pass
- If `tests/assets/checkpoints/` is missing or stale (the checkpoint list in `tox.ini` or the converter changed since it was populated), run `uvx --with tox-uv tox -e fetch-checkpoints` first; otherwise the checkpoint tests are skipped and prove nothing
- Run `uvx --with tox-uv tox -e tests` — must pass, with no checkpoint tests skipped
- Run `uv build` and check that the sdist and wheel in `dist/` carry the expected version and contain only `petjax`; delete `dist/` afterwards
- Do NOT proceed if any of these fail

### 3. Determine version

- Find the latest git tag with `git describe --tags --abbrev=0` (or note if there are no tags yet; the first release is then `0.1.0` unless the user says otherwise)
- If a bump level was given ($ARGUMENTS), compute the new version following semver (e.g., `0.1.0` → `0.1.1` for patch, `0.2.0` for minor, `1.0.0` for major)
- If NO bump level was given, review the changelog (step 4) first, then discuss with the user what the appropriate level should be based on the nature of the changes (breaking → major, new features → minor, fixes/maintenance → patch)
- Confirm the new version with the user before proceeding

### 4. Finalise the changelog

- PRs add their own entries to the **Unreleased** section of `CHANGELOG.md`, so it should already be mostly complete
- Run `git log <last-tag>..HEAD --oneline` (or all commits if no prior tag) and cross-check: flag user-visible changes without an entry, and entries that are unclear or in the wrong group
- Propose fixes to the user, then turn **Unreleased** into `## [<version>] - <YYYY-MM-DD>` and add a fresh, empty `## Unreleased` above it
- Present the finished section to the user for review and approval
- Open a PR titled `Release v<version>` containing only the `CHANGELOG.md` change, and wait for the user to merge it. Then fetch and check out the merge commit on `origin/main`: the tag must point at the commit that contains the versioned section. If anything else was merged in the meantime, add it to the section first.

### 5. Tag and push

- Create an annotated tag on the release PR's merge commit: `git tag -a v<version> -m "Release v<version>"`
- Push the tag: `git push origin v<version>`
- This triggers `.github/workflows/release.yml`, which builds the package and publishes it to PyPI via trusted publishing from the `release` environment
- Watch it: `gh run watch` (or `gh run list --repo lab-cosmo/pet-jax --workflow release.yml --limit 1`), and confirm the publish job succeeded before continuing

### 6. Create GitHub release

- Use `gh release create v<version> --repo lab-cosmo/pet-jax --title "v<version>" --notes-file <file>`, where the file holds the version's `CHANGELOG.md` section without its heading (inline `--notes` with backticks gets mangled by the sandbox)
- Confirm the new version is visible on PyPI: `curl -s https://pypi.org/pypi/pet-jax/json | python -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"`

## Notes

- The version is derived from git tags by `hatch-vcs` (`[tool.hatch.version] source = "vcs"` in `pyproject.toml`). The only file a release modifies is `CHANGELOG.md`.
- The release workflow uses PyPI trusted publishing (no tokens). The one-time setup is a `release` environment on the GitHub repo and a trusted publisher for `pet-jax` on pypi.org pointing at `lab-cosmo/pet-jax`, `release.yml`, environment `release` (a "pending publisher" until the first upload creates the project).
- Sandbox: `git fetch`, `git push`, and `gh` must run as bare commands in their own Bash call, with the shell already inside the repo (use a separate `cd` call first). Wrapping them in `cd … &&`, `$(…)`, `;` or a pipe keeps them inside the sandbox, where ssh keys and the gh token are unreachable.
