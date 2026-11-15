# Design

## Context

The supported range lives in six places: `requires-python`, the classifiers and the `black` and `ruff` targets in
`pyproject.toml`, `PYTHON_VERSIONS` in `noxfile.py`, three lists in `.github/workflows/ci.yml`, `.copier-answers.yml`,
and the constitution with `openspec/config.yaml`. The committed `pdm.lock` was resolved for `>=3.11, <3.14`, and CI
installs from it.

What was checked while planning, on 2026-11-15:

- `uv pip compile` with `--python-version 3.14 --only-binary :all:` resolves every runtime dependency, both extras
  (`mcp`, `execution`) and all 36 dev dependencies, for `x86_64` Linux, `aarch64` macOS and `x86_64` Windows. No
  package needs a source build.
- Resolving on 3.11 gives numpy 2.4.6 and scipy 1.17.1, while 3.12 and 3.14 get numpy 2.5.4 and scipy 1.18.1. The
  two libraries have already dropped 3.11.
- `src/sportsbet/dataloaders/_base.py` builds pandera schemas with `type()` and a hand-set `__annotations__`. Python
  3.14 evaluates annotations lazily (PEP 649), so this was the likeliest break. A reduced copy of the pattern builds
  and validates correctly on 3.14.10 with the current pandera.
- A search of `src`, `tests`, `noxfile.py` and the examples found no API that 3.14 removed, and no `sys.version_info`
  branch that the drop of 3.11 would leave dead.
- The abstract class `TypeError` reads `Can't instantiate abstract class Book without an implementation for abstract
  method 'add_selection'` on 3.12, 3.13 and 3.14 alike.
- The docs build failed on 3.14 inside `mkdocs-gallery` 0.10.4, which tests `isinstance(node, ast.Str)` and gets an
  `AttributeError`. That release is the latest, from September 2024.

## Goals / Non-Goals

**Goals:**

- Install and pass the full gate on 3.12, 3.13 and 3.14 on all three CI operating systems.

**Non-Goals:**

- The free-threaded build (`3.14t`). It needs separate wheels and its own test pass.
- Rewriting code to use 3.12 features by hand. Only what `ruff` asks for under `py312` goes in.

## Decisions

- **Drop 3.11 in the same change as adding 3.14.** Both edit the same six places and the same lock, and doing them
  together keeps CI at three versions. Two changes would mean two lock regenerations and a period with four versions.
- **Replace `mkdocs-gallery` with a `gen-files` script and markdown-exec.** The scripts in `docs/examples` keep their
  gallery format, so `ruff` and `mypy` still check them. At build time each becomes a page. The docstring gives the
  title and introduction, and each `# %%` cell gives text, a shown code block and a hidden markdown-exec block, which
  runs the cell in one session per page through `docs/example_runner.py`. The runner shows the printed output, the
  HTML of the last value and the open figures as SVG, as the gallery did. markdown-exec puts its own `print` in the
  namespace, so the runner points the cell's `print` at its buffer while the cell runs. The alternatives were patching
  `ast.Str` back in before the gallery runs, which edits the standard library under a tool we do not control, or
  keeping the docs job on 3.12 and 3.13, which leaves the build untested on a supported version.
- **Build the docs with `--strict`.** markdown-exec reports a failing block as a warning, not an error, so only strict
  mode fails the build on a broken example. A deliberately broken example was seen to abort the build. The one
  warning the build already had, a relative link to the constitution in `CONTRIBUTING.md`, becomes an absolute link.
- **Cap at `<3.15` rather than drop the cap.** The project has always capped at the next minor, and the copier
  template does the same. An open range would let 3.15 users install a version that was never tested on 3.15.
- **Leave the lower bounds of numpy, pandas and scikit-learn alone.** A resolver on 3.14 picks a release with 3.14
  wheels, and CI installs from the lock. Raising the bounds would only matter for a lowest-version resolution, which
  the project doesn't run.
- **Regenerate the lock with `pdm lock --update-reuse`.** That keeps every pin that still fits and changes only the ones
  that must, so the diff stays small. A full `pdm update` would mix unrelated upgrades into this change.
- **Run every multi-version job on the new range.** The `checks` and `doc` jobs catch version-specific failures that a
  tests-only run would miss.

## Risks / Trade-offs

- [A doctest or test asserts on interpreter text that changed in 3.14] → Assert on a stable part of the message.
- [3.14 emits new `DeprecationWarning`s that the pytest configuration turns into errors] → Fix the source of the
  warning in our code. Filter it only when it comes from a third-party package.
- [`ruff` under `py312` asks for PEP 695 syntax through fixes it marks unsafe] → It flags the six `TypeAlias` aliases
  in `core/_types.py` and the `TypeVar` of `_offload`. A `type` alias is a `TypeAliasType`, which `isinstance` rejects,
  but every one of the six holds a subscripted generic that `isinstance` already rejected, and none is used outside
  annotations. So the fixes are applied, in a separate commit inside the PR.
- [A 3.11 user upgrades and pip installs the old release silently] → The changelog marks the drop as breaking, and
  `requires-python` makes pip pick the last 3.11 release rather than fail at import.

## Migration Plan

Users on 3.11 upgrade Python before taking the next release. To roll back, revert the commits, which restores the old
range and the previous lock.
