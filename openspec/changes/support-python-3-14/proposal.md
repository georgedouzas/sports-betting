# Proposal

## Why

Python 3.14 came out in October 2025, and the package still caps `requires-python` at `<3.14`. Anyone on 3.14 can't
install `sports-betting` at all: pip and uv both refuse it. Every runtime and dev dependency now publishes 3.14 wheels
for Linux, macOS and Windows, so the cap is the only thing in the way.

At the other end, 3.11 has fallen behind the libraries the package is built on. numpy 2.5 and scipy 1.18 no longer
support it, so a 3.11 user already runs an older numerical stack than the one the package is developed against. The
scientific Python support schedule (SPEC 0) dropped 3.11 in October 2025, three years after its release.

## What Changes

- **BREAKING** The supported range becomes `>=3.12, <3.15`. Python 3.11 is no longer supported, and a 3.11 user stays
  on the last release that allowed it.
- The classifiers drop `3.11` and gain `3.14`, and `black` and `ruff` target `py312`.
- `pdm.lock` is regenerated for the new range, so every locked version has a wheel on 3.12, 3.13 and 3.14.
- `PYTHON_VERSIONS` in `noxfile.py` and the CI `checks`, `tests` and `doc` jobs run 3.12, 3.13 and 3.14.
- `.copier-answers.yml` records `>=3.12, <3.15`, so a later template update keeps the range.
- The `BaseBrowserVenue` doctest shows the real `TypeError` again. It printed a containment check only because the 3.11
  message differs.
- The docs examples move off `mkdocs-gallery`, which crashes on 3.14 because it uses `ast.Str`, which 3.14 removed.
  Its latest release is from September 2024, so no fix is coming. A `gen-files` script, `docs/generate_examples.py`,
  turns each script in `docs/examples` into a page, and markdown-exec runs it. A helper, `docs/example_runner.py`,
  shows each cell's printed output, last value and figures. The docs build runs with `--strict`, so a broken example
  still fails it. `mkdocs-gallery`, `plotly`, `kaleido` and `pillow` leave the docs dependencies.
- The constitution's Language rule and its Compliance example name the new range, and `openspec/config.yaml` states
  it. This is a MAJOR amendment, 12.0.0 to 13.0.0, because it narrows a rule.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. No spec states the supported Python range: the constitution's Project Profile owns it, so the change sets
`skip_specs: true`.

## Impact

- Packaging: `pyproject.toml`, `pdm.lock`, `.copier-answers.yml`.
- Tooling: `noxfile.py`, `.github/workflows/ci.yml`. `equivalence.yml` keeps its single 3.13 run.
- Docs: `docs/generate_examples.py` and `docs/example_runner.py` replace `docs/gallery_conf.py`, with `properdocs.yml`,
  the docs dependencies and the nox docs session updated. The template has the same problem, reported as
  georgedouzas/copier-modern-python#17.
- Code: one doctest in `src/sportsbet/execution/_browser_venue.py`, plus whatever `ruff`'s `py312` rules rewrite.
- Governance: `openspec/constitution.md`, `openspec/config.yaml`.
- Users on 3.11 must upgrade Python to take the next release. The public API doesn't change.
- CI and the local pre-commit `pdm tests` hook still run three versions, so they take no longer than before.
