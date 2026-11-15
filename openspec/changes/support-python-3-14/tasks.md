# Tasks

The change ends green on `pdm run formatting`, `pdm run checks`, `pdm run docs build` and `pdm run tests`, read from
the nox session summary lines, with `tests-3.12`, `tests-3.13` and `tests-3.14` among them and the docs build run with
no secret set.

## 1. Packaging

- [x] 1.1 In `pyproject.toml`, set `requires-python = ">=3.12, <3.15"`, replace the `3.11` classifier with `3.14`, and
  set the `black` and `ruff` targets to `py312`. Set `python_versions: '>=3.12, <3.15'` in `.copier-answers.yml`.
  Verify that grep finds no `3.11` or `py311` left in either file, apart from the `matplotlib` pin.
- [x] 1.2 Regenerate the lock with `pdm lock --update-reuse`. Verify that `pdm lock --check` passes and that
  `pdm install -G :all` succeeds.

## 2. Tooling

- [x] 2.1 Set `PYTHON_VERSIONS` in `noxfile.py` to `3.12`, `3.13` and `3.14`. Verify with `nox -l`.
- [x] 2.2 In `.github/workflows/ci.yml`, set the `setup-python` lists of the `checks` and `tests` jobs and the `doc`
  job's matrix to `3.12`, `3.13` and `3.14`. Verify on the PR that the jobs `doc (3.12)` to `doc (3.14)` appear.

## 3. Code

- [x] 3.1 Apply the PEP 695 fixes `ruff` asks for under `py312` (`UP040`, `UP047`) after checking that the aliases are
  used only in annotations, in a separate commit. Verify that `ruff check` and `mypy src` pass.
- [x] 3.2 Make the `BaseBrowserVenue` doctest in `src/sportsbet/execution/_browser_venue.py` show the real `TypeError`
  instead of a containment check. Verify that the doctest passes on 3.12, 3.13 and 3.14.
- [x] 3.3 Run the tests on 3.14 and fix each failure in our code, test first where the failure is behavior rather than
  message text. Verify that `Session tests-3.14 was successful`.
- [x] 3.4 Replace `mkdocs-gallery`: add `docs/generate_examples.py` and `docs/example_runner.py`, delete
  `docs/gallery_conf.py`, point the `Examples` nav at `generated/examples/`, exclude `/examples/` from the docs, and
  drop `mkdocs-gallery`, `plotly`, `kaleido` and `pillow` from the docs dependencies. Verify that every example page
  shows its tables, printed output and figures.
- [x] 3.5 Build the docs with `--strict` in the nox docs session, and make the constitution link in `CONTRIBUTING.md`
  absolute. Verify that an example broken on purpose aborts the build, and that the real build on 3.14 succeeds with
  no warnings and no secret set.
- [x] 3.6 Open an issue on the copier template for the same `mkdocs-gallery` failure. Verify that the issue exists.

## 4. Governance

- [x] 4.1 Amend `openspec/constitution.md` to 13.0.0: the Language rule supports `>=3.12, <3.15` targeting `py312`, the
  Compliance example names 3.12, 3.13 and 3.14, the Documentation rule names the strict build of the example pages, a
  Sync Impact Report entry records the MAJOR bump, and the footer shows the new version and amendment date. Update the
  range in `openspec/config.yaml`. Verify that `openspec validate --strict` passes and that grep finds no old range
  outside the archive and the history.

## 5. Ship

- [ ] 5.1 Run the full gate in order, commit, push and open a PR against `development` that closes #162. Verify that
  every CI job passes.

## Workflow follow-up

- After the PR merges, close #162 if it is still open and archive `support-python-3-14`.
