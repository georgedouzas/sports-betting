# Tasks

Each group ends green on `pdm run formatting`, `pdm run checks`, `pdm run docs build` and `pdm run tests`, read from
the nox session summary line.

## 1. Alias form

- [x] 1.1 Write a test in `tests/cli/` that passes `--alias 'odds name=stats name'` for a club the pairing leaves over,
  using the offline sample sources, and asserts that the club's matches keep their odds.
- [x] 1.2 Change the `--alias` help in `src/sportsbet/cli/_options.py` and the `aliases` docstring of
  `build_dataloader` in `src/sportsbet/dataloaders/_factory.py` to `odds name=stats name`. Rename the variables in
  `_parse_aliases` to the order they are read. Verify with `sportsbet dataloader train extract --help`.
- [x] 1.3 Update every page under `docs/` that shows `--alias` or describes the form.

## 2. Build errors exit with status 1

- [x] 2.1 Write failing tests: a malformed `--alias` exits 1 with the message in the output and no `Traceback`, and a
  malformed `--target-event-time` exits 1 with no `Traceback`.
- [x] 2.2 In `src/sportsbet/cli/_building.py`, turn `_build_selected` and `_build_modelled` into plain builders that
  let `BuildError` propagate. Remove the `None` branches in `_data.py` and `_betting.py`. Verify the tests of 2.1.

## 3. Record

- [x] 3.1 Commit as `fix(cli): ...`.
