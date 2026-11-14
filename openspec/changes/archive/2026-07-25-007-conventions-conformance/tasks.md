# Tasks

## 1. Setup

- [x] 1.1 Confirm the green baseline before starting: `pdm run formatting`, `pdm run checks`, `pdm run tests` all green
  on 3.11, 3.12 and 3.13, and the browser installed with
  `pdm run python -m playwright install --only-shell chromium`. (T001)

## 2. sources (with `_params`)

- [x] 2.1 Reword the module docstrings of `src/sportsbet/_params.py` and `src/sportsbet/sources/__init__.py` to one
  imperative line (FR-005). `__init__` gets a one-liner, not `It provides ...`. (T002)
- [x] 2.2 Reword the module docstrings of the stats sources to one imperative line:
  `src/sportsbet/sources/_stats/_nba.py`, `_euroleague.py`, `_football_data.py`, `_sample.py`, and
  `_stats/__init__.py`. (T003)
- [x] 2.3 Reword the module docstrings of the odds sources to one imperative line:
  `src/sportsbet/sources/_odds/_football_data.py`, `_odds_api.py`, `_sample.py`, and `_odds/__init__.py`. (T004)
- [x] 2.4 Reword the module docstring of `src/sportsbet/sources/_utils.py` (`Includes utilities ...` to the
  imperative), and check `derive_market_outcomes` and any helper for verb-first names and one-line docstrings. (T005)
- [x] 2.5 Read every `def` in the sources package for FR-001, FR-006, FR-010 and FR-011 violations: noun-first or
  empty-verb names, private helpers with Args or Returns blocks, in-function imports that do not defer an optional
  extra, explanatory inline comments. Rename and collapse. Record any public rename in the ledger and propagate it to
  `sources/__init__` `__all__`, callers, tests and docs. (T006)
- [x] 2.6 Bring the sources tests to the mirror: `tests/sources/test_utils.py`, `test_schema.py`, `test_resolver.py`,
  `test_base.py`, `tests/sources/stats/*` and `tests/sources/odds/*`. Rename each test to
  `test_<function>_<behavior>`, collapse docstrings to one line, fix the `test_stastics_schema` typo, and switch any
  private-module import to the public API. (T007)
- [x] 2.7 Run the gate for sources: `pdm run formatting`, `pdm run checks`, `pdm run tests`. Add a `CHANGELOG.md` line
  for any public rename. Commit the sources package. (T008)

## 3. dataloaders

- [x] 3.1 Reword the module docstrings of `src/sportsbet/dataloaders/_base.py`, `_sourced.py` and `__init__.py` to one
  imperative line. (T009)
- [x] 3.2 Read every non-contract `def` in `dataloaders/_base.py` and `_sourced.py` for FR-001, FR-006, FR-010 and
  FR-011. Do not touch the contract surface: `extract_train_data`, `extract_fixtures_data`, `extract_exploration_data`,
  `save`, `load_dataloader`, the fitted attributes and the constructor parameters. Rename any non-conforming private
  helper and collapse its docstring. Confirm `_base.py` imports no sibling (FR-009). (T010)
- [x] 3.3 Bring the dataloaders tests to the mirror: `tests/dataloaders/test_base.py`, `test_sourced.py`,
  `test_extraction.py`, `test_leakage.py`, `test_basketball.py`, `test_soccer.py`, `test_sources_choice.py`. Name each
  `test_<function>_<behavior>` with a one-line docstring and public API imports. A behaviour test that spans several
  functions keeps a behaviour name that starts from the entry point it exercises. (T011)
- [x] 3.4 Run the gate for dataloaders, add a changelog line for any public rename, commit. (T012)

## 4. evaluation

- [x] 4.1 Reword the module docstrings of `src/sportsbet/evaluation/_base.py`, `_model_selection.py`, `_classifier.py`,
  `_rules.py` and `__init__.py` to one imperative line. (T013)
- [x] 4.2 Weigh `complementary_events`, a noun-first public name. If it is a function, rename it to a verb phrase and
  propagate to `evaluation/__all__`, callers, tests and docs, and record it in the ledger. If it is data, leave it. Do
  not rename the contract surface. (T014)
- [x] 4.3 Read the private helpers `latest_odds_column` and `market_base` in `evaluation/_base.py` and any other
  non-contract `def` for FR-001 and FR-006. Collapse docstrings to one line. Export the two helpers from
  `sportsbet.evaluation` or test them through the public API. (T015)
- [x] 4.4 Bring the evaluation tests to the mirror: `tests/evaluation/test_base.py`, `test_classifier.py`,
  `test_model_selection.py`, `test_rules.py` and `tests/evaluation/__init__.py`. Import `BaseBettor` and the helpers
  from the public API. Name each test `test_<function>_<behavior>` with a one-line docstring. (T016)
- [x] 4.5 Run the gate for evaluation, add a changelog line for any public rename, commit. (T017)

## 5. execution

- [x] 5.1 Reword only the `src/sportsbet/execution/__init__.py` module docstring (`It provides ...`) to one imperative
  line. Confirm by a rescan that the rest of the package already conforms. (T018)
- [x] 5.2 Bring the execution tests to the mirror: `tests/execution/test_execution.py`, `test_schedule.py`,
  `test_browser.py`. Confirm each test is `test_<function>_<behavior>` with a one-line docstring and public API imports,
  and align the test modules to the source modules or name behaviour-grouped modules for the entry point they exercise.
  (T019)
- [x] 5.3 Run the gate for execution, commit. (T020)

## 6. selection and artifacts

- [x] 6.1 Reword the module docstrings of `src/sportsbet/_selection.py` and `src/sportsbet/_artifacts.py` to one
  imperative line, and the top `src/sportsbet/__init__.py` docstring if it narrates. (T021)
- [x] 6.2 Read every `def` in `_selection.py` and `_artifacts.py` for FR-001, FR-006, FR-010 and FR-011. Keep
  `build_dataloader`, `build_bettor` and `build_venue`. Check `_load_object`, `_moments`, `_aliases`, `_odds_source` and
  the artifact helpers for one-line docstrings, and confirm the in-function `from .execution import ...` defers an
  optional extra with its `# noqa` (FR-010). (T022)
- [x] 6.3 Bring the tests of the glue to the mirror (`tests/test_selection.py`, `tests/test_artifacts.py`), or note that
  the CLI and MCP tests exercise them. Name them and give each a one-line docstring. (T023)
- [x] 6.4 Run the gate for the glue, add a changelog line for any public rename, commit. (T024)

## 7. cli

- [x] 7.1 Reword the six `Module that contains ... of the CLI` docstrings to one imperative line:
  `src/sportsbet/cli/_cli.py`, `_data.py`, `_betting.py`, `_execution.py`, `_options.py`, `_utils.py`. (T025)
- [x] 7.2 Read every `def` in the cli package for FR-001, FR-006, FR-010 and FR-011, and confirm `sportsbet.cli`
  exports `main` publicly. Rename non-conforming helpers and collapse docstrings. (T026)
- [x] 7.3 Bring the cli tests to the mirror: `tests/cli/test_cli.py`, `test_data.py`, `test_betting.py`,
  `test_execution.py`, `test_main.py`, `test_parity.py`. Name each `test_<function>_<behavior>` with a one-line
  docstring and public API imports. (T027)
- [x] 7.4 Run the gate for cli, add a changelog line for any public rename, commit. (T028)

## 8. mcp

- [x] 8.1 Reword the module docstrings of `src/sportsbet/mcp/_server.py` and `mcp/__init__.py` to one imperative line.
  Read every `def` in `_server.py` for FR-001, FR-006, FR-010 and FR-011. (T029)
- [x] 8.2 Bring the mcp tests to the mirror: `tests/mcp/test_mcp.py` (consider `tests/mcp/test_server.py`) and
  `test_parity_surfaces.py`. Import `main` from `sportsbet.cli`. Name each test `test_<function>_<behavior>` with a
  one-line docstring. (T030)
- [x] 8.3 Run the gate for mcp, add a changelog line for any public rename, commit. (T031)

## 9. Polish and verification

- [x] 9.1 Regenerate `docs/generated` from `docs/examples` through the docs build and confirm it is not hand-edited.
  Confirm every runnable example still runs (SC-008) and every public name kept a runnable example. (T032)
- [x] 9.2 Verify SC-001 to SC-005 across the sweep: no narrating module docstring outside history, every public
  function verb-first except the contract surface, one-line docstrings except public entry points and classes, no base
  importing a sibling and no function-body import except deferred extras, and tests that mirror the source and follow
  the naming pattern. (T033)
- [x] 9.3 Confirm the rename ledger and `CHANGELOG.md` agree, with every public rename in both with before and after
  (SC-007). Run the final full gate on 3.11, 3.12 and 3.13. (T034)
