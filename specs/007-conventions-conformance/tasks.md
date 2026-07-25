---

description: "Task list for conventions conformance"
---

# Tasks: Conventions conformance

**Input**: Design documents from `specs/007-conventions-conformance/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md),
[data-model.md](./data-model.md), [quickstart.md](./quickstart.md),
[contracts/rename-ledger.md](./contracts/rename-ledger.md)

**Organization**: One phase per package, bottom-up. A package is one commit and is done only when the full
gate is green on 3.11/3.12/3.13. The three user stories cut across every package: US1 (source conforms),
US2 (tests mirror and are named for what they test), US3 (behaviour and the gate are preserved). Tasks are
labelled by the story they serve.

---

## Non-negotiables

These bind every task. A task that cannot be done within them stops and reports.

- **BEHAVIOUR-PRESERVING** (FR-016). No test changes what it asserts. The public API is unchanged except the
  Principle VI renames, each recorded in [contracts/rename-ledger.md](./contracts/rename-ledger.md) and in
  `CHANGELOG.md` with before and after.
- **THE scikit-learn CONTRACT SURFACE IS FIXED** (FR-004, research D1). Do NOT rename any of: `fit`,
  `predict`, `predict_proba`, `bet`, `score`, `get_params`, `set_params`; any trailing-underscore fitted
  attribute; any constructor parameter name; the classes `BaseBettor`, `ClassifierBettor`,
  `OddsComparisonBettor`, `BettorGridSearchCV`, `BaseDataLoader`, `DataLoader`, `BaseSource`,
  `BaseStatsSource`, `BaseOddsSource`, `BaseVenue`.
- **A base module imports NO sibling** (FR-009). **No import inside a function body** except a deferred
  optional extra carrying `# noqa: PLC0415` and a reason (FR-010).
- **Tests mirror the source**, are named `test_<function>_<behavior>` with a single-line docstring, use the
  public API, and never touch the network (FR-012 to FR-015).
- **Every package phase ends with the full gate green** on 3.11/3.12/3.13: `pdm run formatting`,
  `pdm run checks`, `pdm run tests`, then one commit for the package.
- **`docs/generated` is regenerated, never hand-edited** (FR-019). `docs/examples` and the user guide follow
  any public rename.

---

## Phase 1: Setup

- [ ] T001 Confirm the green baseline before starting: `pdm run formatting`, `pdm run checks`, `pdm run tests` all green on 3.11/3.12/3.13, and the browser installed with `pdm run python -m playwright install --only-shell chromium`. This is the state every package phase returns to.

**Checkpoint**: The baseline is green. The sweep can begin.

---

## Phase 2: sources (with `_params`) 🎯 the foundation

**Goal**: The foundation package conforms. It is imported by everything below, so it is swept first.

**Skip (already conforming)**: `sources/_base.py`, `sources/_resolver.py`, `sources/_schema.py`.

**Independent Test**: Every module under `sources` and `_params` has a one-line imperative module docstring,
every public function is verb-first, and `pdm run checks` and `pdm run tests` are green.

- [ ] T002 [US1] Reword the module docstrings of `src/sportsbet/_params.py` and `src/sportsbet/sources/__init__.py` to one imperative line (FR-005). `__init__` gets a one-liner, not `It provides ...`.
- [ ] T003 [P] [US1] Reword the module docstrings of the stats sources to one imperative line: `src/sportsbet/sources/_stats/_nba.py`, `_euroleague.py`, `_football_data.py`, `_sample.py`, and `_stats/__init__.py`.
- [ ] T004 [P] [US1] Reword the module docstrings of the odds sources to one imperative line: `src/sportsbet/sources/_odds/_football_data.py`, `_odds_api.py`, `_sample.py`, and `_odds/__init__.py`.
- [ ] T005 [US1] Reword the module docstring of `src/sportsbet/sources/_utils.py` (`Includes utilities ...` → imperative), and check `derive_market_outcomes` and any helper for verb-first names and single-line docstrings.
- [ ] T006 [US1] Read every `def` in the sources package for FR-001/FR-006/FR-010/FR-011 violations: noun-first or empty-verb names, private helpers with Args/Returns blocks, in-function imports that are not optional-extra deferrals, explanatory inline comments. Rename and collapse. Record any public rename in the ledger and propagate to `sources/__init__` `__all__`, callers, tests and docs.
- [ ] T007 [US2] Bring the sources tests to the mirror: `tests/sources/test_utils.py`, `test_schema.py`, `test_resolver.py`, `test_base.py`, and `tests/sources/stats/*`, `tests/sources/odds/*`. Rename each test to `test_<function>_<behavior>`, collapse docstrings to one line, fix the `test_stastics_schema` typo, and switch any private-module import to the public API.
- [ ] T008 [US3] Run the gate for sources: `pdm run formatting`, `pdm run checks`, `pdm run tests`. Add a `CHANGELOG.md` line for any public rename. Commit the sources package.

**Checkpoint**: sources conforms and the gate is green.

---

## Phase 3: dataloaders

**Goal**: The dataloaders conform. Imports sources.

**Skip (already conforming)**: `dataloaders/_schema.py`.

- [ ] T009 [US1] Reword the module docstrings of `src/sportsbet/dataloaders/_base.py` (`Implements the base dataloader ...`), `_sourced.py` (`Implements the dataloader ...`), and `__init__.py` (`It provides ...`) to one imperative line.
- [ ] T010 [US1] Read every non-contract `def` in `dataloaders/_base.py` and `_sourced.py` for FR-001/FR-006/FR-010/FR-011. Do NOT touch the contract surface (D1): `extract_train_data`, `extract_fixtures_data`, `extract_exploration_data`, `save`, `load_dataloader`, the fitted `*_` attributes, and constructor params stay. Rename any genuinely non-conforming private helper and collapse its docstring. Confirm `_base.py` imports no sibling (FR-009).
- [ ] T011 [US2] Bring the dataloaders tests to the mirror: `tests/dataloaders/test_base.py`, `test_sourced.py` (create at the mirrored path if the sourced dataloader is tested elsewhere), `test_extraction.py`, `test_leakage.py`, `test_basketball.py`, `test_soccer.py`, `test_sources_choice.py`. Name each `test_<function>_<behavior>`, one-line docstrings, public-API imports. Note: a behaviour test that spans several functions keeps a behaviour name but still starts from the entry point it exercises.
- [ ] T012 [US3] Run the gate for dataloaders, changelog any public rename, commit.

**Checkpoint**: dataloaders conforms and the gate is green.

---

## Phase 4: evaluation

**Goal**: The estimators conform where Principle I allows. Principle I constrains this package the most.

- [ ] T013 [US1] Reword the module docstrings of `src/sportsbet/evaluation/_base.py` and `_model_selection.py` (`Includes base class and functions ...`), `_classifier.py` and `_rules.py` (`Create a bettor ...` is already verb-first, tighten only), and `__init__.py` (`It provides ...`) to one imperative line.
- [ ] T014 [US1] Weigh `complementary_events`, a noun-first public name: if it is a function, rename to a verb phrase (e.g. `list_complementary_events`) and propagate to `evaluation/__all__`, callers, tests and docs, recording it in the ledger; if it is data, leave it. Do NOT rename the contract surface: `fit`, `predict`, `predict_proba`, `bet`, `score`, `get_params`, `set_params`, `backtest`, `load_bettor`, `save_bettor`, the classes, the fitted `*_` attrs, the constructor params.
- [ ] T015 [US1] Read the private helpers `latest_odds_column` and `market_base` in `evaluation/_base.py` and any other non-contract `def` for FR-001/FR-006. Collapse docstrings to one line. These two are tested directly, so export them from `sportsbet.evaluation` (the resolver-helper precedent) or plan to test them through the public API in T016.
- [ ] T016 [US2] Bring the evaluation tests to the mirror: `tests/evaluation/test_base.py`, `test_classifier.py`, `test_model_selection.py`, `test_rules.py`, and `tests/evaluation/__init__.py`. Switch `from sportsbet.evaluation._base import BaseBettor, latest_odds_column, market_base` to the public API (`BaseBettor` from `sportsbet.evaluation`; the helpers from wherever T015 exported them). Name each test `test_<function>_<behavior>`, one-line docstrings.
- [ ] T017 [US3] Run the gate for evaluation, changelog any public rename, commit.

**Checkpoint**: evaluation conforms, the contract surface intact, the gate green.

---

## Phase 5: execution

**Goal**: The execution package's tests conform. Its source is already conforming (recent refactor), so the
work is mostly the test side.

**Skip (already conforming source)**: `execution/_base.py`, `_browser.py`, `_credentials.py`, `_place.py`,
`_schedule.py`, `__init__.py` module docstring EXCEPT the `__init__.py` opening `It provides ...`.

- [ ] T018 [US1] Reword only the `src/sportsbet/execution/__init__.py` module docstring (`It provides ...`) to one imperative line. Confirm the rest of the package is already conforming by a quick re-scan (names verb-first, docstrings sized, `_base.py` self-contained).
- [ ] T019 [US2] Bring the execution tests to the mirror: `tests/execution/test_execution.py`, `test_schedule.py`, `test_browser.py`. Confirm each test is `test_<function>_<behavior>` with a one-line docstring and public-API imports, and that a test module maps to a source module (execution has `_place`, `_schedule`, `_browser`, `_base`, `_credentials`; align the test modules to those or keep behaviour-grouped modules named for the entry point they exercise).
- [ ] T020 [US3] Run the gate for execution, commit.

**Checkpoint**: execution conforms and the gate is green.

---

## Phase 6: selection & artifacts (the shared glue)

**Goal**: The top-level glue both surfaces share conforms, so cli and mcp import conforming names.

- [ ] T021 [US1] Reword the module docstrings of `src/sportsbet/_selection.py` (`Implements the selection ...`) and `src/sportsbet/_artifacts.py` (`Implements the file a surface writes ...`) to one imperative line, and the top `src/sportsbet/__init__.py` docstring if it narrates.
- [ ] T022 [US1] Read every `def` in `_selection.py` and `_artifacts.py` for FR-001/FR-006/FR-010/FR-011. `build_dataloader`, `build_bettor`, `build_venue` are verb-first (keep). Check `_load_object`, `_moments`, `_aliases`, `_odds_source` and the artifact save/load helpers for single-line docstrings and confirm the in-function `from .execution import ...` is a deferred optional extra with its `# noqa` (FR-010).
- [ ] T023 [US2] Bring the tests of the glue to the mirror: any test of `_selection`/`_artifacts` sits at a mirrored path (`tests/test_selection.py`, `tests/test_artifacts.py`) or, if these are exercised only through the CLI/MCP tests, note that and keep them there. Name and one-line them.
- [ ] T024 [US3] Run the gate for the glue, changelog any public rename, commit.

**Checkpoint**: the glue conforms and the gate is green.

---

## Phase 7: cli

**Goal**: The CLI surface conforms.

- [ ] T025 [US1] Reword the six `Module that contains ... of the CLI` docstrings to one imperative line: `src/sportsbet/cli/_cli.py`, `_data.py`, `_betting.py`, `_execution.py`, `_options.py`, `_utils.py`.
- [ ] T026 [US1] Read every `def` in the cli package for FR-001/FR-006/FR-010/FR-011, and confirm `sportsbet.cli` exports `main` publicly (it is imported privately by a test in T028). Rename non-conforming helpers, collapse docstrings.
- [ ] T027 [US2] Bring the cli tests to the mirror: `tests/cli/test_cli.py`, `test_data.py`, `test_betting.py`, `test_execution.py`, `test_main.py`, `test_parity.py`. Name each `test_<function>_<behavior>`, one-line docstrings, public-API imports.
- [ ] T028 [US3] Run the gate for cli, changelog any public rename, commit.

**Checkpoint**: cli conforms and the gate is green.

---

## Phase 8: mcp

**Goal**: The MCP surface conforms.

- [ ] T029 [US1] Reword the module docstring of `src/sportsbet/mcp/_server.py` (`Implements the server ...`) and `mcp/__init__.py` (`It provides ...`) to one imperative line. Read every `def` in `_server.py` for FR-001/FR-006/FR-010/FR-011 and confirm the tool functions and helpers conform.
- [ ] T030 [US2] Bring the mcp tests to the mirror: `tests/mcp/test_mcp.py` (mirror `_server.py`, so consider `tests/mcp/test_server.py`) and `test_parity_surfaces.py`. Switch `from sportsbet.cli._cli import main` to `from sportsbet.cli import main` (public). Name each test `test_<function>_<behavior>`, one-line docstrings.
- [ ] T031 [US3] Run the gate for mcp, changelog any public rename, commit.

**Checkpoint**: mcp conforms and the gate is green.

---

## Phase 9: Polish and verification

- [ ] T032 Regenerate `docs/generated` from `docs/examples` via the docs build and confirm it is not hand-edited. Confirm every runnable example still runs (SC-008) and every public name kept a runnable example (Principle V).
- [ ] T033 Sweep-wide verification of the success criteria: SC-001 (zero `Implements`/`It provides`/`Module that contains` outside history), SC-002 (every public function verb-first bar the contract surface), SC-003 (single-line docstrings bar public entry points/classes), SC-004 (no base imports a sibling, no function-body import bar deferred extras), SC-005 (tests mirror the source and are named `test_<function>_<behavior>`).
- [ ] T034 Confirm `contracts/rename-ledger.md` and `CHANGELOG.md` agree: every public rename is in both, with before and after (SC-007). Final full gate green on 3.11/3.12/3.13.

---

## Dependencies & Execution Order

- **Setup (Phase 1)**: no dependency. Confirms the baseline.
- **Packages (Phases 2 to 8)**: bottom-up. sources → dataloaders → evaluation → execution → glue → cli → mcp.
  A later package depends on the renames of an earlier one only in that they are already final; each commit is
  self-contained and green.
- **Polish (Phase 9)**: after every package is done.

### Within a package

Reword module docstrings → rename functions and propagate → collapse helper docstrings and structure →
mirror and rename tests → gate → commit. The source (US1) conforms before its tests (US2) so the tests import
the final names; US3 (gate + commit) closes the package.

### Parallel opportunities

- T003 and T004 (stats and odds source docstrings) are independent files.
- The seven package phases are sequential by policy (one commit each, bottom-up), not because they cannot be
  parallelized; keeping them sequential keeps each diff reviewable and each gate meaningful.

---

## Implementation strategy

The MVP is Phase 2 (sources): the foundation conforms and proves the recipe end to end. Each later package is
an independent increment that lands green and can stop the sweep at a clean boundary. The contract surface
(research D1) is the one thing every phase must leave untouched.
