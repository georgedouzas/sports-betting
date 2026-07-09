---
description: "Task list for In-Play (Live) Betting Support"
---

# Tasks: In-Play (Live) Betting Support

**Input**: Design documents from `/specs/001-in-play-betting/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: INCLUDED. The project constitution (Principle III — Test Coverage & Doctest
Discipline) makes tests non-negotiable, and every user story defines an Independent Test.
Test tasks are therefore generated per story and precede their implementation.

**Organization**: Tasks grouped by user story (from spec.md) for independent implementation
and testing. Priorities: US1=P1, US2=P2, US3=P2, US4=P3.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1/US2/US3/US4 (story phases only)
- All paths are repository-relative from repo root.

## Path Conventions

Single Python package: source in `src/sportsbet/`, tests in `tests/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish a known baseline for the in-flight refactor.

- [X] T001 Establish baseline: run `pdm run tests` and record the current failures (broken `tests/conftest.py` CLI `CONFIG`, missing `datasets` exports, unimplemented `extract_fixtures_data`) as the starting state in `specs/001-in-play-betting/quickstart.md` "Definition of done" notes.
- [X] T002 [P] Confirm runtime dependencies `pandera[pandas]` and `cloudpickle` resolve in the active environment (present in `pyproject.toml`); no code change if already satisfied.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared contracts every user story depends on — the column-naming grammar
(produced by US1, consumed by US2), the base schema helpers, the offline sample data, and
the public-API skeleton.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T003 Implement the shared `X`/`Y`/`O` column-naming grammar helpers (fixed → bare name; time-varying → `{col}__{status}__{time}`; odds → `{provider}__{col}__{status}__{time}`; targets → `{col}__{target_status}__{target_time}`; `event_time` rendered as `{n}min`) in `src/sportsbet/datasets/_base/_dataloader.py`, exposed so both extraction and bettors import one definition (per research.md R4).
- [X] T004 [P] Finalize base schema metadata + helpers in `src/sportsbet/datasets/_base/_schema.py`: `snapshot_cols()`, `col_metadata()` (returning safe `include`/`fixed` defaults for required/snapshot columns), and dataframe checks for event-status/time consistency, snapshot uniqueness, and post-match null-odds.
- [X] T005 [P] Verify/complete the offline in-play sample data in `tests/samples/stats.csv` and `tests/samples/odds.csv` and the `stats`/`odds`/`stats_schema`/`odds_schema` fixtures in `tests/conftest.py` (multiple event statuses and in-play times present).
- [X] T006 Establish the public-API export skeleton in `src/sportsbet/datasets/__init__.py`: `BaseDataLoader`, `BaseStatsSchema`, `BaseOddsSchema`, `required_col`, `optional_col`, `load_dataloader` (concrete loaders added in their story phases).

**Checkpoint**: Shared column grammar, schemas, sample data, and base exports ready.

---

## Phase 3: User Story 1 - Moment-aware extraction (Priority: P1) 🎯 MVP

**Goal**: A user can extract `X`/`Y`/`O` for any chosen target moment (pre-match or in-play),
with no information later than the target leaking into features.

**Independent Test**: Using the `BaseDataLoader` with the sample `stats`/`odds` fixtures,
request an in-play target at 60 minutes and confirm `X` excludes all later snapshots, `Y`
reflects the 60-minute state, and `X`/`Y`/`O` share the same rows.

### Tests for User Story 1 ⚠️ (write first, ensure they fail)

- [X] T007 [P] [US1] Extend `tests/datasets/base/test_dataloader.py`: supervised extraction at `postplay` returns aligned `X`/`Y`/`O` whose columns follow the naming grammar (fixed bare, varying suffixed).
- [X] T008 [P] [US1] Add an in-play target case (`target_event_status='inplay'`, `target_event_time=60min`) to `tests/datasets/base/test_dataloader.py` asserting no post-60-minute columns appear in `X` and `Y` is evaluated at 60 minutes.
- [X] T009 [P] [US1] Add mode tests to `tests/datasets/base/test_dataloader.py`: unsupervised returns `(X, None, O)` (uniform three-tuple, `Y=None`); `learning_type='reinforcement'` (and any other unknown value) raises `ValueError` (RL is not a valid extraction mode — research.md R1).
- [X] T010 [P] [US1] Extend `tests/datasets/base/test_schema.py`: invalid `event_status`/type/duplicate snapshot → `pandera.errors.SchemaError`; non-null post-match odds → error; stats/odds snapshot-column mismatch → `AssertionError`.

### Implementation for User Story 1

- [X] T011 [US1] Complete/correct `BaseDataLoader.extract_train_data(learning_type, target_event_status, target_event_time)` pivot logic and mode handling in `src/sportsbet/datasets/_base/_dataloader.py`, using the T003 grammar and T004 metadata. `learning_type` accepts only `'supervised'`/`'unsupervised'` (always returns a uniform three-tuple `(X, Y, O)`, with `Y=None` when unsupervised); remove `'reinforcement'` from the accepted set so it is rejected as invalid (research.md R1). Update the method docstring to drop the RL/gym return description.
- [X] T012 [US1] Implement `BaseDataLoader.extract_fixtures_data()` in `src/sportsbet/datasets/_base/_dataloader.py` (currently `pass`): reproduce the training column layout, return `(X, None, O)`, require a prior `extract_train_data` call to fix columns.
- [X] T013 [US1] Add input validation and clear errors in `src/sportsbet/datasets/_base/_dataloader.py`: no-resolvable-events `ValueError`, invalid `learning_type`/`target_event_status`, negative `target_event_time`, and the snapshot-column-match assertion.

**Checkpoint**: Moment-aware extraction fully functional and testable via the base loader
and sample fixtures — MVP.

---

## Phase 4: User Story 2 - Backtest & predict value bets (Priority: P2)

**Goal**: A user can backtest a bettor on moment-aware training data and identify value bets
on fixtures, for pre-match or in-play targets, without reshaping the data.

**Independent Test**: With `X`/`Y`/`O` from the base loader + sample fixtures, fit a bettor,
backtest it (receive per-period performance), then obtain value-bet selections for a
fixtures extraction.

### Tests for User Story 2 ⚠️ (write first, ensure they fail)

- [X] T014 [P] [US2] Add `tests/evaluation/test_classifier.py` cases: `ClassifierBettor.fit/predict/bet` consume the new `X`/`Y`/`O` (from base loader + fixtures) without manual reshaping.
- [X] T015 [P] [US2] Add `tests/evaluation/test_rules.py` cases: `OddsComparisonBettor` parses markets/odds-types from the new column grammar.
- [X] T016 [P] [US2] Add `tests/evaluation/test_model_selection.py` cases: `backtest` returns per-period performance on moment-aware data; `BettorGridSearchCV` fits over it.

### Implementation for User Story 2

- [X] T017 [US2] Update `BaseBettor` odds/feature-column parsing (`_get_feature_names_odds`, `_validate_X_O`, `_validate_X_Y`) to the T003 grammar in `src/sportsbet/evaluation/_base.py` (CR-2/CR-3).
- [X] T018 [P] [US2] Adapt `ClassifierBettor` to the new `X`/`Y`/`O` in `src/sportsbet/evaluation/_classifier.py`.
- [X] T019 [P] [US2] Adapt `OddsComparisonBettor._check_odds_types` and market handling in `src/sportsbet/evaluation/_rules.py`.
- [X] T020 [US2] Adapt `backtest` and `BettorGridSearchCV` to the new columns in `src/sportsbet/evaluation/_model_selection.py`.
- [X] T021 [US2] Ensure `Y`↔`O` market reconciliation for value-bet identification across all bettors (CR-3), verified against sample fixtures.

**Checkpoint**: End-to-end backtest and value-bet workflow works on moment-aware data.

---

## Phase 5: User Story 3 - Selection interface, concrete & sample loaders, persistence (Priority: P2)

**Goal**: The familiar scikit-learn-style selection UX works on the new model, a real
`SoccerDataLoader` and an offline `DummySoccerDataLoader` exist, and loaders save/reload with
consistent columns.

**Independent Test**: Query `get_all_params()`/`get_odds_types()` offline, construct with a
`param_grid`, extract, save, reload, and confirm reloaded fixtures columns match the
original.

### Tests for User Story 3 ⚠️ (write first, ensure they fail)

- [X] T022 [P] [US3] Add `tests/datasets/test_soccer.py`: `SoccerDataLoader.get_all_params()`/`get_odds_types()` (network mocked), `param_grid` selection, `drop_na_thres`, `odds_type`, and `target_event_status`/`target_event_time` passthrough.
- [X] T023 [P] [US3] Restore `tests/datasets/test_dummy.py`: full offline workflow via `DummySoccerDataLoader` (train + fixtures + save/reload column consistency, in-play target).

### Implementation for User Story 3

- [X] T024 [P] [US3] Implement `SoccerStatsSchema` and `SoccerOddsSchema` in `src/sportsbet/datasets/_soccer/_schema.py` (currently empty).
- [X] T025 [P] [US3] Re-home outcome/odds mapping helpers in `src/sportsbet/datasets/_soccer/_utils.py` (restore deleted utilities needed by the loader).
- [X] T026 [US3] Conform `SoccerDataLoader` to the new base in `src/sportsbet/datasets/_soccer/_dataloader.py`: add `__init__(param_grid=None)`, `get_all_params()`, `get_odds_types()`, `drop_na_thres`/`odds_type` extraction options; build validated `stats`/`odds` + schemas from the feed; delegate the pivot to the base and accept `target_event_status`/`target_event_time`; remove the stale `_stages` string model and empty `_get_*` stubs.
- [X] T027 [US3] Restore `DummySoccerDataLoader` producing in-play sample snapshots with no network in `src/sportsbet/datasets/_dummy.py`, sharing `SoccerDataLoader`'s interface.
- [X] T028 [US3] Re-export `SoccerDataLoader` and `DummySoccerDataLoader` (and confirm `load_dataloader`) in `src/sportsbet/datasets/__init__.py`.
- [X] T029 [US3] Verify `save()`/`load_dataloader()` round-trip reproduces identical fixtures columns (FR-015) in `src/sportsbet/datasets/_base/_dataloader.py` and cover it in `tests/datasets/test_dummy.py`.

**Checkpoint**: Concrete + sample loaders, selection UX, and persistence complete; doctests
can run offline.

---

## Phase 6: User Story 4 - CLI & GUI parity (Priority: P3)

**Goal**: The CLI and GUI expose moment-aware extraction, backtesting, and value-bet
identification equivalent to the API.

**Independent Test**: Run a moment-targeted extraction + backtest through the CLI against the
sample loader and confirm results match the equivalent API workflow.

### Tests for User Story 4 ⚠️ (write first, ensure they fail)

- [X] T030 [P] [US4] Fix the embedded CLI `CONFIG` in `tests/conftest.py` to use the restored `DummySoccerDataLoader` and add `TARGET_EVENT_STATUS`/`TARGET_EVENT_TIME` keys.
- [X] T031 [P] [US4] Add CLI tests in `tests/` (e.g. `tests/test_cli.py`) asserting extraction + backtest via the CLI match the equivalent API results.

### Implementation for User Story 4

- [X] T032 [US4] Add `TARGET_EVENT_STATUS`/`TARGET_EVENT_TIME` config support and corresponding options in `src/sportsbet/cli/_options.py` and `src/sportsbet/cli/_data.py`.
- [X] T033 [US4] Wire moment-aware extraction, backtest, and value-bet output through the CLI commands in `src/sportsbet/cli/_data.py` and `src/sportsbet/cli/_betting.py`.
- [X] T034 [US4] Surface target-moment selection (status + in-play time) in the GUI in `src/sportsbet/gui/app/dataloader_creation.py` and `src/sportsbet/gui/app/model_creation.py`.

**Checkpoint**: All surfaces reach the same capabilities (SC-005).

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, examples, changelog, and full-gate validation (Principles III–V).

- [X] T035 [P] Update the user guide `docs/overview/user_guide/dataloader.md` and `docs/overview/user_guide/bettor.md` to the snapshot/in-play model (they currently describe the old design).
- [X] T036 [P] Update gallery examples `docs/examples/plot_soccer_data.py` and `docs/examples/modelling/plot_classifier_bettor.py` to the new API and confirm they render.
- [X] T037 [P] Add a `CHANGELOG.md` entry describing the in-play data model, schema validation, and the API changes.
- [X] T038 Ensure all docstring examples run under `--doctest-modules`, including the in-play extraction example (per contracts). Do not add a runnable reinforcement doctest — RL is design-only; if referenced in docs it must be prose or `+SKIP`.
- [X] T039 Execute quickstart.md scenarios 1–7 offline against `DummySoccerDataLoader` and confirm expected outcomes.
- [X] T040 Run the full gate: `pdm run formatting`, `pdm run checks`, `pdm run tests`; confirm no new suppressions and coverage does not regress (SC-006).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: depends on Setup — BLOCKS all user stories.
- **US1 (Phase 3)**: depends on Foundational. MVP.
- **US2 (Phase 4)**: depends on Foundational + T003 grammar; needs US1's extraction output to test end-to-end (uses base loader + fixtures, not the concrete loader — stays independent of US3).
- **US3 (Phase 5)**: depends on Foundational + US1 (delegates to the base engine). Independent of US2.
- **US4 (Phase 6)**: depends on US3 (needs the restored `DummySoccerDataLoader`) and US2 (backtest via CLI).
- **Polish (Phase 7)**: depends on all targeted stories.

### User Story Dependencies

- **US1 (P1)**: independent — testable via base loader + sample fixtures.
- **US2 (P2)**: consumes US1's `X`/`Y`/`O` contract; does not require US3.
- **US3 (P2)**: builds the concrete/sample loaders on the US1 base engine; independent of US2.
- **US4 (P3)**: integrates US2 (backtest) and US3 (sample loader) through CLI/GUI.

### Within Each User Story

- Tests written first and failing → implementation.
- Schemas/grammar before extraction; extraction before bettors; loaders before CLI/GUI.

### Parallel Opportunities

- T002 alongside T001.
- Foundational: T004 and T005 in parallel (T003, T006 touch `_dataloader.py`/`__init__.py`).
- US1 tests T007–T010 in parallel; US2 tests T014–T016 in parallel; US3 tests T022–T023 in parallel; US4 tests T030–T031 in parallel.
- US2 impl: T018 and T019 in parallel (different files) after T017; US3 impl: T024 and T025 in parallel before T026.
- US2 and US3 can be developed in parallel by different developers once US1 is done.
- Polish: T035, T036, T037 in parallel.

---

## Parallel Example: User Story 1

```bash
# Launch US1 tests together (write first, expect failure):
Task: "Extend supervised-extraction test in tests/datasets/base/test_dataloader.py"   # T007
Task: "Add in-play 60min target test in tests/datasets/base/test_dataloader.py"        # T008
Task: "Add unsupervised/reinforcement mode tests in tests/datasets/base/test_dataloader.py"  # T009
Task: "Extend schema-rejection tests in tests/datasets/base/test_schema.py"            # T010
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 Setup → 2. Phase 2 Foundational → 3. Phase 3 US1.
4. **STOP and VALIDATE**: in-play/pre-match extraction correct and schema-validated on
   sample fixtures (quickstart Scenarios 1–3, 7).
5. This alone delivers moment-aware, validated data extraction.

### Incremental Delivery

1. Setup + Foundational → foundation ready.
2. US1 → validated moment-aware extraction (MVP).
3. US2 → backtest & value bets on the new data.
4. US3 → concrete + offline loaders, selection UX, persistence (unlocks offline doctests).
5. US4 → CLI/GUI parity.
6. Polish → docs, examples, changelog, full-gate green.

### Parallel Team Strategy

After US1: Developer A on US2 (evaluation), Developer B on US3 (datasets loaders). US4 and
Polish follow once both land.

---

## Notes

- [P] = different files, no incomplete-task dependencies.
- The single riskiest coupling is the column-naming grammar (T003): US1 produces it and US2
  consumes it — keep one shared definition to avoid drift (research.md R4).
- Reinforcement learning is removed from `extract_train_data` and is design-only this feature
  (a separate future `make_env()` method — research.md R1); `learning_type` is
  supervised/unsupervised only. Do not add RL code or a stub method here.
- Real `SoccerDataLoader` feed remains pre/post-match only; in-play is validated on sample
  data ("engine now, data later", research.md R2).
- Commit after each task or logical group; stop at any checkpoint to validate a story.
