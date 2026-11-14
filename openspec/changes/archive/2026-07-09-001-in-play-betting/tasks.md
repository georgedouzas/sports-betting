# Tasks

## 1. Setup

- [x] 1.1 Establish the baseline: run `pdm run tests` and record the current failures (broken `tests/conftest.py` CLI
  `CONFIG`, missing `datasets` exports, unimplemented `extract_fixtures_data`) as the starting state in the quickstart
  "Definition of done" notes (T001)
- [x] 1.2 Confirm the runtime dependencies `pandera[pandas]` and `cloudpickle` resolve in the active environment
  (present in `pyproject.toml`), with no code change if already satisfied (T002)

## 2. Foundational

- [x] 2.1 Implement the shared `X`/`Y`/`O` column naming grammar helpers (fixed as bare name, time-varying as
  `{col}__{status}__{time}`, odds as `{provider}__{col}__{status}__{time}`, targets as
  `{col}__{target_status}__{target_time}`, `event_time` rendered as `{n}min`) in
  `src/sportsbet/datasets/_base/_dataloader.py`, exposed so extraction and bettors import one definition (T003)
- [x] 2.2 Finalize the base schema metadata and helpers in `src/sportsbet/datasets/_base/_schema.py`: `snapshot_cols()`,
  `col_metadata()` (returning safe `include`/`fixed` defaults for required and snapshot columns), and dataframe checks
  for event status and time consistency, snapshot uniqueness, and null post-match odds (T004)
- [x] 2.3 Verify and complete the offline in-play sample data in `tests/samples/stats.csv` and `tests/samples/odds.csv`
  and the `stats`/`odds`/`stats_schema`/`odds_schema` fixtures in `tests/conftest.py`, with several event statuses and
  in-play times present (T005)
- [x] 2.4 Establish the public API export skeleton in `src/sportsbet/datasets/__init__.py`: `BaseDataLoader`,
  `BaseStatsSchema`, `BaseOddsSchema`, `required_col`, `optional_col`, `load_dataloader` (concrete loaders added in
  their story phases) (T006)

## 3. User Story 1: Moment-aware extraction

- [x] 3.1 Extend `tests/datasets/base/test_dataloader.py`: supervised extraction at `postplay` returns aligned
  `X`/`Y`/`O` whose columns follow the naming grammar (fixed bare, varying suffixed) (T007)
- [x] 3.2 Add an in-play target case (`target_event_status='inplay'`, `target_event_time=60min`) to
  `tests/datasets/base/test_dataloader.py` asserting no post-60-minute columns appear in `X` and `Y` is evaluated at 60
  minutes (T008)
- [x] 3.3 Add mode tests to `tests/datasets/base/test_dataloader.py`: unsupervised returns `(X, None, O)` (uniform
  three-tuple, `Y=None`), and `learning_type='reinforcement'` (and any other unknown value) raises `ValueError` (T009)
- [x] 3.4 Extend `tests/datasets/base/test_schema.py`: invalid `event_status`, type or duplicate snapshot raises
  `pandera.errors.SchemaError`, non-null post-match odds raise an error, and a stats/odds snapshot column mismatch
  raises `AssertionError` (T010)
- [x] 3.5 Complete `BaseDataLoader.extract_train_data(learning_type, target_event_status, target_event_time)` pivot
  logic and mode handling in `src/sportsbet/datasets/_base/_dataloader.py`, using the shared grammar and metadata.
  `learning_type` accepts only `'supervised'`/`'unsupervised'` and always returns `(X, Y, O)` with `Y=None` when
  unsupervised. Remove `'reinforcement'` from the accepted set and drop the RL return description from the docstring
  (T011)
- [x] 3.6 Implement `BaseDataLoader.extract_fixtures_data()` in `src/sportsbet/datasets/_base/_dataloader.py`
  (previously `pass`): reproduce the training column layout, return `(X, None, O)`, and require a prior
  `extract_train_data` call to fix columns (T012)
- [x] 3.7 Add input validation and clear errors in `src/sportsbet/datasets/_base/_dataloader.py`: no resolvable events
  `ValueError`, invalid `learning_type`/`target_event_status`, negative `target_event_time`, and the snapshot column
  match assertion (T013)

## 4. User Story 2: Backtest and predict value bets

- [x] 4.1 Add `tests/evaluation/test_classifier.py` cases: `ClassifierBettor.fit/predict/bet` consume the new
  `X`/`Y`/`O` (from base loader and fixtures) without manual reshaping (T014)
- [x] 4.2 Add `tests/evaluation/test_rules.py` cases: `OddsComparisonBettor` parses markets and odds types from the new
  column grammar (T015)
- [x] 4.3 Add `tests/evaluation/test_model_selection.py` cases: `backtest` returns per-period performance on
  moment-aware data, and `BettorGridSearchCV` fits over it (T016)
- [x] 4.4 Update `BaseBettor` odds and feature column parsing (`_get_feature_names_odds`, `_validate_X_O`,
  `_validate_X_Y`) to the shared grammar in `src/sportsbet/evaluation/_base.py` (CR-2, CR-3) (T017)
- [x] 4.5 Adapt `ClassifierBettor` to the new `X`/`Y`/`O` in `src/sportsbet/evaluation/_classifier.py` (T018)
- [x] 4.6 Adapt `OddsComparisonBettor._check_odds_types` and market handling in `src/sportsbet/evaluation/_rules.py`
  (T019)
- [x] 4.7 Adapt `backtest` and `BettorGridSearchCV` to the new columns in `src/sportsbet/evaluation/_model_selection.py`
  (T020)
- [x] 4.8 Ensure `Y` and `O` market reconciliation for value-bet identification across all bettors (CR-3), verified
  against sample fixtures (T021)

## 5. User Story 3: Selection interface, concrete and sample loaders, persistence

- [x] 5.1 Add `tests/datasets/test_soccer.py`: `SoccerDataLoader.get_all_params()`/`get_odds_types()` (network
  mocked), `param_grid` selection, `drop_na_thres`, `odds_type`, and `target_event_status`/`target_event_time`
  passthrough (T022)
- [x] 5.2 Restore `tests/datasets/test_dummy.py`: the full offline workflow via `DummySoccerDataLoader` (train,
  fixtures, save and reload column consistency, in-play target) (T023)
- [x] 5.3 Implement `SoccerStatsSchema` and `SoccerOddsSchema` in `src/sportsbet/datasets/_soccer/_schema.py`
  (previously empty) (T024)
- [x] 5.4 Re-home the outcome and odds mapping helpers in `src/sportsbet/datasets/_soccer/_utils.py` (restore deleted
  utilities needed by the loader) (T025)
- [x] 5.5 Conform `SoccerDataLoader` to the new base in `src/sportsbet/datasets/_soccer/_dataloader.py`: add
  `__init__(param_grid=None)`, `get_all_params()`, `get_odds_types()`, `drop_na_thres`/`odds_type` extraction options,
  build validated `stats`/`odds` and schemas from the feed, delegate the pivot to the base and accept
  `target_event_status`/`target_event_time`, and remove the stale `_stages` string model and empty `_get_*` stubs
  (T026)
- [x] 5.6 Restore `DummySoccerDataLoader` producing in-play sample snapshots with no network in
  `src/sportsbet/datasets/_dummy.py`, sharing the `SoccerDataLoader` interface (T027)
- [x] 5.7 Re-export `SoccerDataLoader` and `DummySoccerDataLoader` (and confirm `load_dataloader`) in
  `src/sportsbet/datasets/__init__.py` (T028)
- [x] 5.8 Verify the `save()`/`load_dataloader()` round trip reproduces identical fixtures columns (FR-015) in
  `src/sportsbet/datasets/_base/_dataloader.py` and cover it in `tests/datasets/test_dummy.py` (T029)

## 6. User Story 4: CLI and GUI parity

- [x] 6.1 Fix the embedded CLI `CONFIG` in `tests/conftest.py` to use the restored `DummySoccerDataLoader` and add
  `TARGET_EVENT_STATUS`/`TARGET_EVENT_TIME` keys (T030)
- [x] 6.2 Add CLI tests in `tests/test_cli.py` asserting extraction and backtest via the CLI match the equivalent API
  results (T031)
- [x] 6.3 Add `TARGET_EVENT_STATUS`/`TARGET_EVENT_TIME` config support and the matching options in
  `src/sportsbet/cli/_options.py` and `src/sportsbet/cli/_data.py` (T032)
- [x] 6.4 Wire moment-aware extraction, backtest and value-bet output through the CLI commands in
  `src/sportsbet/cli/_data.py` and `src/sportsbet/cli/_betting.py` (T033)
- [x] 6.5 Surface target moment selection (status and in-play time) in the GUI in
  `src/sportsbet/gui/app/dataloader_creation.py` and `src/sportsbet/gui/app/model_creation.py` (T034)

## 7. Polish and cross-cutting concerns

- [x] 7.1 Update the user guide `docs/overview/user_guide/dataloader.md` and `docs/overview/user_guide/bettor.md` to the
  snapshot and in-play model (T035)
- [x] 7.2 Update the gallery examples `docs/examples/plot_soccer_data.py` and
  `docs/examples/modelling/plot_classifier_bettor.py` to the new API and confirm they render (T036)
- [x] 7.3 Add a `CHANGELOG.md` entry describing the in-play data model, schema validation and the API changes (T037)
- [x] 7.4 Ensure all docstring examples run under `--doctest-modules`, including the in-play extraction example. Add no
  runnable reinforcement doctest, since RL is design only, and keep any reference to it as prose or `+SKIP` (T038)
- [x] 7.5 Execute the quickstart scenarios 1 to 7 offline against `DummySoccerDataLoader` and confirm the expected
  outcomes (T039)
- [x] 7.6 Run the full gate, `pdm run formatting`, `pdm run checks` and `pdm run tests`, and confirm no new
  suppressions and no coverage regression (SC-006) (T040)
