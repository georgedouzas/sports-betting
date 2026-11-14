# Design

## Context

The `development` branch held an in-flight refactor under `src/sportsbet/datasets/_base/` and `_soccer/`. It had long
format snapshots and `pandera` schemas, but `extract_fixtures_data` was a stub, `SoccerDataLoader` still used a stale
`_stages` string model, the dummy loader and its utilities were deleted, and `tests/conftest.py` imported the deleted
loader. The bettors parsed odds columns by the legacy `odds__market__outcome__market_type` convention. This change
completes the refactor in the single `src/sportsbet` package and carries the new column contract into the evaluation
layer, the CLI and the GUI.

Technical context: Python `>=3.11, <3.14`. scikit-learn for the estimator contract, pandas, `pandera[pandas]` for
validation, aiohttp for download, cloudpickle for persistence, click and rich for the CLI, and reflex for the optional
GUI. Data comes from remote CSVs on the repository's `data` branch and from bundled sample CSVs. There is no database.
Tests run with pytest, branch coverage, `--doctest-modules`, pytest-randomly and xdist. Soccer is the only sport.

## Goals / Non-Goals

**Goals:**

- Extract `X`, `Y` and `O` for a chosen target moment, pre-match or in-play, with no leakage of later information.
- Validate every public statistics and odds frame against a schema before extraction.
- Keep the scikit-learn style selection interface for existing users.
- Let the existing bettors consume the new tables without manual reshaping.
- Run the full workflow offline on a sample dataloader so every doc example is a doctest.
- Reach the same capabilities through the API, the CLI and the GUI.

**Non-Goals:**

- Reinforcement learning. It is designed below as a separate future method and is not implemented.
- In-play data for the real soccer feed. The feed has pre-match and post-match data only.
- Sports other than soccer.
- A new data provider. The current upstream and its access mechanism are reused.
- A redesign of the bettor strategies. Their public interface is kept and only their input format changes.

## Decisions

### R1. Reinforcement learning is not an extraction mode

Decision: `learning_type` accepts only `'supervised'` (the default) and `'unsupervised'`. `extract_train_data` always
returns the three positions `(X, Y, O)`, with `Y` set to `None` when unsupervised, so every caller can write
`X, Y, O = extract_train_data(...)`. `'reinforcement'` is rejected like any other invalid value. No reinforcement code
ships, not even a stub method.

Rationale: a reinforcement agent needs a stateful environment with `reset()` and `step()`. The extraction method returns
a static table for a scikit-learn style bettor. A method that returns either a tuple or an environment by a string
argument has a polymorphic return type that cannot be annotated, cannot be unpacked, and breaks the estimator contract.
The environment was undesigned in the source branch, `gymnasium` is not a dependency, and the real feed has no in-play
data to train on.

Alternatives considered: keep `learning_type='reinforcement'` returning an environment, rejected for the polymorphic
return. Add a `NotImplementedError` stub now, rejected by product decision because it adds an untested path and an API
commitment. Build a full gym environment now, rejected as large and not testable with current data.

Future contract, design only: a dataloader method `make_env(...)` (name provisional) returns a Gymnasium compatible
environment, with `gymnasium` as an optional extra. The observation is the snapshot feature state at the current moment
as the episode advances from `preplay` through `inplay` to `postplay`. The action is bet or no bet per market, with an
optional stake fraction. The reward is the realized return settled at `postplay`, optionally shaped per snapshot. An
episode is one match. An external agent consumes it, not the bettor classes. It reuses the same schemas and snapshot
model, and its validation waits for an in-play data source.

Clarified: where does reinforcement learning live and what ships now? -> It is removed from `extract_train_data` and
specified as a separate future method `make_env()`, with design only and no code in this feature.

### R2. Engine now, data later

Decision: build and validate the in-play engine on bundled synthetic sample data (`tests/samples/stats.csv`,
`tests/samples/odds.csv` and the sample dataloader). `SoccerDataLoader` maps the real feed into `preplay` and `postplay`
snapshots only. In-play targets on the real feed return no rows until an in-play source exists.

Rationale: a confirmed product decision. The sample data holds in-play snapshots at several event times, which is enough
to exercise and doctest the engine offline.

Alternatives considered: block on a real in-play feed, rejected because none exists. Fabricate in-play rows in the real
loader, rejected because the data would mislead.

### R3. The soccer loader keeps its selection interface

Decision: `SoccerDataLoader` keeps `param_grid`, `get_all_params()`, `get_odds_types()`, `drop_na_thres` and
`odds_type`, layered on the snapshot model. It builds the validated `stats` and `odds` frames and the concrete schemas
from the feed, then delegates the pivot to the base engine and accepts `target_event_status` and `target_event_time`.
The base class keeps its low level `(stats, odds, stats_schema, odds_schema, targets)` constructor for direct use.

Rationale: scikit-learn compatibility and `ParameterGrid` parity, with minimal disruption to users and docs.

Alternatives considered: a clean-break constructor-only API, rejected because it breaks users and orphans the guide. Two
parallel APIs, rejected for duplicate surfaces and maintenance.

### R4. One column naming grammar links extraction and bettors

Decision: extraction emits and bettors parse one flattened grammar. Fixed columns keep a bare name such as `league`.
Time-varying features are `{col}__{event_status}__{event_time}`, such as `home_goals__inplay__60min`. Odds are
`{provider}__{col}__{event_status}__{event_time}`, such as `bet365__home_win__preplay__0min`. Targets are
`{col}__{target_event_status}__{target_event_time}`. The delimiter is a double underscore and `event_time` renders as
whole minutes `{n}min`. One shared definition serves both extraction and bettors.

Rationale: the bettors infer markets by parsing column names, so a documented grammar is the smallest testable bridge.

Alternatives considered: MultiIndex columns, rejected for doctest readability and scikit-learn compatibility. A metadata
sidecar mapping columns to markets, rejected as more parts to keep aligned across save and reload.

### R5. Schema metadata drives the pivot

Decision: keep the `pandera` `DataFrameModel` approach. `required_col()` and `optional_col(include, fixed)` fields carry
`snapshot`, `include` and `fixed` metadata. The helpers `snapshot_cols()` and `col_metadata()` read it. Dataframe checks
enforce event status and time consistency, snapshot uniqueness, and null post-match odds. Soccer schemas subclass the
base statistics and odds schemas.

Rationale: already built and tested for the base, and the metadata parameterizes the pivot directly.

Alternatives considered: hand-rolled validation, rejected because the constitution requires schemas over ad hoc checks.

### R6. Restore the sample dataloader

Decision: restore `DummySoccerDataLoader` with in-play sample snapshots in the shape of `tests/samples/*.csv`, no
network access, and the `SoccerDataLoader` selection interface. It backs the doctests, the CLI test configuration and
the docs.

Rationale: an offline end-to-end path is required, and `tests/conftest.py` still referenced the deleted loader.

Alternatives considered: network-dependent examples, rejected because they break offline runs and CI determinism.

### R7. Persistence stays on cloudpickle

Decision: keep `save` and `load_dataloader` on the base and the existing `save_bettor` and `load_bettor`. Re-export
`load_dataloader` from `sportsbet.datasets`.

Rationale: a proven mechanism. Only the missing export needed restoring. No alternative was warranted.

### Data model

- Snapshot identity: `event_status` is one of `preplay`, `inplay` and `postplay`. `event_time` is a time delta.
  `preplay` requires `event_time >= 0`, `inplay` requires `event_time > 0`, and `postplay` requires `event_time == 0`.
  For soccer, `date`, `league`, `division`, `season`, `home_team` and `away_team` complete the identity. No two rows
  share an identity.
- Statistics: identity columns, feature columns with `include` and `fixed` metadata, and the target-source columns named
  by the loader's `targets`, such as `home_goals` and `away_goals`.
- Odds: identity columns plus `provider`, and decimal odds columns per market such as `home_win` and `away_win`.
  `postplay` odds are null. The odds snapshot columns match the statistics snapshot columns.
- Schemas: `BaseSchema` holds `event_status`, `event_time`, the checks, `snapshot_cols()` and `col_metadata()`, with
  `Config.strict=True`. `BaseStatsSchema` is a marker. `BaseOddsSchema` adds `odds_cols()` and the null post-match odds
  check. `SoccerStatsSchema` and `SoccerOddsSchema` add the soccer identity, features, goal targets and market odds.
- Fitted state on the dataloader: `param_grid_`, `drop_na_thres_`, `odds_type_`, `learning_type_`,
  `target_event_status_`, `target_event_time_`, `input_cols_`, `output_cols_` and `odds_cols_`.
- `extract_fixtures_data` requires a prior `extract_train_data` call, which fixes the column layout.

### Contracts

- `BaseDataLoader.extract_train_data(learning_type=None, target_event_status=None, target_event_time=None)` defaults to
  supervised, `postplay` and zero time. It validates both frames, asserts the snapshot columns match, checks that
  resolvable events exist, and pivots. It raises `pandera.errors.SchemaError` for invalid data, `ValueError` for no
  resolvable events, an invalid `learning_type`, an invalid `target_event_status` or a negative time, and
  `AssertionError` for a snapshot column mismatch.
- `SoccerDataLoader.extract_train_data(drop_na_thres=0.0, odds_type=None, target_event_status=None,
  target_event_time=None)`. `drop_na_thres` is in `[0.0, 1.0]`. `odds_type=None` returns no odds.
- Bettors keep `fit(X, Y, O=None)`, `predict_proba(X)`, `predict(X)`, `bet(X, O)`, `score(X, Y, O)` and `classes_`.
  `sportsbet.evaluation` exports `BaseBettor`, `ClassifierBettor`, `OddsComparisonBettor`, `BettorGridSearchCV`,
  `backtest`, `save_bettor` and `load_bettor`.
- The CLI is driven by a Python config module with `DATALOADER_CLASS`, `PARAM_GRID`, `DROP_NA_THRES`, `ODDS_TYPE`,
  `BETTOR` and `CV`, plus the new `TARGET_EVENT_STATUS` (default `postplay`) and `TARGET_EVENT_TIME` (such as
  `'60min'`, ignored unless the status is `inplay`).

## Risks / Trade-offs

- The column grammar is the riskiest coupling. Extraction produces it and bettors consume it. One shared definition
  avoids drift.
- In-play targets on the real feed return no rows. Users see the engine work only on sample data until an in-play source
  exists.
- Keeping the selection interface layered on the snapshot model costs a convenience layer over the base engine, in
  exchange for compatibility.
- Changing the column names breaks code that parsed the legacy names.

## Quality checks

The specification quality checklist passed every item: no implementation details, testable requirements, measurable and
technology-agnostic success criteria, defined scenarios and edge cases, and bounded scope. The former FR-011 open
clarification was resolved by R1. The definition of done required quickstart scenarios 1 to 7 to pass offline, every
docstring example to run under `--doctest-modules`, the user guide and examples to be updated, a CHANGELOG entry, and
the `formatting`, `checks` and `tests` gates to pass with no coverage regression. All were met.
