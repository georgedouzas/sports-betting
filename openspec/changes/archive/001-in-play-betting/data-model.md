# Phase 1 Data Model: In-Play (Live) Betting Support

Derived from the spec's Key Entities and the confirmed snapshot design. This describes the
logical data contract, not implementation. Column-naming grammar is defined in
[research.md R4](./research.md).

## Core concepts

### Match snapshot (row-level unit)

A single observation of one match at one moment. Every row of the `stats` and `odds`
datasets is a snapshot.

Identity fields (the "snapshot columns", `snapshot=True` metadata):
- `event_status` — one of `preplay`, `inplay`, `postplay`.
- `event_time` — duration into the match (a time delta). Constraints:
  - `preplay` ⇒ `event_time >= 0`
  - `inplay` ⇒ `event_time > 0`
  - `postplay` ⇒ `event_time == 0`
- Sport/match identity: `date`, `league`, `division`, `season`, `home_team`, `away_team`
  (soccer). These plus `event_status`/`event_time` uniquely identify a snapshot.

**Validation rules**:
- No two rows may share the same snapshot identity (uniqueness check).
- `event_status` must be one of the three allowed values; `event_time` must satisfy the
  status↔time rule above.

### Statistics dataset (`stats`)

Snapshots carrying on-field information and outcome-bearing columns.

Column categories:
- **Snapshot/identity columns** (`required_col`): as above.
- **Feature columns** (`optional_col(include, fixed)`): observed statistics. Metadata:
  - `include`: the list of event statuses at which this column is meaningful (e.g. a
    live-goals count applies to `['inplay']`; a pre-match streak applies to `['preplay']`).
  - `fixed`: `True` if time-invariant within a match (kept once, not expanded per snapshot);
    `False` if it varies across snapshots (expanded into per-moment columns).
- **Target-source columns**: the subset named by the loader's `targets` (e.g. `home_goals`,
  `away_goals`) from which `Y` outcomes are derived at the target moment.

### Odds dataset (`odds`)

Snapshots carrying bookmaker odds per provider.

Column categories:
- **Snapshot/identity columns**: as above, plus `provider` (fixed, `include=['preplay']`
  metadata in the sample; identifies the bookmaker).
- **Odds columns** (`optional_col(include, fixed=False)`): decimal odds per market/outcome
  (e.g. `home_win`, `away_win`). Metadata `include` lists statuses at which the market trades.

**Validation rules**:
- `postplay` odds MUST be null (settled outcomes have no tradeable odds) — dataframe check.
- The odds snapshot columns MUST match the stats snapshot columns (extraction asserts this).

## Schemas (contracts)

Schemas are `pandera` `DataFrameModel` subclasses. The base layer is sport-agnostic; concrete
soccer schemas add the sport identity and market columns.

- **BaseSchema** (abstract): `event_status`, `event_time` + checks (status/time consistency,
  snapshot uniqueness) + helpers `snapshot_cols()`, `col_metadata(col)`. `Config.strict=True`.
- **BaseStatsSchema(BaseSchema)**: marker base for statistics.
- **BaseOddsSchema(BaseSchema)**: adds `odds_cols()` helper + post-match-null-odds check.
- **SoccerStatsSchema(BaseStatsSchema)** *(new)*: `date`, `league`, `division`, `season`,
  `home_team`, `away_team` (required) + soccer feature columns with `include`/`fixed`
  metadata and the target-source goal columns.
- **SoccerOddsSchema(BaseOddsSchema)** *(new)*: sport identity + `provider` + soccer market
  odds columns (`home_win`, `away_win`, draw, over/under, …).

## Extraction outputs

`learning_type` accepts `'supervised'` (default) or `'unsupervised'` only; `extract_train_data`
always returns a uniform three-tuple `(X, Y, O)` (`Y` is `None` when unsupervised), so callers
can always unpack `X, Y, O = extract_train_data(...)`. Reinforcement learning is a separate
future method, not a `learning_type` value (see [research.md R1](./research.md)).

### `TrainData = (X, Y, O)` (supervised)

- **X** — one row per match; columns = fixed features (bare names) + time-varying features
  for every included snapshot strictly before the target moment, named
  `{col}__{status}__{time}`.
- **Y** — one row per match; target outcomes evaluated at the target moment, named
  `{col}__{target_status}__{target_time}`.
- **O** — one row per match; odds per provider/market up to the target moment, named
  `{provider}__{col}__{status}__{time}`.
- All three share the same index/rows.

### Unsupervised

Returns `(X, None, O)` — same shape as supervised, with `Y` set to `None`.

### `FixturesData = (X, None, O)`

Upcoming matches: same `X`/`O` column structure as the training extraction from the same
loader; `Y` is always `None`.

### Reinforcement (future — design only, not implemented)

RL is **not** a `learning_type` value and is **not** part of `extract_train_data`. It is
designed as a separate future method (`make_env()`) returning a stateful, Gymnasium-style
environment (observation = snapshot state, action = bet/no-bet per market, reward = realized
return, episode = one match through its snapshots). No RL code ships this feature; the full
forward contract lives in [research.md R1](./research.md).

## Dataloader (behavioral entity)

- **BaseDataLoader**: constructed from `(stats, odds, stats_schema, odds_schema, targets)`.
  - `extract_train_data(learning_type, target_event_status, target_event_time)` → validates,
    pivots, returns `TrainData`/unsupervised tuple (`learning_type` ∈ {supervised,
    unsupervised}). Sets fitted state
    `learning_type_`, `target_event_status_`, `target_event_time_`.
  - `extract_fixtures_data()` → `FixturesData` (to be implemented; mirrors train-data columns,
    no targets). MUST be called after `extract_train_data` to fix the column layout.
  - `save(path)` / module-level `load_dataloader(path)`.
- **SoccerDataLoader(BaseDataLoader)**: adds `param_grid` selection, `get_all_params()`,
  `get_odds_types()`, `drop_na_thres`, `odds_type`; builds `stats`/`odds` + schemas from the
  feed, then delegates to the base engine.
- **DummySoccerDataLoader**: same interface, in-play sample data, no network.

**Fitted-state attributes** (trailing underscore, per Principle I): `param_grid_`,
`drop_na_thres_`, `odds_type_`, `learning_type_`, `target_event_status_`,
`target_event_time_`, and the resolved `input_cols_` / `output_cols_` / `odds_cols_`.

## Bettor (behavioral entity)

Unchanged scikit-learn estimator contract; input format updated to the new columns.

- `fit(X, Y, O=None)`, `predict_proba(X)`, `predict(X)`, `bet(X, O)`, `score(X, Y, O)`.
- Market discovery: parses market names from the `O`/`X` column grammar (research.md R4)
  rather than the legacy convention.
- Bettors: `ClassifierBettor`, `OddsComparisonBettor`, `BettorGridSearchCV`; evaluated via
  `backtest(...)`. Persisted via `save_bettor`/`load_bettor`.

## Entity relationships

```text
Schema ──validates──> stats / odds  (Match snapshots)
                          │
             Dataloader ──┴── selects/builds ──> (X, Y, O) tables
                          │                          │
                   Target moment                     ├──> Bettor.fit/predict/bet
             (event_status, event_time)              └──> backtest ──> performance
```
