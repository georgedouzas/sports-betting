## Purpose

Turn match statistics and odds, recorded as time-ordered snapshots, into aligned feature, target and odds tables for a
chosen match moment, with schema validation and no leakage of later information.

## ADDED Requirements

### Requirement: Match data as event snapshots
The system SHALL represent match data as time-ordered snapshots. Each snapshot MUST carry an `event_status` of
`preplay`, `inplay` or `postplay` and an `event_time` within the match. A `preplay` snapshot MUST have
`event_time >= 0`, an `inplay` snapshot MUST have `event_time > 0`, and a `postplay` snapshot MUST have
`event_time == 0`.

#### Scenario: A snapshot with an inconsistent status and time is rejected
- **WHEN** a statistics dataset holds an `inplay` row with `event_time` equal to zero
- **THEN** extraction raises `pandera.errors.SchemaError` before any table is produced

#### Scenario: An unknown event status is rejected
- **WHEN** a statistics dataset holds an `event_status` outside `preplay`, `inplay` and `postplay`
- **THEN** extraction raises `pandera.errors.SchemaError` naming the `event_status` field

### Requirement: Separate statistics and odds datasets with a shared snapshot identity
The system SHALL keep statistics and betting odds as two datasets that are validated independently. Both MUST share the
snapshot identity columns that uniquely identify a match moment. Extraction MUST refuse the data when the snapshot
columns of the two datasets differ, rather than produce misaligned `X` and `O`.

#### Scenario: Mismatched snapshot columns are refused
- **WHEN** the odds dataset's snapshot columns differ from the statistics dataset's snapshot columns
- **THEN** extraction raises `AssertionError` and returns no tables

### Requirement: Schema validation before extraction
The system SHALL validate every statistics and odds dataset that crosses a public boundary against an explicit schema
before any extraction or modelling. Invalid data MUST be rejected with an error that names the offending field. A
dataset with more than one snapshot for the same match moment MUST be rejected. Non-null `postplay` odds MUST be
rejected, and null `postplay` odds MUST be accepted as valid.

#### Scenario: A missing required field is rejected
- **WHEN** a statistics dataset lacks a required field
- **THEN** extraction raises `pandera.errors.SchemaError` naming that field, before any modelling

#### Scenario: A duplicate snapshot is rejected
- **WHEN** a dataset holds two rows for the same match moment
- **THEN** extraction raises `pandera.errors.SchemaError`

#### Scenario: Post-match odds are null
- **WHEN** an odds dataset holds `postplay` rows whose odds are all null
- **THEN** validation accepts the dataset

#### Scenario: Post-match odds are present
- **WHEN** an odds dataset holds a `postplay` row with a non-null odd
- **THEN** validation rejects the dataset with an error

### Requirement: Feature metadata in the schema
A schema SHALL let each feature declare the event statuses it applies to (`include`) and whether it is time-invariant
(`fixed`). Schemas MUST be built from `required_col` and `optional_col`, and `BaseStatsSchema` and `BaseOddsSchema` MUST
be the bases a user subclasses. A fixed feature MUST appear once in `X`, and a varying feature MUST appear once per
included snapshot.

#### Scenario: A fixed feature keeps a bare name
- **WHEN** a schema declares `league` as fixed and the user extracts training data
- **THEN** `X` holds one column named `league`

#### Scenario: A varying feature is expanded per snapshot
- **WHEN** a schema declares `home_goals` as varying with `include=['inplay']`
- **THEN** `X` holds one `home_goals` column per included in-play snapshot before the target moment

### Requirement: Target moment selection
`extract_train_data` SHALL accept a target moment made of `target_event_status` and `target_event_time`. The default
target MUST be `postplay` at zero time. An invalid `target_event_status` MUST raise `ValueError`. A negative
`target_event_time` MUST raise `ValueError` with a clear message.

#### Scenario: Default target is the full-time outcome
- **WHEN** the user calls `extract_train_data()` with no target arguments
- **THEN** `Y` reflects the `postplay` outcome

#### Scenario: Negative target time is rejected
- **WHEN** the user passes a negative `target_event_time`
- **THEN** `extract_train_data` raises `ValueError`

#### Scenario: Unknown target status is rejected
- **WHEN** the user passes a `target_event_status` that is not a supported status
- **THEN** `extract_train_data` raises `ValueError`

### Requirement: Aligned training tables
`extract_train_data` SHALL return three tables, input features `X`, target outcomes `Y` and odds `O`, that share the
same rows and the same date index. The return MUST always have three positions, so a caller can always write
`X, Y, O = loader.extract_train_data(...)`.

#### Scenario: Post-match target returns aligned tables
- **GIVEN** data with pre-match, in-play and post-match snapshots
- **WHEN** the user extracts training data targeting `postplay`
- **THEN** `X`, `Y` and `O` share the same rows, `Y` holds final outcomes, and `X` holds pre-match and in-play
  information

### Requirement: No information after the target moment
`X` SHALL hold only information available strictly before the target moment. `Y` SHALL be derived from the match state
at the target moment. `O` SHALL hold the odds offered up to the target moment.

#### Scenario: In-play target at 60 minutes
- **GIVEN** sample data with in-play snapshots at several times
- **WHEN** the user extracts training data with `target_event_status='inplay'` and `target_event_time` of 60 minutes
- **THEN** no column of `X` refers to a snapshot after 60 minutes and `Y` reflects the match state at 60 minutes

### Requirement: Column naming grammar
Extracted columns SHALL follow one grammar. A fixed column MUST keep its bare name. A varying feature MUST be named
`{col}__{event_status}__{event_time}`. An odds column MUST be named `{provider}__{col}__{event_status}__{event_time}`. A
target MUST be named `{col}__{target_event_status}__{target_event_time}`. `event_time` MUST render as whole minutes
`{n}min`.

#### Scenario: Names follow the grammar
- **WHEN** the user extracts training data with odds
- **THEN** a varying feature is named like `home_goals__inplay__60min` and an odds column like
  `bet365__home_win__preplay__0min`

### Requirement: Fixtures extraction
`extract_fixtures_data` SHALL return `(X, None, O)` for upcoming matches at the same target moment as the last training
extraction. The `X` and `O` columns MUST equal the columns of the training extraction from the same loader. It MUST be
called after `extract_train_data`, which fixes the column layout.

#### Scenario: Fixtures match the training layout
- **WHEN** the user calls `extract_train_data` and then `extract_fixtures_data` on the same loader
- **THEN** `Y` is `None`, `X` has the training `X` columns and `O` has the training `O` columns

### Requirement: Supervised and unsupervised extraction only
`extract_train_data` SHALL accept `learning_type` of `'supervised'`, the default, or `'unsupervised'`. Unsupervised
extraction MUST return `(X, None, O)`. Any other value, including `'reinforcement'`, MUST raise `ValueError`.
Extraction MUST always return static tables and never an environment.

#### Scenario: Unsupervised extraction
- **WHEN** the user calls `extract_train_data(learning_type='unsupervised')`
- **THEN** the result is `(X, None, O)`

#### Scenario: Reinforcement is rejected
- **WHEN** the user calls `extract_train_data(learning_type='reinforcement')`
- **THEN** it raises `ValueError` listing `supervised` and `unsupervised` as the valid values

### Requirement: No resolvable outcomes
Extraction SHALL fail with a clear `ValueError` when the data holds no in-play or post-match snapshots from which a
target can be resolved. When the target moment lies beyond every recorded snapshot of a match, that match MUST yield no
target row, and the condition MUST be reported rather than returned as a silently empty result.

#### Scenario: Pre-match data only
- **GIVEN** data with only `preplay` snapshots
- **WHEN** the user extracts training data
- **THEN** extraction raises `ValueError` explaining that no resolvable outcomes are available

#### Scenario: Target time beyond the data
- **WHEN** the user targets an in-play time later than any recorded snapshot of a match
- **THEN** that match yields no target row and the system reports it

### Requirement: Parameter discovery
A soccer dataloader SHALL expose `get_all_params()` as a class method that returns the selectable parameter
combinations, such as league, division and year, without downloading the full data.

#### Scenario: Discover parameters offline
- **WHEN** the user calls `DummySoccerDataLoader.get_all_params()`
- **THEN** it returns the selectable combinations with no network access

### Requirement: Odds type discovery
A soccer dataloader SHALL expose `get_odds_types()`, which returns the odds types available for its selection without
downloading the full data.

#### Scenario: Discover odds types
- **WHEN** the user calls `loader.get_odds_types()`
- **THEN** it returns the selectable odds types, such as `market_average`

### Requirement: Parameter grid selection
A soccer dataloader SHALL accept a `param_grid` in the scikit-learn `ParameterGrid` style, such as
`{'league': ['England']}`, to select which data to load. `param_grid=None` MUST select all data. A selection that
matches no data MUST be reported as a clear empty-result condition rather than an obscure failure.

#### Scenario: Select by league
- **WHEN** the user builds `DummySoccerDataLoader(param_grid={'league': ['England']})` and extracts training data
- **THEN** every row of `X` belongs to the England league

#### Scenario: Selection matches nothing
- **WHEN** the user selects a parameter combination that matches no data and extracts
- **THEN** the system reports an empty-result condition with a clear message

### Requirement: Missingness threshold
`extract_train_data` on a soccer dataloader SHALL accept `drop_na_thres` in `[0.0, 1.0]` and drop feature columns whose
missingness exceeds that threshold. Fixtures extraction MUST drop the same columns.

#### Scenario: Sparse columns are dropped consistently
- **WHEN** the user extracts training data with a `drop_na_thres` above zero and then extracts fixtures
- **THEN** the sparse feature columns are absent from both `X` tables and the two `X` tables have equal columns

### Requirement: Odds type selection
`extract_train_data` on a soccer dataloader SHALL accept `odds_type`, one of the values from `get_odds_types()`. With
`odds_type=None` it MUST return no odds.

#### Scenario: Choose an odds type
- **WHEN** the user extracts training data with `odds_type='market_average'`
- **THEN** `O` holds the market average odds

#### Scenario: No odds type
- **WHEN** the user extracts training data with `odds_type=None`
- **THEN** no odds are returned

### Requirement: Dataloader persistence
A dataloader SHALL be saved with `save(path)` and reloaded with `load_dataloader(path)`. The reloaded loader MUST
reproduce the same fixtures column structure as the original.

#### Scenario: Save and reload
- **WHEN** the user extracts training data, saves the loader, reloads it with `load_dataloader` and extracts fixtures
- **THEN** the reloaded loader's fixtures `X` and `O` columns equal the original loader's fixtures columns

### Requirement: Offline sample dataloader
The library SHALL provide `DummySoccerDataLoader`, which holds in-play sample data and exposes the `SoccerDataLoader`
selection interface. It MUST need no network access, so the full workflow of extraction, backtest and value bets runs
offline.

#### Scenario: Offline end-to-end workflow
- **WHEN** the user extracts training and fixtures data from `DummySoccerDataLoader`, backtests a bettor and asks for
  value bets
- **THEN** every step completes with no network access

#### Scenario: In-play target on sample data
- **WHEN** the user targets `inplay` at 60 minutes on `DummySoccerDataLoader`
- **THEN** extraction succeeds and no `X` column contains `90min`

### Requirement: Public dataset exports
`sportsbet.datasets` SHALL export `BaseDataLoader`, `SoccerDataLoader`, `DummySoccerDataLoader`, `load_dataloader`,
`BaseStatsSchema`, `BaseOddsSchema`, `required_col` and `optional_col`.

#### Scenario: Import the public names
- **WHEN** the user imports each of those names from `sportsbet.datasets`
- **THEN** every import succeeds
