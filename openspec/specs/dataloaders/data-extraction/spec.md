# dataloaders/data-extraction Specification

## Purpose

A dataloader reads long event snapshots of statistics and odds from its sources and turns them into moment-aware
modelling data: features `X`, targets `Y` and odds `O` for training, and the same columns for upcoming fixtures.

## Requirements

### Requirement: Event snapshots are the input format
A dataloader SHALL read two long tables, `stats` and `odds`. Each row MUST carry the identity columns `date`, `league`,
`division`, `year`, `home_team` and `away_team`, and the event columns `event_status` and `event_time`. The `odds`
table MUST also carry a `provider` column. A missing identity or event column SHALL raise `ValueError` naming the
missing columns. A missing `provider` column SHALL raise `ValueError`.

#### Scenario: Statistics without a team column are refused
- **GIVEN** statistics snapshots without the `home_team` column
- **WHEN** the user calls `extract_train_data(odds_type='market_average')`
- **THEN** a `ValueError` is raised whose message contains `missing the required columns`

### Requirement: Event statuses and event times
The `event_status` of a snapshot SHALL be one of `preplay`, `inplay` and `postplay`, ranked in that order. The
`event_time` SHALL be a time since kick-off. A numeric `event_time` SHALL be read as minutes. The `date` SHALL be
converted to a UTC timestamp. Together, `date` plus `event_time` SHALL name the instant of the snapshot.

#### Scenario: Numeric event times are minutes
- **GIVEN** snapshots whose `event_time` is the integer `0`
- **WHEN** the dataloader extracts the data
- **THEN** the event time is read as `pd.Timedelta('0min')`

### Requirement: Layout is derived from the data
The dataloader SHALL derive the layout from the snapshots. The odds providers SHALL come from the `provider` column.
The markets SHALL be the odds value columns. The features SHALL be the statistics value columns. A statistics column
whose non-null values are constant within every match SHALL be fixed. Any other statistics column, and every odds
column, SHALL be time varying.

#### Scenario: Providers and markets come from the odds
- **GIVEN** a `BaseDataLoader` subclass whose odds carry one provider `acme` and one value column `home_win`
- **WHEN** the user calls `get_odds_types()` and then `extract_train_data(odds_type='acme')`
- **THEN** `get_odds_types()` returns `['acme']`
- **AND** `Y` has the single column `home_win__postplay__0min`
- **AND** `O` has the single column `acme__home_win__preplay__0min`

### Requirement: Snapshots are validated against derived schemas
The dataloader SHALL build pandera schemas for `stats` and `odds` from the derived layout and validate the snapshots
against them on every extraction. A snapshot that breaks its schema, such as an unknown `event_status`, SHALL raise
`pandera.errors.SchemaError`. Statistics and odds whose snapshot columns differ SHALL raise `ValueError`. The
validated snapshots SHALL be held as `stats_` and `odds_`, and the schemas as `stats_schema_` and `odds_schema_`.

#### Scenario: An unknown event status in the statistics is refused
- **GIVEN** statistics with one snapshot whose `event_status` is `halftime`
- **WHEN** the user calls `extract_train_data()`
- **THEN** a `pandera.errors.SchemaError` is raised

#### Scenario: An unknown event status in the odds is refused
- **GIVEN** odds with one snapshot whose `event_status` is `halftime`
- **WHEN** the user calls `extract_train_data()`
- **THEN** a `pandera.errors.SchemaError` is raised

#### Scenario: Mismatched snapshot columns are refused
- **GIVEN** an odds schema with an extra snapshot column the statistics schema lacks
- **WHEN** the user calls `extract_train_data()`
- **THEN** a `ValueError` is raised whose message contains `Stats and odds snapshots columns do not match`

### Requirement: DataLoader takes its sources explicitly
`DataLoader(param_grid=None, stats=None, odds=None, aliases=None)` SHALL read statistics from `stats` and odds from
`odds`. A dataloader without a `stats` source SHALL raise `ValueError` on extraction, with a message containing
`does not choose where its data comes from`. The `odds` source is optional. A dataloader without odds SHALL carry no
markets.

#### Scenario: No statistics source
- **WHEN** the user calls `DataLoader().extract_train_data()`
- **THEN** a `ValueError` is raised whose message contains `does not choose where its data comes from`

### Requirement: The sport belongs to the sources
`DataLoader.sport_` SHALL return the sport of the statistics source. An odds source whose sport is `None` SHALL pair
with a statistics source of any sport. Statistics and odds sources of different sports SHALL raise `ValueError`, with
a message containing `about different sports`, when `sport_`, `sources_` or an extraction reads them. `sources_` SHALL
return the statistics and odds sources as a tuple.

#### Scenario: The sport comes from the statistics source
- **WHEN** the user reads `DataLoader(stats=FootballDataStats(), odds=FootballDataOdds()).sport_`
- **THEN** it returns `'soccer'`

#### Scenario: A multi-sport odds vendor takes the paired sport
- **WHEN** the user reads `DataLoader(stats=NBAStats(), odds=OddsApi(key_env='ODDS_API_KEY')).sport_`
- **THEN** it returns `'basketball'`

#### Scenario: Different sports are refused
- **WHEN** the user reads `DataLoader(stats=NBAStats(), odds=FootballDataOdds()).sources_`
- **THEN** a `ValueError` is raised whose message contains `about different sports`

### Requirement: Selection of training seasons by param_grid
`param_grid` SHALL select the training seasons in the manner of scikit-learn's `ParameterGrid`. `None` SHALL select
every combination. A dictionary with keys among `league`, `division` and `year` SHALL keep the combinations whose
values are in its lists, and a missing key SHALL allow every value. A list of dictionaries SHALL keep the union. The
candidates SHALL be the combinations both sources publish.

#### Scenario: Only the selected season is downloaded for training
- **GIVEN** a statistics source publishing England 2024 and England 2026
- **AND** a dataloader with `param_grid={'league': ['England'], 'division': [1], 'year': [2024]}`
- **WHEN** the user calls `extract_exploration_data()`
- **THEN** only the England 2024 training item is fetched

### Requirement: Loaded combinations are recorded
After an extraction, `param_grid_` SHALL hold the distinct `league`, `division` and `year` combinations of the loaded
statistics, as a list of dictionaries.

#### Scenario: A reloaded dataloader keeps its combinations
- **GIVEN** a dataloader that extracted training data and was saved
- **WHEN** the user loads it with `load_dataloader`
- **THEN** its `param_grid_` equals the `param_grid_` of the saved dataloader

### Requirement: Extraction reuses the snapshots it holds
An extraction SHALL download the snapshots it needs and hold them on the object. A later training or exploration
extraction with the same `odds_type` SHALL reuse the held snapshots and SHALL NOT fetch again. An extraction with a
different `odds_type` SHALL load the snapshots again.

#### Scenario: A second extraction fetches nothing
- **GIVEN** a dataloader that already called `extract_exploration_data()`
- **WHEN** the user calls `extract_exploration_data()` again
- **THEN** no item is fetched

### Requirement: Available odds types
`get_odds_types()` SHALL load the snapshots and return the distinct values of the odds `provider` column, sorted.

#### Scenario: Providers of the football-data odds
- **GIVEN** a dataloader over snapshots whose providers are `market_average` and `market_maximum`
- **WHEN** the user calls `get_odds_types()`
- **THEN** it returns `['market_average', 'market_maximum']`

### Requirement: Training data shape
`extract_train_data` SHALL take keyword-only arguments `drop_na_thres`, `odds_type`, `target_event_status`,
`target_event_time`, `input_event_status` and `input_event_time`. It SHALL return `(X, Y, O)`, three pandas DataFrames
with the same rows and the same `DatetimeIndex` named `date`. Each row SHALL be one match.

#### Scenario: Aligned training data
- **WHEN** the user calls `extract_train_data(odds_type='market_average')`
- **THEN** `X`, `Y` and `O` are DataFrames of equal length
- **AND** `X.index` is a `DatetimeIndex` equal to `Y.index` and to `O.index`

### Requirement: Target moment parameters
`target_event_status` SHALL be `None`, `'inplay'` or `'postplay'`, and `None` SHALL mean `'postplay'`. Any other string
SHALL raise `ValueError` with `Invalid target event status`. A value that is not a string SHALL raise `TypeError`.
`target_event_time` SHALL be `None` or a `pd.Timedelta`, and `None` SHALL mean zero minutes. Another type SHALL raise
`TypeError`. A negative time SHALL raise `ValueError`.

#### Scenario: An unknown target status
- **WHEN** the user calls `extract_train_data(target_event_status='invalid')`
- **THEN** a `ValueError` is raised whose message contains `Invalid target event status`

#### Scenario: A target status that is not a string
- **WHEN** the user calls `extract_train_data(target_event_status=123)`
- **THEN** a `TypeError` is raised

#### Scenario: A negative target time
- **WHEN** the user calls `extract_train_data(target_event_time=pd.Timedelta('-10min'))`
- **THEN** a `ValueError` is raised whose message contains `The event time`

#### Scenario: A target time that is not a Timedelta
- **WHEN** the user calls `extract_train_data(target_event_time='10min')`
- **THEN** a `TypeError` is raised

### Requirement: Input horizon parameters
`input_event_status` SHALL be `None` or one of `'preplay'`, `'inplay'` and `'postplay'`. Another string SHALL raise
`ValueError` with `Invalid input event status`, and a value that is not a string SHALL raise `TypeError`.
`input_event_time` SHALL be `None` or a non-negative `pd.Timedelta`, and `None` SHALL mean zero minutes. When set, the
horizon SHALL keep only snapshots at or before it as features.

#### Scenario: A pre-match horizon
- **WHEN** the user calls `extract_train_data(odds_type='market_average', input_event_status='preplay',
  input_event_time=pd.Timedelta('0min'))`
- **THEN** no column of `X` contains `__inplay__`

#### Scenario: An in-play horizon at 30 minutes
- **WHEN** the user calls `extract_train_data(odds_type='market_average', input_event_status='inplay',
  input_event_time=pd.Timedelta('30min'))` on data with in-play snapshots at 30, 60 and 90 minutes
- **THEN** some column of `X` contains `__inplay__30min`
- **AND** no column of `X` contains `__inplay__60min` or `__inplay__90min`

### Requirement: Resolved moment attributes
After an extraction, the dataloader SHALL expose the resolved `target_event_status_`, `target_event_time_`,
`input_event_status_` and `input_event_time_`, the checked `drop_na_thres_` and `odds_type_`, the market columns
`targets_`, and the column indexes `input_cols_`, `output_cols_` and `odds_cols_`. `output_cols_` SHALL be `None` after
an exploration extraction.

#### Scenario: Defaults are resolved
- **WHEN** the user calls `extract_train_data(odds_type='market_average')`
- **THEN** `target_event_status_` is `'postplay'` and `target_event_time_` is `pd.Timedelta('0min')`

### Requirement: Features precede the target moment
Every feature in `X` SHALL come from a snapshot strictly before the target moment. A snapshot precedes the target when
its status ranks lower, or when its status is the same and its time is earlier.

#### Scenario: An in-play target at 60 minutes
- **WHEN** the user calls `extract_train_data(target_event_status='inplay', target_event_time=pd.Timedelta('60min'))`
- **THEN** `X` contains `home_goals__inplay__30min`
- **AND** no column of `X` contains `60min` or `90min`

### Requirement: A feature is never later than the odds
The bet moment SHALL be the latest moment before the target at which the selected odds are quoted. When no input
horizon is set, features and odds SHALL be capped at the bet moment. When an input horizon later than the bet moment is
set, extraction SHALL raise `ValueError` with a message containing `could not have had`. Exploration and odds-free
extractions have no bet moment.

#### Scenario: Pre-match odds keep in-play features out
- **GIVEN** statistics with an in-play snapshot at 45 minutes and odds quoted only before kick-off
- **WHEN** the user calls `extract_train_data(odds_type='book')`
- **THEN** no column of `X` contains `inplay`

#### Scenario: In-play odds allow in-play features
- **GIVEN** the same statistics and odds quoted before kick-off and at 45 minutes in play
- **WHEN** the user calls `extract_train_data(odds_type='book')`
- **THEN** some column of `X` contains `inplay`

#### Scenario: A horizon later than the odds is refused
- **GIVEN** odds quoted only before kick-off
- **WHEN** the user calls `extract_train_data(odds_type='book', input_event_status='inplay',
  input_event_time=pd.Timedelta('45min'))`
- **THEN** a `ValueError` is raised whose message contains `could not have had`

### Requirement: Feature column naming
`X` SHALL hold the identity columns `league`, `division`, `year`, `home_team` and `away_team`, and the feature columns.
A fixed feature SHALL keep its bare name. A time varying feature SHALL be named `{col}__{event_status}__{n}min`, with
the event time in whole minutes. A feature SHALL appear only for the statuses at which it carries values.

#### Scenario: Fixed and time varying feature names
- **WHEN** the user calls `extract_train_data()` on data with a fixed `home_latest_streak` and an in-play `home_goals`
- **THEN** `X` contains `league`, `home_team`, `away_team` and `home_latest_streak`
- **AND** `X` contains `home_goals__inplay__30min`

### Requirement: Target column naming
`Y` SHALL hold one column per market, named `{market}__{target_event_status}__{n}min`. Its values SHALL be the market
outcomes of the snapshot at the target moment.

#### Scenario: Default postplay targets
- **WHEN** the user calls `extract_train_data()` on data whose markets are `home_goals` and `away_goals`
- **THEN** the columns of `Y` are `['home_goals__postplay__0min', 'away_goals__postplay__0min']`

#### Scenario: In-play targets
- **WHEN** the user calls `extract_train_data(odds_type='market_average', target_event_status='inplay',
  target_event_time=pd.Timedelta('60min'))`
- **THEN** `Y` contains `home_win__inplay__60min`

### Requirement: Odds column naming and provider selection
`O` SHALL hold the odds of the provider named by `odds_type`, with columns named
`{provider}__{market}__{event_status}__{n}min`. `odds_type=None` SHALL give an `O` with no columns. An `odds_type`
that is not among the providers SHALL raise `ValueError` with `Invalid odds type`.

#### Scenario: One provider is selected
- **WHEN** the user calls `extract_train_data(odds_type='market_maximum')`
- **THEN** every column of `O` starts with `market_maximum__`

#### Scenario: No odds type
- **WHEN** the user calls `extract_train_data(odds_type=None)`
- **THEN** `O` has zero columns

#### Scenario: An unknown odds type
- **WHEN** the user calls `extract_train_data(odds_type='bet365')` on data whose providers are `market_average` and
  `market_maximum`
- **THEN** a `ValueError` is raised whose message contains `Invalid odds type`

### Requirement: Every match with a target is kept
A match that has a snapshot at the target moment SHALL be a row of `X`, even when it has no feature values. A match
whose target outcomes are missing SHALL be dropped from `X`, `Y` and `O` together.

#### Scenario: A match with no features survives
- **GIVEN** two matches, the first with no pre-match feature value
- **WHEN** the user calls `extract_train_data(odds_type='book')`
- **THEN** `X`, `Y` and `O` each have two rows

### Requirement: Extraction errors for unusable data
Extraction SHALL raise `ValueError` with `` No `inplay` or `postplay` events were found `` when the statistics hold
only `preplay` snapshots. It SHALL raise `ValueError` with `No resolvable events were found` when no match has a
snapshot at the target moment.

#### Scenario: Only pre-match statistics
- **GIVEN** statistics with only `preplay` snapshots
- **WHEN** the user calls `extract_train_data()`
- **THEN** a `ValueError` is raised whose message contains `` No `inplay` or `postplay` events were found ``

### Requirement: Training needs markets
`extract_train_data` SHALL raise `ValueError` when the odds carry no market columns, with a message containing
`no markets to predict` that names `extract_exploration_data`.

#### Scenario: A dataloader without odds
- **GIVEN** a dataloader whose odds are empty
- **WHEN** the user calls `extract_train_data()`
- **THEN** a `ValueError` is raised whose message contains `no markets to predict`

### Requirement: Dropping sparse feature columns
`drop_na_thres` SHALL be a number in `[0.0, 1.0]`, and `None` SHALL mean `0.0`. A feature column SHALL be kept when its
share of missing values is at most one minus `drop_na_thres`. `0.0` SHALL keep every column, and `1.0` SHALL drop every
column with a missing value. A value outside the range SHALL raise `ValueError`, and a non-number SHALL raise
`TypeError`, each naming `drop_na_thres`.

#### Scenario: Keep every column
- **WHEN** the user calls `extract_train_data(drop_na_thres=0.0, odds_type='market_average')` on the England sample
- **THEN** `X` has 7 columns

#### Scenario: Drop every column with a missing value
- **WHEN** the user calls `extract_train_data(drop_na_thres=1.0, odds_type='market_average')` on the England sample
- **THEN** `X` has 5 columns

#### Scenario: A threshold outside the range
- **WHEN** the user calls `extract_exploration_data(drop_na_thres=1.5)`
- **THEN** a `ValueError` is raised whose message contains `drop_na_thres`

### Requirement: A bare call repeats the last extraction
A call to `extract_train_data()` with no arguments, on a dataloader that already extracted, SHALL reuse the stored
`odds_type_`, `drop_na_thres_` and moment attributes. On a reloaded dataloader it SHALL rebuild the same data without
downloading.

#### Scenario: A reloaded dataloader rebuilds its data
- **GIVEN** a dataloader that extracted with `odds_type='market_average'` and was saved and reloaded
- **WHEN** the user calls `extract_train_data()` on the reloaded dataloader
- **THEN** the returned `X` equals the `X` of the first extraction

### Requirement: Exploration data
`extract_exploration_data` SHALL take keyword-only `drop_na_thres` (default `0.0`), `target_event_status`,
`target_event_time`, `input_event_status` and `input_event_time`, with the same validation as training. It SHALL return
only the features `X`, with no targets and no odds, and SHALL work for a dataloader without odds.

#### Scenario: Features without odds
- **GIVEN** a dataloader over two matches and no odds
- **WHEN** the user calls `extract_exploration_data()`
- **THEN** it returns a non-empty DataFrame with two rows

### Requirement: Fixtures need a prior extraction
`extract_fixtures_data()` SHALL raise `NotExtractedError` when the dataloader has not yet extracted training or
exploration data. `NotExtractedError` SHALL subclass scikit-learn's `NotFittedError`, so it is also a `ValueError`.

#### Scenario: Fixtures before training
- **WHEN** the user calls `extract_fixtures_data()` on a fresh dataloader
- **THEN** a `NotExtractedError` is raised whose message contains `extract_train_data`

### Requirement: Fixtures are downloaded separately
`extract_fixtures_data` SHALL download the fixtures items of the selected leagues, which describe the current season.
It SHALL NOT download the training seasons, and it SHALL download on every call.

#### Scenario: Fixtures fetch the current data
- **GIVEN** a dataloader that called `extract_exploration_data()` for the 2024 season
- **WHEN** the user calls `extract_fixtures_data()`
- **THEN** the current fixtures item is fetched
- **AND** the 2024 training item is not fetched

### Requirement: Fixtures are future matches without a target snapshot
`extract_fixtures_data` SHALL keep only matches whose `date` is at or after the current UTC time. It SHALL exclude
every match that already has a snapshot at the stored target moment. Its odds SHALL be those of the stored
`odds_type_`, and `None` SHALL give no odds.

#### Scenario: The upcoming match is returned
- **GIVEN** snapshots with played matches and one upcoming match between Arsenal and Chelsea
- **WHEN** the user calls `extract_train_data(odds_type='market_average')` and then `extract_fixtures_data()`
- **THEN** the fixtures `X` has the single row with `home_team` `Arsenal` and `away_team` `Chelsea`

### Requirement: Fixtures share the training columns
`extract_fixtures_data` SHALL return `(X, None, O)`. `X` and `O` SHALL have exactly the columns of the training `X` and
`O`, in the same order, built with the stored target moment and input horizon. A column the fixtures lack SHALL be
filled with missing values. With no fixtures, `X` and `O` SHALL be empty frames with those columns and a `date` index.

#### Scenario: Matching columns
- **WHEN** the user calls `extract_train_data()` and then `extract_fixtures_data()`
- **THEN** the fixtures `Y` is `None`
- **AND** the fixtures `X` and `O` have the columns of the training `X` and `O`

#### Scenario: A finished season has no fixtures
- **GIVEN** a dataloader over the bundled sample season, which is finished
- **WHEN** the user calls `extract_train_data(odds_type='market_average')` and then `extract_fixtures_data()`
- **THEN** the fixtures `X` and `O` have zero rows and the training columns

### Requirement: Save and load a dataloader
`save(path)` SHALL write the dataloader, with the snapshots and settings it holds, to `path` with cloudpickle and
return the dataloader. `load_dataloader(path)` SHALL read it back. A source API key SHALL NOT be stored in the file,
since a source names its key by environment variable.

#### Scenario: A loaded dataloader extracts fixtures with the training columns
- **GIVEN** a dataloader that extracted training data and was saved with `save(path)`
- **WHEN** the user calls `load_dataloader(path).extract_fixtures_data()`
- **THEN** the fixtures `X` has the columns of the training `X`

### Requirement: A dataloader of your own
`BaseDataLoader(param_grid=None)` SHALL be abstract, and instantiating it SHALL raise `TypeError`. A subclass that
implements `_load_snapshots`, returning the `stats` and `odds` snapshots, SHALL support every extraction method,
`get_odds_types`, `save` and `load_dataloader`. Its `sources_` SHALL be an empty tuple.

#### Scenario: The base cannot be instantiated
- **WHEN** the user calls `BaseDataLoader()`
- **THEN** a `TypeError` is raised whose message contains `abstract`

#### Scenario: A subclass with snapshots in memory
- **GIVEN** a subclass whose `_load_snapshots` returns one pre-match feature, one postplay `home_win` and odds from
  `acme`
- **WHEN** the user calls `extract_train_data(odds_type='acme')`
- **THEN** `Y` has the column `home_win__postplay__0min`

### Requirement: Build a dataloader from source names
`build_dataloader(stats, odds=None, leagues=None, divisions=None, years=None, odds_key_env='ODDS_API_KEY',
odds_markets=None, odds_regions=None, odds_moments=None, aliases=None)` SHALL return a `DataLoader`. `stats` SHALL be
one of `football-data`, `euroleague` and `nba`. `odds` SHALL be `None`, `football-data` or `odds-api`. The non-empty
lists among `leagues`, `divisions` and `years` SHALL form the `param_grid`.

#### Scenario: Sources by name
- **WHEN** the user calls `build_dataloader(stats='football-data', odds='football-data', leagues=['England'],
  divisions=[1], years=[2025])`
- **THEN** the dataloader's `stats` is a `FootballDataStats` and its `odds` is a `FootballDataOdds`

### Requirement: Build errors
`build_dataloader` SHALL raise `BuildError`, a subclass of `ValueError`, when `stats` or `odds` names no known source,
when `odds='odds-api'` and the environment variable named by `odds_key_env` is empty, when an `odds_moments` entry is
not `status:minutes` with a known status, or when an `aliases` entry is not two non-empty names joined by `=`.

#### Scenario: An unknown statistics source
- **WHEN** the user calls `build_dataloader(stats='unknown')`
- **THEN** a `BuildError` is raised whose message contains `is not a statistics source`

#### Scenario: A keyed odds source without its key
- **GIVEN** the variable `ODDS_API_KEY` is unset
- **WHEN** the user calls `build_dataloader(stats='football-data', odds='odds-api')`
- **THEN** a `BuildError` is raised whose message names `ODDS_API_KEY`

### Requirement: Build extraction settings from text
`build_extraction_settings(odds_type=None, drop_na_thres=None, target_event_status=None, target_event_time=None,
input_event_status=None, input_event_time=None)` SHALL return the keyword arguments for `extract_train_data`. It SHALL
convert the two times from text such as `45min` to `pd.Timedelta`, and it SHALL drop every argument that is `None`.

#### Scenario: Text times become Timedeltas
- **WHEN** the user calls `build_extraction_settings(odds_type='market_average', target_event_time='45min')`
- **THEN** it returns `{'odds_type': 'market_average', 'target_event_time': pd.Timedelta('45min')}`
