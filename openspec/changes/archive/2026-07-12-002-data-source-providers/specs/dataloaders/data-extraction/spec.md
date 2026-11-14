## ADDED Requirements

### Requirement: Dataloader composes injected sources
`SoccerDataLoader` SHALL stay one public class configured by injection, as
`SoccerDataLoader(param_grid=None, stats=None, odds=None, store=None, aliases=None, max_unmatched_rate=0.0)`, not by a
mode. The statistics and odds sources MUST be selectable independently. Source settings MUST live on the source.
Constructor arguments MUST be stored unmodified and unvalidated. The `_snapshots()` contract MUST NOT change.

#### Scenario: Mix a free and a keyed source
- **WHEN** the user builds `SoccerDataLoader(stats=FootballDataStats(), odds=OddsApi(key=..., markets=['h2h']))`
- **THEN** the loader accepts the pair and no key or market argument appears on the dataloader

#### Scenario: A new source needs no dataloader change
- **WHEN** a developer adds a source that implements the source contract
- **THEN** it works with `SoccerDataLoader` with no change to the dataloader, the extraction engine or other sources

### Requirement: Extraction never fetches
`extract_train_data` and `extract_fixtures_data` on a source-backed dataloader SHALL read only from the store and MUST
NEVER fetch. When the store is not prepared for the `param_grid`, they MUST raise `NotPreparedError`, which carries the
`PreparationReport`, names what is missing and what obtaining it would cost, and points the user to `prepare()`.

#### Scenario: Extract before preparing
- **WHEN** the user builds a `SoccerDataLoader` and calls `extract_train_data()` without `prepare()`
- **THEN** it raises `NotPreparedError` naming the missing items and no network call is attempted

### Requirement: Offline entry points need no preparation
`DummySoccerDataLoader` and the loaders built by `from_snapshots` and `from_dataframe` SHALL keep working with no store,
no source and no preparation. Their inherited `prepare()` MUST do nothing and return an empty `PreparationReport`. The
documented examples MUST keep working, with updates limited to those that build a downloading dataloader.

#### Scenario: Dummy loader without preparation
- **WHEN** the user extracts training data from `DummySoccerDataLoader`
- **THEN** extraction succeeds with no network access and no `prepare()` call

#### Scenario: In-memory snapshots
- **WHEN** the user builds a loader with `from_snapshots` and extracts training data
- **THEN** extraction succeeds with no store and no source

## MODIFIED Requirements

### Requirement: Parameter grid selection
A soccer dataloader SHALL accept a `param_grid` in the scikit-learn `ParameterGrid` style, such as
`{'league': ['England']}`, to select which data to load. `param_grid=None` MUST select every combination available. The
available combinations MUST be the intersection of what the statistics source and the odds source publish. A selected
combination outside it MUST be reported as unavailable, never as an empty or smaller dataset.

#### Scenario: Select by league
- **WHEN** the user builds `DummySoccerDataLoader(param_grid={'league': ['England']})` and extracts training data
- **THEN** every row of `X` belongs to the England league

#### Scenario: Selection matches nothing
- **WHEN** the user selects a parameter combination that the sources do not publish
- **THEN** the system reports it as unavailable with a clear message

#### Scenario: Sources with different coverage
- **WHEN** the statistics source publishes seasons back to 1994 and the odds source only from 2020
- **THEN** the loader selects only the seasons both publish

## REMOVED Requirements

### Requirement: Parameter discovery
**Reason**: A `param_grid` cannot be written before it is known what exists, so discovery cannot live on an object
constructed with a `param_grid`. With injected sources, availability depends on the configured sources, which a class
method cannot see.
**Migration**: Ask the source instead, for example `FootballDataStats().available_params()`.
