## ADDED Requirements

### Requirement: Basketball dataloader
The library SHALL provide `BasketballDataLoader`, exported from `sportsbet.datasets`. It SHALL use the same extraction
contract, preparation step and store as `SoccerDataLoader`, with no sport-specific branch in the engine. Its default
statistics source SHALL be `EuroLeagueStats` and it SHALL have no default odds source, because no free basketball odds
source exists.

#### Scenario: Prepare and extract a EuroLeague season
- **WHEN** a user prepares `BasketballDataLoader(param_grid={'league': ['Euroleague'], 'division': [1], 'year': [2025]},
  odds=OddsApi(key=..., markets=['h2h']))` and calls `extract_train_data`
- **THEN** the returned `X`, `Y` and `O` share the same rows and index

#### Scenario: Fixtures share the training columns
- **WHEN** the user calls `extract_fixtures_data` on the same dataloader
- **THEN** the fixtures carry the same feature columns as the training data, with the unplayed result absent

#### Scenario: Only seasons both sources publish are selectable
- **WHEN** the statistics source publishes a season that the odds source does not cover
- **THEN** the dataloader does not offer that season

#### Scenario: Dry run prices a basketball season
- **WHEN** the user calls `prepare(dry_run=True)` with a metered odds source
- **THEN** the report gives an exact `estimated_cost` and no credit is spent

### Requirement: Mutually exclusive outcomes come from the data
The extraction SHALL derive which outcomes are mutually exclusive from the outcome columns the data carries, not from a
sport setting. A dataset with `home_win` and `away_win` and no `draw` SHALL be treated as two-way, and a soccer dataset
SHALL keep its three-way outcome with a draw.

#### Scenario: Basketball probabilities sum to one
- **WHEN** a bettor is fitted on basketball data and predicts probabilities
- **THEN** the probabilities of a home win and an away win sum to one

#### Scenario: Soccer keeps its draw
- **WHEN** targets are derived for a soccer dataset
- **THEN** they still contain the draw

### Requirement: No markets to predict
When the odds carry no markets, extraction SHALL fail with an error that says there are no markets to predict. It SHALL
NOT fail with an internal schema validation error. This SHALL hold both when the dataloader has no odds source and when
the odds source carries no markets.

#### Scenario: No odds source
- **WHEN** a user calls `extract_train_data` on a `BasketballDataLoader` with no odds source
- **THEN** it raises an error stating that there are no markets to predict

#### Scenario: Odds source without markets
- **WHEN** the odds source produces a table with no market columns
- **THEN** extraction raises an error stating that there are no markets to predict

### Requirement: Odds columns are never fixed
Every odds value column SHALL be treated as time-varying and SHALL carry its provider in its column name, because a
price is a provider's price at a moment. This SHALL hold when the odds source has a single provider.

#### Scenario: Single provider keeps its prefix
- **WHEN** the odds come from a source with exactly one provider
- **THEN** each odds column in `O` carries the provider in its name and a bettor accepts the odds

#### Scenario: Soccer odds unchanged
- **WHEN** soccer odds with two providers are extracted
- **THEN** the result matches the soccer frozen reference

### Requirement: Public market_outcomes is sport-neutral
`sportsbet.datasets.market_outcomes` SHALL remain importable at that path and SHALL serve both soccer and basketball
sources.

#### Scenario: Import path still works
- **WHEN** a user imports `market_outcomes` from `sportsbet.datasets`
- **THEN** the import succeeds and the function derives the outcomes from a final score
