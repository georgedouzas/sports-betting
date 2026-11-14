# sources/sample Specification

## Purpose

Defines `SampleSoccerStats` and `SampleSoccerOdds`, the frozen soccer season bundled with the library so that examples
and tests run offline with the same interface as the network sources.

## Requirements

### Requirement: Sample source identity

`SampleSoccerStats` and `SampleSoccerOdds` SHALL share the name `sample_soccer` and the sport `soccer`, with the kinds
`stats` and `odds`. Since the two share a name, a dataloader pairing them treats the odds as already carrying the
identity of the statistics.

#### Scenario: Statistics identity

- **WHEN** a user reads `name`, `kind` and `sport` of `SampleSoccerStats()`
- **THEN** they are `sample_soccer`, `stats` and `soccer`

#### Scenario: Odds identity

- **WHEN** a user reads `name`, `kind` and `sport` of `SampleSoccerOdds()`
- **THEN** they are `sample_soccer`, `odds` and `soccer`

### Requirement: Fixed catalogue

The sample sources SHALL declare no index items and SHALL publish exactly the England and Spain first divisions of
the 2024 season, so `list_available_params()` reads nothing.

#### Scenario: Catalogue without the network

- **WHEN** a user calls `SampleSoccerStats().list_available_params()`
- **THEN** it returns `[{'division': 1, 'league': 'England', 'year': 2024}, {'division': 1, 'league': 'Spain',
  'year': 2024}]`

### Requirement: Bundled files as items

`list_required_items(params)` SHALL return, for each selected combination with a bundled file, one item with the key
`<league>_<division>_<year>_<kind>` and the `file://` URL of the gzip compressed CSV file shipped in the package. A
combination without a bundled file MUST be skipped.

#### Scenario: Odds item key

- **WHEN** a user calls `SampleSoccerOdds().list_required_items` with the England 2024 combination
- **THEN** it returns one item whose key is `England_1_2024_odds`

#### Scenario: Unbundled league

- **WHEN** a user calls `SampleSoccerStats().list_required_items([{'league': 'Italy', 'division': 1, 'year': 2024}])`
- **THEN** it returns an empty list

### Requirement: Sample statistics snapshots

`SampleSoccerStats.to_snapshots` SHALL return, for each of the 380 matches of a league, a `preplay` row at time zero
with `home_points_avg` and `away_points_avg`, an `inplay` row at 45 minutes with the half-time goals, and a `postplay`
row at time zero with the full-time goals. The in-play and post-play rows MUST carry the outcomes of `home_win`,
`draw`, `away_win`, `over_2.5` and `under_2.5` at that moment.

#### Scenario: Three moments

- **WHEN** a user reads the England 2024 statistics through `fetch_payloads` and `to_snapshots`
- **THEN** the event statuses are `inplay`, `postplay` and `preplay`

### Requirement: Sample odds snapshots

`SampleSoccerOdds.to_snapshots` SHALL return one `preplay` row at time zero per match and provider, for the providers
`market_average` and `market_maximum`, with the prices of `home_win`, `draw`, `away_win`, `over_2.5` and `under_2.5`.

#### Scenario: Two providers

- **WHEN** a user reads the England 2024 odds through `fetch_payloads` and `to_snapshots`
- **THEN** the providers are `market_average` and `market_maximum`

### Requirement: Sample snapshot types

The sample sources SHALL return `date` as a UTC timestamp and `event_time` as a time delta. Given no payloads, they
MUST return an empty data frame.

#### Scenario: No payloads

- **WHEN** a user calls `SampleSoccerStats().to_snapshots([])`
- **THEN** it returns an empty data frame

### Requirement: Sample pairing

The sample statistics and odds SHALL name their teams the same way, so `resolve_odds` keeps every sample odds row.

#### Scenario: All odds resolved

- **WHEN** a user resolves the snapshots of both sample seasons with `resolve_odds`
- **THEN** the result has as many rows as the odds snapshots
