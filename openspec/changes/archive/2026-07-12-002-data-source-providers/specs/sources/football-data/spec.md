## Purpose

The free, key-less football-data.co.uk statistics and odds sources download the public soccer feed on the user's
machine and transform it locally into long snapshots.

## ADDED Requirements

### Requirement: Free soccer sources
The library SHALL provide `FootballDataStats` and `FootballDataOdds`, which need no key, charge a cost of zero, and
fetch the public football-data.co.uk files on the user's machine. They MUST be the default `stats` and `odds` sources of
`SoccerDataLoader`.

#### Scenario: Default construction uses the free sources
- **WHEN** the user builds `SoccerDataLoader(param_grid=...)` with no sources
- **THEN** the loader reads its statistics and odds from football-data.co.uk through the free sources

### Requirement: Fetch only the selected leagues and seasons
The free sources SHALL fetch only the league index pages of the leagues in the `param_grid` and only the season files
those leagues and seasons need, plus the single global fixtures file, never the full upstream catalogue.

#### Scenario: Two leagues selected
- **WHEN** the `param_grid` selects two leagues and preparation runs
- **THEN** only those two leagues' index pages and selected season files are fetched

### Requirement: Discovery from the league index pages
`FootballDataStats.available_params()` SHALL derive the published league, division and year combinations from the
upstream league index pages. A combination the upstream does not publish MUST NOT be offered.

#### Scenario: An unpublished division is not offered
- **WHEN** the user lists the available parameters
- **THEN** no combination appears that has no season file upstream, such as a second Dutch division

### Requirement: One download serves both sources
`FootballDataStats.required_items` and `FootballDataOdds.required_items` SHALL return items with identical identities,
one per league-season file plus the fixtures file, so each upstream file is fetched once and transformed by both.

#### Scenario: Shared season file
- **WHEN** a loader with both free sources prepares one league-season
- **THEN** the season file is fetched once

### Requirement: Reproduce the mirrored dataset
For the reference `param_grid` of England, division 1, 2024, the free sources SHALL reproduce the previously mirrored
`stats`, `odds`, `X`, `Y` and `O` exactly, same rows, columns, dtypes and values. A committed fingerprint of per-column
hashes, dtypes, row counts and column order MUST gate this, and a failure MUST name the drifted column.

#### Scenario: Equivalence gate
- **WHEN** the equivalence test rebuilds the five frames from the free sources
- **THEN** every fingerprint field matches the committed reference

### Requirement: Preserved transform behaviour
The transform SHALL fill missing integer fields with `-1` and treat a match with no played goals as a fixture that gets
only a pre-play snapshot. It SHALL back-fill each odds column from its closing twin when the plain column is absent or
empty. It SHALL compute form features within a season after appending the fixtures, shifted by one match.

#### Scenario: Fixtures get a pre-play snapshot only
- **WHEN** the transform reads a fixture with no result
- **THEN** it emits a `preplay` snapshot and no `postplay` snapshot for that match

#### Scenario: Closing odds back-fill
- **WHEN** a season file carries only closing odds for a market
- **THEN** the transformed odds hold those prices under the plain market column

### Requirement: Season and fixtures volatility
Items for completed seasons SHALL be immutable (`volatile=False`). Items for the current season and the fixtures file
MUST be volatile (`volatile=True`).

#### Scenario: Current season is refreshed
- **WHEN** the user prepares a store that holds a completed season and the current season
- **THEN** the current season and the fixtures are fetched again and the completed season is not

### Requirement: UK kick-off time converted to UTC
The free sources SHALL read the feed's kick-off `Time` as UK time for every league and convert it to the UTC `date`. A
season without kick-off times MUST keep `date` at midnight UTC of the match day.

#### Scenario: Spanish kick-off
- **WHEN** the feed lists a Spanish match at 20:00 in summer
- **THEN** `date` is 19:00 UTC on that day

### Requirement: Upstream layout changes
When the upstream adds a bookmaker column or drops an old column, the transform SHALL keep the new column and MUST NOT
crash on the missing one.

#### Scenario: A new bookmaker appears
- **WHEN** a season file carries a bookmaker column the transform has not seen before
- **THEN** extraction keeps that bookmaker's odds and does not fail
