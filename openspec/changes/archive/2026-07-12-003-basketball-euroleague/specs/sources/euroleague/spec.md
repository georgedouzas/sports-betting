## Purpose

The EuroLeague statistics source reads the competition's official public API and turns its games into long statistics
snapshots: pre-play form for every game and the final score and two-way outcome for every played game.

## ADDED Requirements

### Requirement: Free EuroLeague statistics source
The library SHALL provide `EuroLeagueStats`, a basketball statistics source over the EuroLeague's official public API
that needs no key and no credential. Its data SHALL be fetched on the user's machine during preparation. The project
SHALL NOT redistribute any basketball data.

#### Scenario: Source needs no credential
- **WHEN** a user constructs `EuroLeagueStats()` with no arguments
- **THEN** the source is ready to use and requires no key

#### Scenario: Data is fetched on the user's machine
- **WHEN** the user prepares a dataloader that uses `EuroLeagueStats`
- **THEN** the store downloads the season from the EuroLeague API to the local store
- **AND** the repository contains no EuroLeague data

### Requirement: Seasons read from the API
The source SHALL report the seasons it publishes by reading the competition's season list from the API, and SHALL NOT
fabricate a range of years. Each season SHALL carry `league` `Euroleague`, `division` `1` and `year` equal to the year
the season ends, so the API's `E2024` season is `year` 2025. Reading the season list SHALL cost nothing.

#### Scenario: Catalogue comes from the season list
- **WHEN** the source reads the season list payload
- **THEN** it returns one parameter set for each season in the payload and no other

#### Scenario: Season named by its ending year
- **WHEN** the season list contains the season code `E2024`
- **THEN** the source reports it with `year` 2025

#### Scenario: Unpublished season is reported unavailable
- **WHEN** the user selects a season the source does not publish
- **THEN** the dataloader reports it as unavailable rather than silently returning nothing

### Requirement: One request per season
The source SHALL fetch a whole season with a single request, because the API returns every game of a season in one
response.

#### Scenario: Selected season needs one item
- **WHEN** the source lists the items required for one selected season
- **THEN** it returns exactly one item for that season

### Requirement: Tip-off in UTC
The `date` of every EuroLeague game SHALL be the tip-off instant in UTC. The source SHALL read the API's UTC field and
SHALL NOT read the head-office time field as UTC or as venue time, because the API publishes that field in Central
European Time for every venue and documents no time zone.

#### Scenario: Istanbul game converts correctly
- **WHEN** the API publishes a game in Istanbul at `18:30` head-office time
- **THEN** the snapshot `date` is `17:30` UTC

### Requirement: Played and scheduled games
The source SHALL produce final scores for played games, and SHALL decide per game whether a game was played. A played
game SHALL yield a pre-play snapshot and a post-play snapshot. An unplayed game, including a postponed one, SHALL yield
only a pre-play snapshot, so it becomes a fixture and never a training row with an invented result.

#### Scenario: Played game yields two snapshots
- **WHEN** the payload contains a played game
- **THEN** the source emits a `preplay` snapshot with form and a `postplay` snapshot with the score and outcome

#### Scenario: Scheduled game becomes a fixture
- **WHEN** the payload contains a game that is scheduled or postponed and not yet played
- **THEN** the source emits only a `preplay` snapshot for it

#### Scenario: Season in progress
- **WHEN** a season has both played and unplayed games
- **THEN** one extraction returns the played games as training data and the unplayed games as fixtures

### Requirement: Form features without leakage
The source SHALL produce per-team form features from points scored and points conceded, as an expanding mean over the
season so far and a rolling mean over the last three games. The features SHALL be shifted by one game and computed with
the upcoming fixtures included, so a game never sees its own result and a fixture carries the form of the games before
it. The source SHALL NOT invent other statistics from a score line.

#### Scenario: A game does not see its own result
- **WHEN** form is computed for a played game
- **THEN** the form values use only the games that team played before it

#### Scenario: Fixture carries prior form
- **WHEN** form is computed for an upcoming game
- **THEN** it carries the form of the games each team played before it

#### Scenario: Feature columns
- **WHEN** the source emits a pre-play snapshot
- **THEN** it carries `home_points_for_avg`, `home_points_against_avg`, `home_points_for_latest_avg`,
  `home_points_against_latest_avg` and the same four columns for `away_`

### Requirement: Two-way head-to-head outcome
The outcome of a EuroLeague game SHALL be `home_win` and `away_win` only. It SHALL NOT include a draw, because a tied
game goes to overtime and the final score is the score after overtime. It SHALL NOT include a totals market, because
the bookmaker line moves from game to game and the library expresses a market as a column.

#### Scenario: No draw column
- **WHEN** outcomes are derived for a EuroLeague season
- **THEN** they contain `home_win` and `away_win` and no `draw`

#### Scenario: Overtime game
- **WHEN** a game is tied at the end of regulation and decided in overtime
- **THEN** the final score is the score after overtime and exactly one of `home_win` and `away_win` is true

#### Scenario: No totals column
- **WHEN** outcomes are derived for a EuroLeague season
- **THEN** they contain no `over_` or `under_` column

### Requirement: Source never fetches
The source's `index_items`, `catalogue`, `required_items` and `to_snapshots` SHALL be free of network access. Only the
store SHALL fetch, so a dry-run preparation and every extraction reach no network through this source.

#### Scenario: Transform runs offline
- **WHEN** `to_snapshots` runs on a recorded season payload with network access blocked
- **THEN** it returns the snapshots and opens no connection
