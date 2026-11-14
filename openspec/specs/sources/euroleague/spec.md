# sources/euroleague Specification

## Purpose

Defines `EuroLeagueStats`, the free basketball statistics source that reads the EuroLeague schedule and final scores
from the competition's official API and turns them into pre-play and post-play snapshots with two-way markets.

## Requirements

### Requirement: EuroLeague source identity

`EuroLeagueStats` SHALL have the name `euroleague`, the kind `stats` and the sport `basketball`. It publishes the league
`Euroleague` with the single division 1 and its requests carry no credential.

#### Scenario: Identity

- **WHEN** a user reads `name`, `kind` and `sport` of `EuroLeagueStats()`
- **THEN** they are `euroleague`, `stats` and `basketball`

### Requirement: EuroLeague catalogue

`list_index_items` SHALL declare one item keyed `seasons` at
`https://api-live.euroleague.net/v2/competitions/E/seasons`. `read_catalogue` SHALL return one combination per season
the API lists, named by the year the season ends in, which is the API's season year plus one, sorted by year. Without
payloads it returns an empty list.

#### Scenario: Seasons from the API

- **WHEN** the API lists the seasons `E2024` with year 2024 and `E2023` with year 2023
- **THEN** the catalogue is `[{'league': 'Euroleague', 'division': 1, 'year': 2024}, {'league': 'Euroleague',
  'division': 1, 'year': 2025}]`

### Requirement: One request per season

`list_required_items(params)` SHALL declare one item per selected combination whose league is `Euroleague`, keyed
`Euroleague_<division>_<year>` and requesting
`https://api-live.euroleague.net/v2/competitions/E/seasons/E<code>/games`, where `<code>` is the year before `year`.
Combinations of other leagues MUST be ignored. The upcoming games come from the same items.

#### Scenario: Season 2025

- **WHEN** a user calls `list_required_items([{'league': 'Euroleague', 'division': 1, 'year': 2025}])`
- **THEN** it returns one item keyed `Euroleague_1_2025` whose URL holds `seasons/E2024/games`

#### Scenario: Other league ignored

- **WHEN** a user calls `list_required_items([{'league': 'NBA', 'division': 1, 'year': 2025}])`
- **THEN** it returns an empty list

### Requirement: EuroLeague games

`to_snapshots` SHALL read each game of the payload with the `local` club as the home team, the `road` club as the away
team, and `date` as the API's `utcDate` in UTC without a time zone. The score MUST be used only when the game is
flagged `played`. A game without both club names or without `utcDate` MUST be skipped.

#### Scenario: Tip-off in UTC

- **WHEN** the API gives a game in Istanbul the `utcDate` `2024-10-03T16:30:00Z`
- **THEN** its `date` is `2024-10-03 16:30`

### Requirement: EuroLeague snapshots

`to_snapshots` SHALL return a `preplay` row at time zero for every game and a `postplay` row at time zero for every
played game, carrying `home_points`, `away_points`, `home_win` and `away_win`. There MUST be no `draw` market and no
`over_` or `under_` market, since a basketball game cannot end level and its totals line moves.

#### Scenario: Played game

- **WHEN** Alba Berlin lost 77 to 87 at home
- **THEN** its post-play row has `home_points` 77, `away_points` 87, `home_win` 0 and `away_win` 1

#### Scenario: Unplayed game

- **WHEN** the season holds three played games and one unplayed game
- **THEN** the source returns seven rows and the unplayed game has only a `preplay` row

#### Scenario: Two-way markets only

- **WHEN** the source transforms a season
- **THEN** no column is named `draw` or starts with `over_` or `under_`

### Requirement: EuroLeague form features

The pre-play row SHALL carry, for the home and the away team, the mean over all earlier games of the season (`_avg`)
and over the last three earlier games (`_latest_avg`) of `points_for`, `points_against` and `wins`, named such as
`home_points_for_avg`. A feature MUST never include the result of its own game.

#### Scenario: Form before the game

- **WHEN** Panathinaikos Athens won 87 to 77 and then plays at home
- **THEN** the first game of the season has no `home_points_for_avg`, and the later pre-play row has
  `home_points_for_avg` 87 and `home_wins_avg` 1

### Requirement: Empty EuroLeague payloads

`to_snapshots` SHALL return an empty data frame when no payload holds a game.

#### Scenario: No payloads

- **WHEN** a caller calls `EuroLeagueStats().to_snapshots([])`
- **THEN** it returns an empty data frame
