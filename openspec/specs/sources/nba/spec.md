# sources/nba Specification

## Purpose

Defines `NBAStats`, the free basketball statistics source that reads the NBA schedule and final scores from ESPN,
updated through a season, and turns them into pre-play and post-play snapshots with two-way markets.

## Requirements

### Requirement: NBA source identity

`NBAStats` SHALL have the name `nba`, the kind `stats` and the sport `basketball`. It publishes the league `NBA` with
the single division 1 and its requests carry no credential.

#### Scenario: Identity

- **WHEN** a user reads `name`, `kind` and `sport` of `NBAStats()`
- **THEN** they are `nba`, `stats` and `basketball`

### Requirement: NBA catalogue

`list_index_items` SHALL declare one item keyed `seasons` at
`https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons?limit=100`. `read_catalogue` SHALL return
one combination per season reference the feed lists, taking the year from the last path segment of the reference
without conversion, since the feed already names a season by the year it ends in. The result is sorted by year.

#### Scenario: Seasons from the feed

- **WHEN** the feed lists the season references ending in `/2026`, `/2025` and `/2024`
- **THEN** the catalogue holds the `NBA` division 1 seasons 2024, 2025 and 2026 in that order

### Requirement: One request per month

`list_required_items(params)` SHALL declare, for each selected combination whose league is `NBA`, eleven items, one
per calendar month from September of the year before `year` to July of `year`. Each item MUST be keyed
`NBA_<division>_<year>_<YYYYMM>` and request the ESPN scoreboard at
`https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard` with `dates=<first day>-<last day>` of the
month and `limit=1000`. Combinations of other leagues MUST be ignored.

#### Scenario: Season 2026 windows

- **WHEN** a user calls `list_required_items([{'league': 'NBA', 'division': 1, 'year': 2026}])`
- **THEN** it returns eleven items, each window starts on the first and ends in the same month, the first window
  starts in `202509` and the last in `202607`

### Requirement: Competition games only

`to_snapshots` SHALL keep regular season and play-off games. It MUST drop pre-season games, which the feed files under
season type 1, and exhibition games whose competition the feed labels `ALLSTAR`.

#### Scenario: All-star game dropped

- **WHEN** a month holds all-star games between `Team Stripes`, `Team Stars` and `World`, filed under the regular
  season
- **THEN** none of those teams appears in the snapshots

#### Scenario: Pre-season dropped

- **WHEN** a month holds a pre-season game between Philadelphia 76ers and Brooklyn Nets
- **THEN** neither team appears in the snapshots

#### Scenario: Play-off kept

- **WHEN** a month holds a play-off game of Los Angeles Lakers against New York Knicks
- **THEN** the game has a `preplay` and a `postplay` row

### Requirement: NBA games

`to_snapshots` SHALL read each kept event with the `home` and `away` competitors' display names as teams, the event
`date` as the tip-off in UTC without a time zone, and the score only when the competition status is completed. An event
without a home or away competitor or without a date MUST be skipped. Months of the same season MUST be combined and
sorted by date before the form is computed.

#### Scenario: Tip-off in UTC

- **WHEN** the feed dates a Boston Celtics home game `2025-10-21T23:30Z`
- **THEN** its `date` is `2025-10-21 23:30`

### Requirement: NBA snapshots

`to_snapshots` SHALL return a `preplay` row at time zero for every kept game and a `postplay` row at time zero for
every completed game, carrying `home_points`, `away_points`, `home_win` and `away_win`. There MUST be no `draw` market
and no `over_` or `under_` market.

#### Scenario: Completed game

- **WHEN** Boston Celtics beat New York Knicks 120 to 96 at home
- **THEN** the post-play row has `home_points` 120, `away_points` 96, `home_win` 1 and `away_win` 0

#### Scenario: Game never played

- **WHEN** a season holds a game that was not completed
- **THEN** that game has only a `preplay` row

#### Scenario: Two-way markets only

- **WHEN** the source transforms a month of games
- **THEN** no column is named `draw` or starts with `over_` or `under_`

### Requirement: NBA form features

The pre-play row SHALL carry, for the home and the away team, the mean over all earlier games of the season (`_avg`)
and over the last three earlier games (`_latest_avg`) of `points_for`, `points_against` and `wins`, named such as
`home_points_for_avg`. A feature MUST never include the result of its own game.

#### Scenario: Form before the game

- **WHEN** Boston Celtics won 120 to 96 and then play at home again
- **THEN** the season's first game has no `home_points_for_avg`, and the later pre-play row has `home_points_for_avg`
  120 and `home_wins_avg` 1

### Requirement: Empty NBA payloads

`to_snapshots` SHALL return an empty data frame when no payload holds a kept game.

#### Scenario: No payloads

- **WHEN** a caller calls `NBAStats().to_snapshots([])`
- **THEN** it returns an empty data frame
