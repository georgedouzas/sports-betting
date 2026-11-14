## Purpose

The NBA statistics source reads ESPN's public feed and turns the league's games into long statistics snapshots, so the
existing basketball dataloader can build NBA training data and fixtures.

## ADDED Requirements

### Requirement: Free NBA statistics source
The library SHALL provide `NBAStats`, exported from `sportsbet.datasets`, a statistics source for the NBA that takes no
arguments, is free and requires no credential. Its data SHALL be fetched on the user's machine during preparation, and
its cost estimate SHALL be zero for every item. The project SHALL ship no NBA data.

#### Scenario: Source needs no credential
- **WHEN** a user constructs `NBAStats()` with no arguments
- **THEN** the source is ready to use and requires no key

#### Scenario: Statistics cost nothing
- **WHEN** a dry-run preparation prices the items `NBAStats` requires
- **THEN** the estimated cost of those items is zero

### Requirement: NBA through the basketball dataloader
The NBA SHALL be usable through the existing `BasketballDataLoader` by passing `stats=NBAStats()`, with no new
dataloader and no new sport. Training data and fixtures SHALL come from the same steps and objects used for soccer and
the EuroLeague.

#### Scenario: Prepare and extract an NBA season
- **WHEN** a user prepares `BasketballDataLoader(param_grid={'league': ['NBA'], 'year': [2025]}, stats=NBAStats(),
  odds=OddsApi(key=...))` and extracts training data
- **THEN** they receive the games of the selected season with their outcomes and their prices

#### Scenario: Fixtures carry form and no outcome
- **WHEN** the user extracts fixtures from the same dataloader
- **THEN** they receive the games not yet played, with the form of the teams going into them and no outcome

#### Scenario: Only seasons both sources publish are offered
- **WHEN** the feed lists a season that the odds vendor does not cover
- **THEN** the dataloader does not offer that season

#### Scenario: No odds source
- **WHEN** a user extracts training data from a basketball dataloader with `NBAStats` and no odds source
- **THEN** it fails with a message saying there are no markets to predict, not with an empty dataset or a schema error

### Requirement: Seasons read from the feed
The source SHALL read the seasons it publishes from the feed's season index and SHALL NOT offer a season from a
fabricated range of years. Each season SHALL carry `league` `NBA`, `division` `1` and `year` equal to the year the
season ends, which is the feed's own season year with no conversion.

#### Scenario: Catalogue comes from the index
- **WHEN** the source reads the season index payload
- **THEN** it returns one parameter set for each season listed and no other

#### Scenario: No year conversion
- **WHEN** the feed lists season year 2026 for the season that starts in October 2025
- **THEN** the source reports it with `year` 2026

### Requirement: Complete seasons through monthly requests
The source SHALL return every game of a selected season. Because the feed silently truncates a request at 1,000
events, the source SHALL request one calendar month per item, from September of the year before the season's `year` to
July of that `year`, so each request stays far below the limit.

#### Scenario: Every game is returned
- **WHEN** a season's games are requested and transformed
- **THEN** the result contains every game of its regular season, play-in and post-season

#### Scenario: Each request spans one month
- **WHEN** the source lists the items required for one season
- **THEN** it returns eleven items, each spanning a single calendar month from September to July

### Requirement: Only real games between real clubs
The source SHALL exclude pre-season games and all-star exhibitions, and SHALL include the regular season, the play-in
and the post-season. The exclusion SHALL NOT rely on the feed's season label alone, which files the exhibition under
the regular season. A game with an unrecognised competition label SHALL be excluded rather than admitted.

#### Scenario: Exhibition is excluded
- **WHEN** a season contains an all-star exhibition labelled as regular season with teams `Team Stars` and `World`
- **THEN** no exhibition game and no invented team appears in the result

#### Scenario: Pre-season is excluded
- **WHEN** a season contains pre-season games
- **THEN** no pre-season game appears in the result

#### Scenario: Post-season is included
- **WHEN** a season contains playoff games whose competition label is not the regular-season label
- **THEN** those playoff games appear in the result

#### Scenario: Unrecognised label is excluded
- **WHEN** a game carries a competition label the source does not recognise
- **THEN** the game is excluded

#### Scenario: Exactly the league's clubs
- **WHEN** a complete season is transformed
- **THEN** the number of distinct teams is exactly 30

### Requirement: Tip-off in UTC
The `date` of every NBA game SHALL be the instant the game starts, in UTC, read from the feed's UTC timestamp and never
inferred.

#### Scenario: UTC timestamp is read
- **WHEN** the feed gives a game the date `2026-01-01T23:00Z`
- **THEN** the snapshot `date` is 2026-01-01 23:00 UTC

### Requirement: Played and unplayed games
The source SHALL produce a pre-play snapshot for every kept game, carrying the form each team brought into it. It SHALL
produce a post-play snapshot with the final score and the outcome only for a game that has been played, decided per
game and never from the season being over. It SHALL carry results for a season in progress.

#### Scenario: Season in progress
- **WHEN** the statistics of a season in progress are read
- **THEN** the games already played carry their results and the games still to come do not

#### Scenario: Postponed game in a finished season
- **WHEN** a finished season contains a game that was postponed and never played
- **THEN** that game has only a pre-play snapshot and is a fixture, not a training row

#### Scenario: Season with no games played
- **WHEN** a season's schedule is published and no game has been played
- **THEN** the source yields fixtures and no training rows, without an error

### Requirement: Two-way outcome
The outcome of an NBA game SHALL be `home_win` and `away_win` only. It SHALL NOT include a draw, because a tie goes to
overtime, and SHALL NOT include a totals market, because the line moves from game to game.

#### Scenario: No draw and no totals
- **WHEN** outcomes are derived for an NBA season
- **THEN** they contain `home_win` and `away_win`, and no `draw`, `over_` or `under_` column

### Requirement: Form features without leakage
The source SHALL produce per-team form features over points scored, points conceded and wins, as an expanding mean and a
rolling mean over the last three games. The features SHALL be computed only from earlier games, so a game never sees its
own result, and a fixture SHALL carry the form of the games played before it.

#### Scenario: A game does not see its own result
- **WHEN** form is computed for a played game
- **THEN** the form values use only the games each team played before it

#### Scenario: Fixture carries prior form
- **WHEN** form is computed for an upcoming game
- **THEN** it carries the form of the games each team played before it

### Requirement: Source never fetches
The source's `index_items`, `catalogue`, `required_items` and `to_snapshots` SHALL be free of network access. Only
preparation SHALL download.

#### Scenario: Transform runs offline
- **WHEN** `to_snapshots` runs on recorded monthly payloads with network access blocked
- **THEN** it returns the snapshots and opens no connection
