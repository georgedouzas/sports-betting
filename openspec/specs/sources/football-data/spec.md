# sources/football-data Specification

## Purpose

Defines `FootballDataStats` and `FootballDataOdds`, the free soccer statistics and odds sources read from the
football-data.co.uk feed, which publishes the schedule, results, match statistics and bookmaker prices in one file.

## Requirements

### Requirement: Football-data source identity

`FootballDataStats` and `FootballDataOdds` SHALL share the name `football_data` and the sport `soccer`, with the kinds
`stats` and `odds`. They read the same files and need no credential.

#### Scenario: Statistics identity

- **WHEN** a user reads `name`, `kind` and `sport` of `FootballDataStats()`
- **THEN** they are `football_data`, `stats` and `soccer`

#### Scenario: Shared files declared once

- **GIVEN** a statistics and an odds source that read the same catalogue
- **WHEN** both declare the items of the England first division of 2025
- **THEN** the two lists are equal, so the file is fetched once

### Requirement: Covered leagues

The sources SHALL cover England, Scotland, Germany, Italy, Spain, France, Netherlands, Belgium, Portugal, Turkey and
Greece through one index page each at `https://www.football-data.co.uk/<league in lower case>m.php`. They SHALL cover
Argentina, Austria, Brazil, China, Denmark, Finland, Ireland, Japan, Mexico, Norway, Poland, Romania, Russia, Sweden,
Switzerland and USA through one whole-history file each at `https://www.football-data.co.uk/new/<code>.csv`.

#### Scenario: Index page of a league

- **WHEN** a user calls `FootballDataStats().list_index_items({'league': ['Italy']})`
- **THEN** it returns one item with key `index_Italy` and URL `https://www.football-data.co.uk/italym.php`

#### Scenario: Whole-history league

- **WHEN** a user calls `FootballDataStats().list_index_items({'league': ['Brazil']})`
- **THEN** it returns one item with key `Brazil_1` and URL `https://www.football-data.co.uk/new/BRA.csv`

### Requirement: Index items follow the selection

`list_index_items(selection)` SHALL declare the index pages and whole-history files of the leagues the selection
names, reading a list of grids as a whole. When the selection is `None`, empty, or holds a grid without a `league`
key, it MUST declare every league.

#### Scenario: Several grids

- **WHEN** a user calls `list_index_items([{'league': ['Italy']}, {'league': ['Brazil']}])`
- **THEN** the declared keys are `Brazil_1` and `index_Italy`

#### Scenario: Grid without a league

- **WHEN** a user calls `list_index_items([{'league': ['Italy']}, {'year': [2020]}])`
- **THEN** it declares as many items as `list_index_items()`

### Requirement: Football-data catalogue

`read_catalogue` SHALL return, sorted, the league, division and year of every season file an index page links to and
of every season a whole-history file holds. A season is named by the year it ends in. A two-digit season year at or
below 68 MUST map to the 2000s and above 68 to the 1900s. Feed division `0` to `3` of England and Scotland MUST map to
divisions 1 to 4, `C` to division 5, and other leagues keep their digit.

#### Scenario: Season file of the Premier League

- **WHEN** an England index page links to `mmz4281/2425/E0.csv`
- **THEN** the catalogue holds `{'league': 'England', 'division': 1, 'year': 2025}`

### Requirement: Football-data required items

`list_required_items(params)` SHALL return one item per selected combination found in the catalogue the source last
read, keyed `<league>_<division>_<year>` with the catalogue URL. A whole-history league MUST be declared once, keyed
`<league>_1`, whatever seasons are selected. A combination absent from that catalogue, or any combination before a
catalogue is read, MUST yield no item.

#### Scenario: Before the catalogue is read

- **WHEN** a user calls `FootballDataStats().list_required_items` with the England 2025 combination on a new source
- **THEN** it returns an empty list

### Requirement: Football-data fixtures items

`list_fixtures_items(params)` SHALL return the latest season file of each selected league and division in the
catalogue, followed by the item keyed `fixtures` at `https://www.football-data.co.uk/fixtures.csv`.

#### Scenario: Season in progress and fixtures file

- **GIVEN** a catalogue holding the England first division of 2025
- **WHEN** a user calls `list_fixtures_items` with that combination
- **THEN** it returns the `England_1_2025` item and then the `fixtures` item

### Requirement: Kick-off instant

The sources SHALL set `date` to the kick-off instant in UTC, read from the feed's `Date` and `Time` in UK time
(`Europe/London`), as a timestamp without a time zone. A match without a `Time` MUST start at midnight UK time.

#### Scenario: Summer kick-off

- **WHEN** the feed lists a match on 10/08/2024 at 20:00
- **THEN** its `date` is `2024-08-10 19:00`

#### Scenario: Winter kick-off

- **WHEN** the feed lists a match on 26/12/2024 at 15:00
- **THEN** its `date` is `2024-12-26 15:00`

### Requirement: Football-data statistics moments

`FootballDataStats.to_snapshots` SHALL return for each played match a `preplay` row at time zero with form features,
an `inplay` row at 45 minutes with the half-time goals when the feed has them, and a `postplay` row at time zero with
the full-time goals. The in-play and post-play rows MUST carry `home_win`, `draw`, `away_win`, `over_2.5` and
`under_2.5` resolved at that moment. `event_time` is given in minutes.

#### Scenario: Every moment

- **WHEN** the source transforms a season of four played matches with half-time scores
- **THEN** it returns twelve rows with the statuses `preplay`, `inplay` and `postplay`, and the in-play time is 45

#### Scenario: Half-time goals

- **WHEN** a match was 0 to 2 at half time and 1 to 3 at full time
- **THEN** its in-play `home_goals` is 0 and its post-play `home_goals` is 1

#### Scenario: Wall-clock address

- **WHEN** a match kicks off at `2024-08-10 19:00`
- **THEN** its in-play `date` plus its `event_time` in minutes is `2024-08-10 19:45`

### Requirement: Football-data form features

The pre-play row SHALL carry, for the home and the away team, the mean over all earlier matches of the season
(`_avg`) and over the last three earlier matches (`_latest_avg`) of `points`, `adj_points`, `goals_for`,
`goals_against`, `adj_goals_for` and `adj_goals_against`, named such as `home_points_avg`. A feature MUST never include
the result of its own match. The adjusted goals of a team in a match SHALL be its goals, plus its red cards, plus a
twentieth of its yellow cards, plus a tenth of its shots on target, plus a fiftieth of its shots, plus a hundredth of
its corners, and the adjusted goals against SHALL be the same sum over the opponent's numbers.

#### Scenario: First match has no form

- **WHEN** the source transforms a season whose first match is Arsenal at home
- **THEN** that match's `home_points_avg` is missing

#### Scenario: Form from the earlier match

- **WHEN** Chelsea lost its first match and then plays at home
- **THEN** that pre-play `home_points_avg` is 0

#### Scenario: Adjusted goals count each statistic once

- **GIVEN** a team that in its first match scored 2 goals with 1 red card, 2 yellow cards, 5 shots on target, 10 shots
  and 4 corners
- **WHEN** the source transforms the season
- **THEN** the `adj_goals_for` average that team carries into its second match is 3.84

### Requirement: Football-data odds snapshots

`FootballDataOdds.to_snapshots` SHALL return one `preplay` row at time zero per match and provider, for the providers
`market_average` and `market_maximum`, with the prices of `home_win`, `draw`, `away_win`, `over_2.5` and `under_2.5`.
A price column the feed lacks or leaves empty MUST be filled from its closing price column.

#### Scenario: Providers exploded

- **WHEN** the source transforms a season with average and maximum prices
- **THEN** the providers are `market_average` and `market_maximum`, every status is `preplay`, and Arsenal's average
  `home_win` is 1.80

#### Scenario: Closing price fills the gap

- **WHEN** the feed carries only the closing average columns `AvgCH`, `AvgCD` and `AvgCA`
- **THEN** Arsenal's `market_average` `home_win` is 1.80

### Requirement: Football-data upcoming matches

When the payloads include the `fixtures` item, the sources SHALL add the fixtures dated today or later to the latest
season of their league and division. An upcoming match MUST get only a `preplay` statistics row, with form features
from the season's played matches, and only `preplay` odds rows.

#### Scenario: Fixture has no result

- **WHEN** the fixtures file lists a match of the England first division dated after today
- **THEN** the statistics hold a `preplay` row for it and no `inplay` or `postplay` row

### Requirement: Unplayed season matches

A match of a season file without a full-time score SHALL get only a `preplay` statistics row, and no `inplay` row
without a half-time score.

#### Scenario: Postponed match

- **WHEN** a season file lists a match with empty goal columns
- **THEN** the statistics hold only a `preplay` row for it
