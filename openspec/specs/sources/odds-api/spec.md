# sources/odds-api Specification

## Purpose

Defines `OddsApi`, the paid odds source that reads time-stamped historical and live bookmaker prices from The Odds API,
pricing each match at the moments the statistics carry, with the key read from an environment variable.

## Requirements

### Requirement: Odds API identity and settings

`OddsApi(key_env, markets=None, regions=None, moments=None)` SHALL have the name `odds_api`, the kind `odds` and the
sport `None`. It MUST keep its arguments as attributes of the same names. `markets` defaults to `['h2h', 'totals']`,
`regions` defaults to `['eu']`, and `moments` holds `(event_status, minutes)` pairs.

#### Scenario: Settings kept

- **WHEN** a user builds `OddsApi(key_env='ODDS_API_KEY', markets=['h2h'], regions=['eu'])`
- **THEN** `key_env`, `markets`, `regions` and `moments` are `ODDS_API_KEY`, `['h2h']`, `['eu']` and `None`

### Requirement: Credential by variable name

`request_url(item)` SHALL append `apiKey=<value>` to the item URL, after `&` when the URL has a query and `?` otherwise,
where the value is read from the environment variable named by `key_env` at the moment of the request. No declared
item MUST carry the key. A missing variable MUST raise `KeyError` naming it.

#### Scenario: Key added at request time

- **GIVEN** the variable `ODDS_API_KEY` holds `secret`
- **WHEN** a caller calls `request_url` on an item with URL `https://api.the-odds-api.com/v4/sports?all=true`
- **THEN** it returns `https://api.the-odds-api.com/v4/sports?all=true&apiKey=secret`

#### Scenario: Key never stored

- **WHEN** the source declares the items of a schedule
- **THEN** no item URL holds the key or `apiKey`

### Requirement: Odds API catalogue

`list_index_items` SHALL declare one item keyed `sports` at `https://api.the-odds-api.com/v4/sports?all=true`.
`read_catalogue` SHALL return each vendor sport the library maps, as its league and division, for every year from 2021
to the current year plus one. A vendor sport the library does not map MUST be left out. Without payloads it returns an
empty list.

#### Scenario: Unmapped competition left out

- **WHEN** the vendor lists `soccer_epl`, `soccer_spain_la_liga` and `americanfootball_nfl`
- **THEN** the catalogue leagues are `England` and `Spain`

#### Scenario: History start

- **WHEN** the catalogue is read
- **THEN** its earliest year is 2021 and it holds no 2019 season

### Requirement: Mapped competitions

The source SHALL map the vendor sports to the leagues of the statistics sources: the English first to fourth
divisions, the Scottish, Dutch, Belgian, Portuguese, Turkish and Greek first divisions, the German, Italian, Spanish
and French first and second divisions, the first divisions of Argentina, Austria, Brazil, China, Denmark, Finland,
Ireland, Japan, Mexico, Norway, Poland, Russia, Sweden, Switzerland and USA, and the basketball leagues `Euroleague`,
`NBA`, `WNBA`, `NCAAB` and `NBL`.

#### Scenario: NBA mapped

- **WHEN** the vendor lists `basketball_nba`
- **THEN** the catalogue holds `NBA` division 1

### Requirement: Schedule required

`needs_schedule()` SHALL return `True`. `list_required_items` MUST return an empty list when the schedule is `None` or
empty, since a season alone does not say when its matches are played.

#### Scenario: No schedule

- **WHEN** a caller calls `list_required_items([{'league': 'England', 'division': 1, 'year': 2025}])`
- **THEN** it returns an empty list

### Requirement: Priced moments

When `moments` is `None`, the source SHALL price every distinct pair of `event_status` and `event_time` in minutes that
the schedule carries, except `postplay`. When `moments` is given, it MUST price exactly those pairs.

#### Scenario: Moments from the statistics

- **WHEN** the schedule carries `preplay` at 0, `inplay` at 45 and `postplay` at 0
- **THEN** the source declares historical items for `preplay` at 0 and `inplay` at 45 only

### Requirement: Historical snapshot items

For each league, division and year of the schedule that maps to a vendor sport, and each priced moment, the source
SHALL declare one item per distinct snapshot instant: one minute before kick-off for `preplay`, and kick-off plus the
minutes for `inplay`. Matches sharing an instant MUST share the item. Instants before 2020-06-06 or not before now
MUST be skipped.

#### Scenario: Matches kicking off together

- **WHEN** the schedule holds two matches with the same kick-off and two priced moments
- **THEN** the source declares two historical items

### Requirement: Historical request format

A historical item SHALL request `https://api.the-odds-api.com/v4/historical/sports/<sport>/odds` with the query
parameters `regions` and `markets` joined by commas, `oddsFormat=decimal` and `date` as `YYYY-MM-DDTHH:MM:SSZ`, and
SHALL be keyed `<sport>__<year>__<YYYYMMDDTHHMMSSZ>__<event_status>__<minutes>`.

#### Scenario: In-play item key

- **WHEN** an England 2025 match kicks off at `2024-08-16 19:00` UTC and `inplay` at 45 is priced
- **THEN** the source declares an item keyed `soccer_epl__2025__20240816T194500Z__inplay__45`

### Requirement: Live item

For each league, division and year of the schedule that maps to a vendor sport, the source SHALL declare one item
keyed `<sport>__<year>__live` requesting `https://api.the-odds-api.com/v4/sports/<sport>/odds` with the same
`regions`, `markets` and `oddsFormat` query parameters and no `date`.

#### Scenario: Live item declared

- **WHEN** the source declares the items of an England 2025 schedule
- **THEN** one item is keyed `soccer_epl__2025__live`

### Requirement: Historical snapshot rows

For a historical payload, `to_snapshots` SHALL keep only the events whose kick-off plus the moment offset lies within
five minutes of the item's instant. The rows MUST take their status and minutes from the item, and their league,
division and year from the item rather than from the kick-off.

#### Scenario: Price at the minute

- **WHEN** the in-play 45 item returns two matches priced by `pinnacle` with a home price of 1.8
- **THEN** every row has status `inplay`, event time 45, provider `pinnacle` and `home_win` 1.8

#### Scenario: Unrequested match left out

- **WHEN** the in-play 45 snapshot of a 19:00 kick-off also prices a match that kicked off at 14:00
- **THEN** only the 19:00 match is returned

#### Scenario: Season from the statistics

- **WHEN** a match kicking off in August 2024 belongs to the 2025 schedule
- **THEN** its rows have year 2025, league `England` and division 1

### Requirement: Live snapshot rows

For a live payload, `to_snapshots` SHALL take the latest bookmaker `last_update` as the current instant. A match whose
kick-off is before it MUST be `inplay` at the elapsed minutes rounded to the nearest five. Any other match MUST be
`preplay` at minute zero.

#### Scenario: Running match

- **WHEN** a match kicked off at 19:00 and the latest update is at 19:30
- **THEN** its rows are `inplay` at 30 minutes

#### Scenario: Upcoming match

- **WHEN** a match kicks off at 19:00 and the latest update is at 18:30
- **THEN** its rows are `preplay` at 0 minutes

### Requirement: Odds API markets

`to_snapshots` SHALL return one row per event and bookmaker, with the bookmaker key as `provider` and `date` as the
kick-off in UTC. The `h2h` outcomes named after the home team, the away team and `Draw` MUST become `home_win`,
`away_win` and `draw`. The `totals` outcomes `Over` and `Under` at the point 2.5 MUST become `over_2.5` and `under_2.5`,
and other points MUST be ignored. A bookmaker with no such outcome yields no row.

#### Scenario: Markets renamed

- **WHEN** a bookmaker prices home 1.8, away 4.2, draw 3.5, over 2.5 at 1.9, under 2.5 at 1.95 and over 3.5 at 2.6
- **THEN** the row has `home_win` 1.8, `away_win` 4.2, `draw` 3.5, `over_2.5` 1.9 and `under_2.5` 1.95

### Requirement: Empty odds

`to_snapshots` SHALL return an empty data frame when the payloads hold no priced event.

#### Scenario: No payloads

- **WHEN** a caller calls `to_snapshots([])`
- **THEN** it returns an empty data frame
