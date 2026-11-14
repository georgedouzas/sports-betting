# sources/source-contract Specification

## Purpose

Defines the contract every data source implements: the items it declares, the catalogue it reads, the payloads the
library fetches for it, the long snapshots it returns, and the schemas that validate statistics and odds snapshots.

## Requirements

### Requirement: Source identity

Every source SHALL expose the class attributes `name`, `kind` and `sport`. `kind` MUST be `stats` for a subclass of
`BaseStatsSource` and `odds` for a subclass of `BaseOddsSource`. `sport` MUST name the one sport of a statistics source
and MUST be `None` for an odds vendor that carries several sports and takes the sport of the source it is paired with.

#### Scenario: Statistics source identity

- **WHEN** a user reads `name`, `kind` and `sport` of `FootballDataStats()`
- **THEN** they are `football_data`, `stats` and `soccer`

#### Scenario: Multi-sport odds vendor

- **WHEN** a user reads `sport` of `OddsApi(key_env='ODDS_API_KEY', markets=['h2h'])`
- **THEN** it is `None`

### Requirement: Abstract source contract

A source SHALL implement `list_index_items`, `read_catalogue`, `list_required_items` and `to_snapshots`. Instantiating
a source class that leaves any of these methods unimplemented MUST raise `TypeError`.

#### Scenario: Base class cannot be instantiated

- **WHEN** a user calls `BaseSource()`
- **THEN** Python raises `TypeError` whose message mentions `abstract`

### Requirement: Statistics and odds sources are distinguishable

A statistics source SHALL be an instance of `BaseStatsSource` and an odds source SHALL be an instance of
`BaseOddsSource`, so a caller can tell the two kinds apart by type.

#### Scenario: Odds source is not a statistics source

- **WHEN** a caller checks `isinstance(source, BaseStatsSource)` for a concrete `BaseOddsSource` subclass
- **THEN** the check returns `False`

### Requirement: Catalogue discovery

`list_index_items(selection=None)` SHALL return the `RawItem` objects whose payloads describe what the source
publishes, where `selection` is a parameter grid or `None` for everything. `read_catalogue(payloads)` SHALL turn the
payloads of those items into a list of dictionaries with the keys `league`, `division` and `year`.

#### Scenario: Catalogue from index payloads

- **GIVEN** a source whose `read_catalogue` returns the England and Spain first divisions of 2024
- **WHEN** a caller passes it the payloads of its index items
- **THEN** it returns `[{'league': 'England', 'division': 1, 'year': 2024}, {'league': 'Spain', 'division': 1,
  'year': 2024}]`

### Requirement: Available parameters

`list_available_params()` SHALL fetch the source's index items through `fetch_payloads` with the source's
`request_url`, and return what `read_catalogue` makes of the payloads. It is an instance method because a configured
source, such as one with a key, publishes according to its configuration.

#### Scenario: Bundled catalogue read without the network

- **WHEN** a user calls `SampleSoccerStats().list_available_params()`
- **THEN** it returns `[{'division': 1, 'league': 'England', 'year': 2024}, {'division': 1, 'league': 'Spain',
  'year': 2024}]`

### Requirement: Required items

`list_required_items(params, schedule=None)` SHALL return the `RawItem` objects the selected parameter combinations
need. The result MUST be the same list in the same order for the same parameters. `schedule` carries the matches and
their kick-off instants for an odds source that prices by timestamp, and is `None` otherwise.

#### Scenario: Deterministic declaration

- **WHEN** a caller calls `list_required_items` twice with the same parameter list
- **THEN** the two results are equal

### Requirement: Fixtures items

`list_fixtures_items(params, schedule=None)` SHALL return the items whose payloads yield the upcoming matches. By
default it MUST return the same items as `list_required_items(params, schedule)`. A source whose upcoming matches live
in another file overrides it.

#### Scenario: Default fixtures items

- **WHEN** a caller calls `list_fixtures_items` on `SampleSoccerStats` with the England 2024 combination
- **THEN** it returns the same items as `list_required_items` with that combination

### Requirement: Schedule need

`needs_schedule()` SHALL return whether `list_required_items` needs the schedule of the matches. By default it MUST
return `False`, for a source that carries its own schedule.

#### Scenario: Source with its own schedule

- **WHEN** a caller calls `needs_schedule()` on `FootballDataOdds()`
- **THEN** it returns `False`

### Requirement: Request URL

`request_url(item)` SHALL return the URL to fetch an item from, adding any credential at the moment of the request. By
default it MUST return `item.url` unchanged.

#### Scenario: Default request URL

- **WHEN** a caller calls `request_url` of a source without an override on an item with URL `https://example.com/a.csv`
- **THEN** it returns `https://example.com/a.csv`

### Requirement: Raw items

A `RawItem` SHALL be an immutable record of `source`, `key` and `url`, where `url` is a web URL or a `file://` path
for a bundled file. Two items with equal `source`, `key` and `url` MUST be equal and hash the same, so a set of
declared items holds each item once.

#### Scenario: Duplicate declarations collapse

- **WHEN** a caller builds a set from two `RawItem('football_data', 'England_1_2024', 'a.csv')` objects and one
  `RawItem('football_data', 'Spain_1_2024', 'b.csv')`
- **THEN** the set has two members

### Requirement: Raw payloads

A `RawPayload` SHALL pair the fetched `item` with the `content` bytes of the response, kept verbatim and not parsed.

#### Scenario: Content kept verbatim

- **WHEN** a caller builds `RawPayload(item=item, content=b'league,year\nEngland,2024\n')`
- **THEN** `payload.content` equals those bytes and `payload.item` is `item`

### Requirement: Fetching payloads

`fetch_payloads(items, authorize)` SHALL read each item at the URL `authorize(item)` returns and return one
`RawPayload` per item in the order given. A `file://` URL MUST be read from disk. Every other URL MUST be read over
HTTP, concurrently with at most 20 connections, decoded and re-encoded as ISO-8859-1.

#### Scenario: Order kept

- **GIVEN** three files `a`, `b` and `c` on disk holding their own names
- **WHEN** a caller fetches items pointing at their `file://` URIs with `authorize` returning `item.url`
- **THEN** the payload contents are `b'a'`, `b'b'` and `b'c'` and each payload carries its own item

#### Scenario: URL taken from authorize

- **GIVEN** an item whose `url` is `https://example.invalid/never`
- **WHEN** a caller fetches it with an `authorize` that returns the `file://` URI of a file holding `content`
- **THEN** the payload content is `b'content'`

### Requirement: HTTP errors raise

`fetch_payloads` SHALL raise `aiohttp.ClientResponseError` when a server answers a request with an error status, and
MUST NOT return a payload for an error response.

#### Scenario: Missing remote file

- **WHEN** a caller fetches an item whose server answers with status 404
- **THEN** `fetch_payloads` raises `aiohttp.ClientResponseError`

### Requirement: Reading CSV content

`read_csv_content(content)` SHALL decode CSV bytes as ISO-8859-1 and return a data frame whose columns are the names in
the first row. Rows that do not parse MUST be skipped rather than raise.

#### Scenario: Rows read from bytes

- **WHEN** a caller calls `read_csv_content(b'home,away\nArsenal,Chelsea\nEverton,Fulham\n')`
- **THEN** it returns a frame with columns `home` and `away`, two rows, and `home` equal to `Arsenal` and `Everton`

### Requirement: Market outcomes

`derive_market_outcomes(home_points, away_points, markets)` SHALL return one integer 0 or 1 column per market:
`home_win`, `draw` and `away_win` compare the points, `over_<line>` is 1 when the total exceeds the line and
`under_<line>` is 1 when the total is below it. A market of any other name MUST produce no column. The index of a
series input MUST be kept.

#### Scenario: Three-way and totals markets

- **WHEN** a caller derives `home_win`, `draw`, `away_win` and `over_2.5` for home points `[2, 1, 0]` and away points
  `[1, 1, 3]`
- **THEN** the columns are `[1, 0, 0]`, `[0, 1, 0]`, `[0, 0, 1]` and `[1, 0, 1]`

#### Scenario: Two-way market

- **WHEN** a caller derives `home_win`, `away_win` and `over_188` for home points `[100, 95, 78]` and away points
  `[90, 97, 80]`
- **THEN** the columns are `[1, 0, 0]`, `[0, 1, 1]` and `[1, 1, 0]`

### Requirement: Market outcome input validation

`derive_market_outcomes` SHALL raise `ValueError` when the home and away points differ in length or cannot be cast to
integers.

#### Scenario: Inconsistent lengths

- **WHEN** a caller passes two home points and three away points
- **THEN** it raises `ValueError` with the message `Found input variables with inconsistent numbers of samples: [2, 3]`

#### Scenario: Non-numeric points

- **WHEN** a caller passes away points `['point', '1', '3']`
- **THEN** it raises `ValueError`

### Requirement: Snapshot output

`to_snapshots(payloads)` SHALL return a long data frame with one row per match and moment, carrying `event_status`,
`event_time`, the kick-off instant `date` in UTC, `league`, `division`, `year`, `home_team` and `away_team`, so that
`date + event_time` is the wall-clock instant of a snapshot. An odds source MUST also carry a `provider` column and one
column per market.

#### Scenario: Statistics source builds moments

- **GIVEN** the `MyStats` example source of the `BaseStatsSource` documentation and one played match scored 2 to 1
- **WHEN** a caller calls `to_snapshots` with its payload
- **THEN** the event statuses are `postplay` and `preplay` and the post-play `home_win` is 1

### Requirement: Sources perform no input or output

`list_index_items`, `read_catalogue`, `list_required_items` and `to_snapshots` SHALL only declare items and transform
payloads. The library performs all reading through `fetch_payloads`, so a source works on payloads built in memory.

#### Scenario: Transform without fetching

- **WHEN** a test passes a `RawPayload` built from bytes in memory to `FootballDataStats().to_snapshots`
- **THEN** it returns snapshots without any network request

### Requirement: Snapshot column definitions

`required_col(alias=None)` SHALL define a non-nullable column marked as part of the snapshot identity.
`optional_col(include, fixed, alias=None)` SHALL define a nullable column whose metadata records the event statuses
`include` at which it is meaningful and whether it is `fixed` within a match.

#### Scenario: Required column

- **WHEN** a schema declares `home_team: str = required_col()`
- **THEN** the column is not nullable and its metadata has `snapshot` set to `True`

#### Scenario: Optional column metadata

- **WHEN** a schema declares `stadium_capacity: float = optional_col(include=['preplay'], fixed=True)`
- **THEN** the column metadata has `include` equal to `['preplay']` and `fixed` equal to `True`

### Requirement: Schema metadata accessors

A snapshot schema SHALL list its snapshot identity columns through `list_snapshot_cols()` and return the `include`,
`fixed` and `snapshot` metadata of a column through `get_col_metadata(col)`, empty when the column carries none.

#### Scenario: Identity columns listed

- **WHEN** a caller calls `list_snapshot_cols()` on a statistics schema declaring `date`, `league`, `division`,
  `year`, `home_team` and `away_team` with `required_col`
- **THEN** the result holds `event_status`, `event_time` and those six columns

### Requirement: Event status and time validation

`BaseStatsSchema` and `BaseOddsSchema` SHALL require the columns `event_status` and `event_time`. Validation MUST raise
`pandera.errors.SchemaError` when a status is not `preplay`, `inplay` or `postplay`, when a pre-play time is negative,
when an in-play time is not positive, or when a post-play time is not zero.

#### Scenario: Unknown status

- **WHEN** a row of valid snapshots gets the status `halftime` and the schema validates the frame
- **THEN** validation raises `SchemaError`

#### Scenario: In-play at minute zero

- **WHEN** an `inplay` row gets an event time of zero and the schema validates the frame
- **THEN** validation raises `SchemaError`

#### Scenario: Post-play with a minute

- **WHEN** a `postplay` row gets an event time of ten minutes and the schema validates the frame
- **THEN** validation raises `SchemaError`

### Requirement: Snapshot uniqueness

Schema validation SHALL raise `pandera.errors.SchemaError` when two rows share the same values in every snapshot
identity column.

#### Scenario: Duplicated row

- **WHEN** a frame of valid snapshots gets a copy of its first row appended and the schema validates it
- **THEN** validation raises `SchemaError`

### Requirement: Strict columns

Schema validation SHALL be strict. It MUST raise `pandera.errors.SchemaError` or `pandera.errors.SchemaErrors` for a
column the schema does not declare, and `pandera.errors.SchemaError` for a missing required column. A valid frame MUST
be returned unchanged.

#### Scenario: Undeclared column

- **WHEN** a frame of valid snapshots gets a column `unexpected` and the schema validates it
- **THEN** validation raises `SchemaError` or `SchemaErrors`

#### Scenario: Missing required column

- **WHEN** a frame of valid snapshots loses `home_team` and the schema validates it
- **THEN** validation raises `SchemaError`

#### Scenario: Valid frame

- **WHEN** the schema validates a frame of valid statistics snapshots
- **THEN** the returned frame equals the input

### Requirement: Odds schema markets

`BaseOddsSchema.list_odds_cols()` SHALL return every declared column that is neither a snapshot identity column nor
`provider`. Validation MUST raise `pandera.errors.SchemaError` when a `postplay` row carries a price in any of these
columns.

#### Scenario: Price after the match

- **WHEN** a `postplay` row of valid odds snapshots gets the price 1.23 in every market column and the schema
  validates the frame
- **THEN** validation raises `SchemaError`
