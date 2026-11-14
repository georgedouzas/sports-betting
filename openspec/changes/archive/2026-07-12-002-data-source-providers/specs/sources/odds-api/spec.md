## Purpose

The keyed The Odds API odds source buys time-stamped historical, upcoming and in-play prices with the user's own key, so
an in-play target can be priced at the moment it occurred.

## ADDED Requirements

### Requirement: Keyed commercial odds source
The library SHALL provide `OddsApi(key, markets=None, regions=None)`, an odds source with time-stamped historical
snapshots, upcoming fixtures and in-play prices. The key, markets and regions MUST live on the source and never on the
dataloader.

#### Scenario: Free statistics with paid odds
- **GIVEN** a loader built with `stats=FootballDataStats()` and `odds=OddsApi(key=..., markets=['h2h'])`
- **WHEN** the user prepares and extracts training data
- **THEN** the odds columns come from the bookmakers of the keyed source and the feature columns from the free source

### Requirement: Selected markets only
`OddsApi` SHALL request and return only the markets the user selected, and only those markets MUST appear as targets and
odds in the extracted tables.

#### Scenario: One market selected
- **WHEN** the user selects `markets=['h2h']` and extracts training data
- **THEN** only the match result market appears in `Y` and `O`

### Requirement: Price at the target moment
`OddsApi.to_snapshots` SHALL map each vendor snapshot to the long odds schema, with `event_status` and `event_time`
taken from the snapshot timestamp relative to kick-off, so an in-play target is priced with the odds available at that
moment and not the pre-match closing price.

#### Scenario: In-play target
- **WHEN** the user extracts training data with an in-play target at 45 minutes
- **THEN** `O` holds the prices available at kick-off plus 45 minutes

### Requirement: Items planned from the schedule
`OddsApi` SHALL declare that it needs a schedule and build its items from the kick-offs of the selected matches: one
historical snapshot item per requested moment for completed seasons, and the upcoming and in-play endpoint for the
current season. Matches that kick off together MUST share one snapshot, paid once. Only the live endpoint items MUST be
volatile.

#### Scenario: Shared kick-off
- **WHEN** two selected matches kick off at the same instant
- **THEN** the plan holds one historical snapshot item for that instant

### Requirement: Cost estimate
`OddsApi.estimate(items)` SHALL return credits from the vendor's published model: a historical request costs
`10 x markets x regions`, a live request costs `markets x regions`, and the sports list is free.

#### Scenario: Dry-run estimate
- **WHEN** the user runs `prepare(dry_run=True)` with `OddsApi`
- **THEN** `estimated_cost` holds the credits for `odds_api` and no credit is consumed

### Requirement: Historical coverage bound
Seasons before the vendor's historical coverage, which starts on 2020-06-06, SHALL be reported as unavailable before the
user pays for the obtainable part, and MUST NOT be silently truncated.

#### Scenario: Request before coverage
- **WHEN** the `param_grid` selects a 2018 season with `OddsApi` and the user runs a dry run
- **THEN** the report lists that season as unavailable

### Requirement: Key injected at request time
`OddsApi` SHALL add its key to the request URL through `request_url(item)` only when the store makes the request. The
key MUST NOT appear in a `RawItem`, the manifest, any file under the store, a log line or an error message.

#### Scenario: Key absent from errors
- **WHEN** a request with the key fails
- **THEN** the raised message does not contain the key

### Requirement: Rate limit and quota failures
When the key is invalid, rate-limited or out of quota, preparation SHALL fail with a message that names the source and
the reason, and MUST leave the store unchanged.

#### Scenario: Exhausted key
- **WHEN** preparation runs with an exhausted key
- **THEN** it raises an error naming `odds_api` and the quota reason, and the store holds what it held before

### Requirement: Reuse of paid raw data
When the user changes the requested markets after preparing, preparation SHALL reuse the raw payloads already paid for
wherever they cover the new request.

#### Scenario: Market change
- **WHEN** the user prepares with one set of markets and then with a set the held payloads already cover
- **THEN** no paid item is fetched again
