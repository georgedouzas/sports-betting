## Purpose

Define how a data source declares the raw items it needs and turns them into long snapshots, and how the preparation
step and the local store fetch, keep and refresh those items without surprise cost.

## ADDED Requirements

### Requirement: Sources declare items and transform payloads
A source SHALL subclass `BaseSource` and implement `index_items()`, `catalogue(payloads)`, `required_items(params)` and
`to_snapshots(payloads)`. `estimate(items)` SHALL return the units a fetch costs, the sum of the item costs by default.
The four methods MUST be pure and MUST NOT open a socket or a file. `required_items` MUST be deterministic, returning
the same items in the same order for the same parameters.

#### Scenario: Required items are deterministic
- **WHEN** a source's `required_items` is called twice with the same parameters
- **THEN** both calls return the same items in the same order

#### Scenario: Planning performs no input or output
- **WHEN** a test calls `required_items` and `to_snapshots` with the socket guard active
- **THEN** no connection and no file access is attempted

### Requirement: Statistics and odds source kinds
A source SHALL be either a statistics source, subclassing `BaseStatsSource`, or an odds source, subclassing
`BaseOddsSource`. `to_snapshots` of a statistics source MUST return the long `stats` table and of an odds source the
long `odds` table, both conforming to the existing snapshot schemas.

#### Scenario: Snapshot tables conform to the schemas
- **WHEN** a statistics source and an odds source transform their payloads
- **THEN** the two tables validate against the statistics and odds snapshot schemas

### Requirement: Raw items share identity across sources
A `RawItem` SHALL carry `source`, `key`, `url`, `params`, `volatile` and `cost`, and its identity MUST be
`(source, key)`. Two sources that declare the same identity MUST receive the same payload from a single fetch.

#### Scenario: A shared upstream file is fetched once
- **WHEN** two sources declare items with the same `(source, key)` and preparation runs
- **THEN** the store fetches that item once and hands the same payload to both sources

### Requirement: Discovery on the configured source
A user SHALL ask a source what it publishes with the instance method `available_params(store=None)`, with no dataloader
and no `param_grid`. The answer MUST depend on the configured instance, so a keyed source answers for its own
subscription. With no store given, the source MUST use a default `LocalStore`.

#### Scenario: Discover before building a dataloader
- **WHEN** the user calls `FootballDataStats().available_params()`
- **THEN** it returns the league, division and year combinations the source publishes

### Requirement: Explicit preparation step
A source-backed dataloader SHALL expose `prepare(dry_run=False, refresh=False)`, which populates the store for its
`param_grid` and returns a `PreparationReport`. Preparation MUST report progress for long-running fetches.

#### Scenario: Prepare then extract
- **GIVEN** a `SoccerDataLoader` for England, division 1, 2024, with an empty store
- **WHEN** the user calls `prepare()` and then `extract_train_data()`
- **THEN** preparation fetches the needed files and extraction returns `X`, `Y` and `O`

### Requirement: Incremental and resumable preparation
Preparation SHALL fetch only the items the store does not hold. A preparation run that was interrupted MUST resume on
the next run without fetching or paying again for completed items. Re-running preparation on an unchanged, fully
prepared store MUST fetch nothing except volatile items.

#### Scenario: Re-preparing a complete store
- **WHEN** the user calls `prepare()` on a store that already holds every completed season
- **THEN** no completed season is fetched again

#### Scenario: Resume after an interruption
- **WHEN** a preparation run stops midway and the user runs it again
- **THEN** only the items not yet held are fetched

### Requirement: Dry run spends nothing
`prepare(dry_run=True)` SHALL return a `PreparationReport` with `to_fetch`, `held`, `estimated_cost` per source, and
`unavailable`, and MUST NOT fetch any priced item or spend anything. To make the estimate exact it SHALL read only free
data, namely a source's catalogue and the statistics that give the match schedule.

#### Scenario: Estimate for a metered source
- **WHEN** the user calls `prepare(dry_run=True)` on a loader with an `OddsApi` odds source
- **THEN** the report lists the items to fetch, the items held and an estimated credit cost, and no credit is consumed

### Requirement: Unavailable selections are reported at plan time
Preparation SHALL report the `param_grid` combinations that the sources do not publish in
`PreparationReport.unavailable`, including seasons before a metered source's historical coverage, before anything is
paid, and MUST NOT surface them as fetch-time network errors.

#### Scenario: A combination the source does not publish
- **WHEN** the `param_grid` selects a league-season the source does not publish and the user runs a dry run
- **THEN** the report lists that combination as unavailable and no request is made for it

### Requirement: Local store location
The library SHALL persist fetched data in a `LocalStore(path=None)` at a user-configurable path, with a default location
when no path is given. The store MUST hold raw payloads, derived snapshot tables and a manifest of what is held, from
which source, and whether each item is volatile.

#### Scenario: Custom store path
- **WHEN** the user builds `LocalStore(path)` and prepares a loader with it
- **THEN** the raw payloads, snapshot tables and manifest are written under that path

### Requirement: Raw payloads are retained
The store SHALL keep every raw payload exactly as fetched and MUST never delete or rewrite one. Re-fetching a volatile
item MUST write a new payload and supersede the manifest entry while the old payload is kept.

#### Scenario: Refreshing a volatile item
- **WHEN** preparation re-fetches a volatile fixtures file
- **THEN** the new payload is stored and the earlier payload remains on disk

### Requirement: Derived tables rebuild for free
Derived snapshot tables SHALL be rebuildable from the retained raw payloads with no fetching and no cost. Derived data
MUST be invalidated when the library version that derives it changes, so an upgrade never serves the output of an
earlier transform.

#### Scenario: Transform change after preparation
- **GIVEN** a fully prepared store
- **WHEN** the transform changes and the user prepares again
- **THEN** the derived tables are rebuilt from raw payloads, nothing is fetched and nothing is spent

### Requirement: Volatile and immutable items
The store SHALL treat an item as held when the manifest records it and it is not volatile. Volatile items, such as
fixtures, the current season and upcoming odds, MUST be fetched again on each preparation. `prepare(refresh=True)` MUST
fetch again every item the store holds, including finished seasons, and only on request.

#### Scenario: Finished seasons stay held
- **WHEN** the user prepares a store holding a finished season and the current season
- **THEN** only the current season and the fixtures are fetched again

#### Scenario: Explicit refresh
- **WHEN** the user calls `prepare(refresh=True)`
- **THEN** the finished seasons are fetched again as well

### Requirement: Atomic and concurrent-safe writes
Every store write SHALL go to a temporary file in the destination directory and then be renamed into place, so a partial
write is never read back as complete. Two processes preparing the same store MUST NOT corrupt it, and a manifest with a
torn last line MUST be read up to its last complete line.

#### Scenario: Interrupted write
- **WHEN** a write is interrupted before the rename
- **THEN** the item is not visible under its final name and the store stays readable and resumable

### Requirement: Column types survive the store
Reading a derived table back from the store SHALL return the same columns and dtypes that were written. An empty result
MUST NOT degrade the dtypes of the data it is combined with. An upstream file that exists but is empty MUST be recorded
as fetched and empty, and MUST NOT be fetched forever or poison the extracted column types.

#### Scenario: Dtype round trip
- **WHEN** a snapshot table is written to the store and read back
- **THEN** the columns and dtypes equal the written ones

#### Scenario: Empty season file
- **WHEN** an upstream season file exists but holds no rows
- **THEN** it is recorded as fetched and the extracted `division` and `year` keep their types

### Requirement: Credentials stay out of items and the store
A credential SHALL be added to a request only when the request is made, through the source's `request_url(item)` or
headers. It MUST NOT be part of a `RawItem`, MUST NOT be written to the raw payloads, the snapshot tables or the
manifest, MUST NOT appear in logs or error messages, and MUST be sent only to the source it authenticates.

#### Scenario: No key on disk
- **WHEN** the user prepares a store with a keyed source and the store is scanned for the key
- **THEN** the key appears in no file under the store path

### Requirement: Kick-off instant in UTC
Every source SHALL emit `date` as the tz-aware UTC kick-off instant of the match, converting its own local
representation at its boundary, so `date + event_time` is the wall-clock instant of a snapshot. Where a source has no
kick-off time, `date` MUST be midnight UTC of the match day.

#### Scenario: Snapshot instant
- **WHEN** a source transforms a match that kicked off at 15:00 UK time in winter
- **THEN** `date` is 15:00 UTC on that day and the in-play snapshot at 45 minutes addresses 15:45 UTC

### Requirement: Schedule for instant-addressed odds sources
An odds source whose prices are addressed by instant SHALL declare `needs_schedule()` and receive the schedule of the
selected matches through `required_items(params, schedule)`. The dataloader MUST read the statistics first and derive
the schedule from their kick-offs before it plans the odds.

#### Scenario: Odds planned from the statistics
- **WHEN** a loader combines `FootballDataStats` with `OddsApi` and plans preparation
- **THEN** the odds items are derived from the kick-offs in the statistics

### Requirement: No redistribution of third-party odds
The project SHALL NOT publish, mirror or distribute third-party odds data in any form, including repository content,
release artifacts or documentation fixtures beyond the minimum needed for tests. The redistributing `data` branch MUST
be retired. The equivalence reference MUST be a fingerprint of hashes, not the data.

#### Scenario: Published artifacts hold no odds
- **WHEN** anyone inspects the repository branches and published artifacts
- **THEN** no `data` branch exists and the only odds in the tree are hand-authored rows for the offline dummy loader
