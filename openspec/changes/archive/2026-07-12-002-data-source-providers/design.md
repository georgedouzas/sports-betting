# Design

## Context

`SoccerDataLoader` fetched pre-built CSVs that a Prefect flow on the `data` branch produced from football-data.co.uk,
odds included. That redistribution had no licence. The only odds available were pre-match closing prices, so in-play
targets from `001-in-play-betting` could not be priced at the moment they occurred.

This change replaces the mirror with injectable sources. `SoccerDataLoader` keeps its shape and gains `stats`, `odds`
and `store`. The football-data.co.uk transform moves into the library and runs on the user's machine. A keyed `OddsApi`
source adds time-stamped prices. Fetching happens only in an explicit `prepare()` step. Everything lives behind the
existing `BaseDataLoader._snapshots()` seam, which does not change, and the snapshot schemas and column grammar from
`001-in-play-betting` stay as they were.

Technical context: Python `>=3.11, <3.14`. pandas, numpy, scikit-learn, pandera, aiohttp for fetching, beautifulsoup4
for index discovery, click and rich for the CLI, and the new `pyarrow` for Parquet. Storage is the local filesystem:
zstd Parquet for derived tables, gzipped raw payloads and a JSONL manifest, with atomic writes. Tests use pytest with
`--doctest-modules`, branch coverage, pytest-randomly and pytest-xdist, and no test may touch the network. The target
scale is about 900 league-seasons and 10^5 matches, gigabytes at the extreme once five-minute in-play snapshots multiply
the rows.

## Goals / Non-Goals

**Goals:**

- Stop distributing third-party odds in any form.
- Reproduce today's data exactly on the user's machine from the original public source.
- Let free statistics combine with a user's own paid odds, with the key never leaving the user's machine.
- Make every fetch explicit, incremental, resumable and estimable before any money is spent.
- Refuse a join across sources that leaves odds missing, rather than return a quietly smaller dataset.
- Keep the dummy dataloader, the factory functions and the documented examples working.

**Non-Goals:**

- Basketball and other sports. The design must not preclude them.
- Statistics sources beyond the relocated free one.
- A hosted or shared store, since a shared store of vendor odds would recreate the redistribution problem.
- Live in-play streaming and automated live execution.
- An embedded analytical engine, an embedded SQL database or a remote database.

## Decisions

### D1. Sources declare work and do not fetch

Decision: `BaseSource` exposes `required_items(params)` returning `RawItem`s and `to_snapshots(payloads)` returning a
long table. Fetching and persistence belong to the store, orchestrated by `prepare()`.

Rationale: `FootballDataStats` and `FootballDataOdds` read the same upstream CSV, so declared items let the plan be a
set union in which each item is fetched once. A dry run becomes a network-free set difference of required and held
items. A source that cannot fetch cannot make extraction fetch by accident.

Alternatives considered: a source that fetches and caches internally, rejected because each source then needs its own
cache, dedup and dry run, and avoiding double downloads becomes a cross-source handshake.

### D2. A raw item is the unit of fetching, caching and cost

Decision: a `RawItem` is an immutable descriptor with `source`, `key`, `url`, `params`, `volatile` and `cost`. Identity
is `(source, key)`, and the manifest is keyed by it.

Rationale: one object serves as the dedup key, the cache key, the resume unit and the cost unit.

### D3. Discover upstream files from the league index pages

Decision: fetch each selected league's index page and extract the `.csv` links, only for the leagues in the
`param_grid`, and cache the discovered manifest in the store as a volatile item.

Rationale: an earlier hardcoded product of leagues, divisions and years fabricated combinations the source does not
publish, such as `Netherlands_2`, which returned 404. The upstream is the only authority on what exists.
`beautifulsoup4` was already a declared dependency.

Alternatives considered: build URLs deterministically from a pattern, rejected because a wrong guess becomes a
fetch-time 404 instead of a plan-time answer. Ship a static manifest, rejected because it goes stale at the start of
each season.

### D4. Port the transform, do not rewrite it

Decision: move `_preprocess_data`, `_convert_data_types`, `_extract_features` and `_to_snapshots` from the `data`
branch's `soccer.py` into `_sources/_football_data.py` substantially as they were, and call the public
`market_outcomes` instead of the old private copy.

Rationale: the equivalence gate demands exact equality, and three behaviours change the output if missed.
`_convert_data_types` fills missing integers with `-1`, and `_to_snapshots` reads that sentinel to decide whether a
match was played, which is how fixtures get only a pre-play snapshot. `_extract_features` computes expanding and rolling
means within a season frame that already has the fixtures appended, then shifts by one. `_preprocess_data` back-fills
each odds column from its `_closing` twin when the plain column is absent or empty, then drops the closing columns.

### D5. Parquet, gzipped raw payloads and a JSONL manifest

Decision: derived tables are zstd Parquet partitioned by league, division and year through `pyarrow`. Raw payloads are
gzipped, one file per item. The manifest is append-only JSONL. Writes go to a temporary file in the destination
directory and are moved into place with `os.replace`.

Rationale: Parquet round-trips dtypes. CSV does not, and an empty `fixtures.csv` read back as all-object once poisoned
`division` and `year` and broke schema validation. Raw is kept forever because credits cannot be re-spent. A rename is
atomic on POSIX and on Windows. A JSONL manifest survives a partial write by truncating to its last complete line.

Alternatives considered: DuckDB and SQLite, rejected as an engine and a query language for an access pattern that is a
slice read and a set difference over about 10^4 keys. A remote database, rejected because it recreates the
redistribution problem. Pickle, rejected as an unsafe and non-durable archive.

### D6. Preparation is required on every path

Decision: extraction on an unprepared store raises `NotPreparedError` whatever the source, free or metered.

Rationale: a safety property with an exception is not a safety property. If extraction fetched when free, a user who
swaps in a metered source would inherit a path that used to fetch silently. Uniformity gives the CLI, the GUI and the
docs one two-step shape. The cost is a breaking change to the user flow, accepted deliberately.

### D7. The equivalence gate is a fingerprint, not the data

Decision: commit per-column hashes, dtypes, row counts and the ordered column list for `stats`, `odds`, `X`, `Y` and
`O` at a fixed `param_grid` (England, division 1, 2024). Do not commit the frames.

Rationale: committing the frames would redistribute odds. A column hash is an exact check that is not data and names
the drifted column. The reference had to be captured before the `data` branch was purged, and a local uncommitted copy
was kept for row-level diffs.

### D8. The cost estimate is arithmetic over the plan

Decision: the estimate comes from the declared items and the source's published multiplier, not from a measurement.
`PreparationReport` carries it and states that it is an estimate.

Rationale: a dry run cannot call the API to learn the charge. The multiplier and coverage start were pinned later
against the vendor's documentation (D14).

### D9. Resolve match identity only when the sources differ

Decision: resolve `(competition, kickoff, home_team, away_team)` by mapping each source's league key to the canonical
league and division, normalizing team names, and matching kickoff within a tolerance window. Report matched and
unmatched counts and fail above `max_unmatched_rate`.

Rationale: an unmatched row becomes a missing odd, a dropped bet and a clean but wrong backtest. A hard gate turns the
quietest failure into the loudest. The free path needs no resolution, because both free sources derive from the same
upstream row.

### D10. No network in tests

Decision: a session socket guard in `tests/conftest.py` fails any test that opens a connection. Sources are tested on
recorded or synthetic payloads.

Rationale: the only way to know extraction never fetches is to make fetching impossible in tests and see the suite pass.
It also removes the upstream site as a source of CI flakiness. Network-touching classes carry no executable doctests.

### D11. Discovery belongs to the source

Decision: `BaseSource.available_params()` is the public discovery API. The dataloader has no public discovery method and
keeps a private filter that takes the intersection of what the statistics source and the odds source publish.

Rationale: a `param_grid` cannot be written before it is known what exists, so discovery cannot live on an object built
with a `param_grid`. Availability depends on a source's configuration, such as a key's subscription tier, so it is per
instance. The earlier private filter asked only the statistics source and would have offered seasons back to 1994 next
to an odds vendor whose history starts in 2020.

Alternatives considered: keep `get_all_params()` as a classmethod, rejected because it cannot see an injected source. A
classmethod on the source, rejected because a source's own configuration decides availability. A descriptor that works
on the class and the instance, rejected as machinery for a call site that should not exist.

The invariant "a source never fetches" was over-applied. What must hold is that extraction never fetches and planning
is free. A source resolving its own free catalogue through a store breaks neither.

### D12. Derived data is keyed by the code that derived it, and nothing upstream is immutable

Decision: the derived-snapshot cache key includes the library version. `prepare(refresh=True)` re-reads everything the
store holds, including finished seasons.

Rationale: keyed on payloads alone, a release that fixed the transform would serve the old output. Including the version
over-invalidates on every release at no cost, since rebuilding from raw is free. football-data.co.uk does correct
historical files, so "finished" means "not re-read unless asked". A refresh is explicit because a metered source charges
for a re-read, and a schedule or TTL would spend money unasked.

Alternatives considered: conditional requests with `If-Modified-Since` or `ETag`, judged the better long-term answer for
free sources but placed in the fetch layer and not needed to close the correctness hole.

### D13. `date` is the kick-off instant in UTC

Decision: `date` carries the tz-aware UTC kick-off instant in both tables and from every source, so
`date + event_time` is the wall-clock instant of a snapshot. Each source converts its own zone at its boundary.

Rationale: The Odds API addresses historical prices by wall-clock timestamp, so "the price at minute 45" is a request
for the snapshot at kick-off plus 45 minutes. The feed's `Time` column had been dropped. The zone was found
empirically: Spanish kick-offs land one hour before the standard La Liga CEST slots, so the feed publishes every league
in UK time. The feed carries `Time` from 2019/20 and The Odds API history starts on 2020-06-06, so the seasons with
obtainable in-play odds are the seasons with kick-off times. Earlier seasons keep a midnight UTC `date`. Verified on the
same 380 matches with identical per-match features.

### D14. The odds plan is built from the statistics

Decision: `required_items(params, schedule)`. A source declares `needs_schedule()`. The dataloader reads the statistics
first, derives the kick-off schedule, and then plans the odds. A dry run may read free data to build the plan and never
a priced item.

Rationale: an odds source addressed by timestamp cannot plan from a season alone. Self-discovery through the vendor's
events endpoint costs about 40 credits a season, while the free statistics give the schedule for nothing. Measured on a
real season: 432 snapshots and 8,642 credits, quoted for zero credits. FR-012 was amended from "no fetch" to "spends
nothing".

Credential handling: the vendor takes its key as a query parameter, so a source exposes `request_url(item)`, which the
store calls at the moment of the request. The key never reaches a `RawItem`, the manifest or the disk, and a test scans
the store for it.

Cost model, pinned against the vendor's documentation: historical odds cost `10 x markets x regions`, live odds cost
`markets x regions`, and `/v4/sports` is free. Matches that kick off together share a snapshot and are paid once.

### D15. Pair the rosters of a league-season

Decision: reconcile team names within a `(league, division, year)` group, where both sources hold the same roster of
about twenty clubs. Exact-match what agrees, then pair what is left by token-prefix similarity, accepting only a clear
winner above a floor and with a margin over the runner-up. Ship no alias table and keep `aliases` as the escape hatch.
Default `max_unmatched_rate=0.0` and raise `UnmatchedError` past it.

Rationale: the first cut compared each vendor name against every club in the library, which forced users to hand-build
a mapping for about 27 leagues and shipped 14 unverifiable aliases. Within a league-season, confusable names are present
on both sides and claim each other first. String similarity inverts the right answer: `SequenceMatcher` rates
`Everton` and `Liverpool` at 0.50 and `Wolves` and `Wolverhampton Wanderers` at 0.41, while the token-prefix score
gives 0.00 and 0.83. The last unpaired name is not paired for being last, and unrelated leftovers are never paired.
Verified: the 20 clubs of the 2024/25 English top flight pair 20 of 20 with no aliases.

### D16. Complementary markets are derived from the data

Decision: `complementary_events(markets)` derives the mutually exclusive and exhaustive groups from the markets the data
carries. `BaseBettor.COMPLEMENTARY_EVENTS` becomes an override that defaults to `None`.

Rationale: the hardcoded soccer list was a live bug. `over_1.5` and `under_1.5` probabilities were left un-normalized
and summed to 0.60, and `bet()` raised `ValueError: No objects to concatenate` for them. `over_X` and `under_X` pair at
any `X`. The outcome group is whichever of home win, draw and away win the data holds, so a sport with no draw gets a
two-way outcome. `IDENTITY_COLS` was checked and needed no change for basketball.

### D17. Live smoke test: the sport keys are right, and a league key is not a league

Finding: all 38 mapped sport keys exist among the 170 that `/v4/sports` returns. `soccer_epl` returned League Cup
opponents such as `Coventry City`, so the vendor's event list is not the league roster.

Consequence: an extra club has no counterpart, so the floor and margin never pair it. Its odds rows are dropped and the
gate does not fire, because the gate counts our games without odds, not their odds without a game. The report had told
the user to add an alias for such a club, and that advice was fixed: a name is reported only when a club is left for it
to be. Real names on both sides paired 16 of 16 with no aliases, and the vendor writes `Bournemouth`, not
`AFC Bournemouth`.

### Data model

- `RawItem`: `source`, `key` (such as `England_1_2024`, `fixtures` or `soccer_epl@2024-08-17T14:00:00Z`), `url`,
  `params` (never credentials), `volatile` and `cost`.
- `RawPayload`: `item`, `content` as the exact bytes returned, and `fetched_at`.
- Manifest entry: `source`, `key`, `volatile`, `fetched_at`, `digest` and `bytes`.
- `PreparationReport`: `to_fetch`, `held`, `estimated_cost` per source, and `unavailable` combinations.
- `NotPreparedError`: carries the `PreparationReport`.
- `ReconciliationReport`: `matched`, `unmatched_stats`, `unmatched_odds` and `unmatched_rate`.
- Match identity: `league`, `division`, `year`, `date` compared within a window, `home_team` and `away_team`.
- `LocalStore(path=None)` layout: `raw/<source>/<key>.gz`, `snapshots/<source>/<kind>/` and `manifest.jsonl`.
- `SoccerDataLoader(param_grid=None, stats=None, odds=None, store=None, aliases=None, max_unmatched_rate=0.0)`, with the
  football-data sources and a `LocalStore` as defaults. Constructor parameters are stored unmodified and unvalidated.
  `reconciliation_` is set when the sources differ.

## Risks / Trade-offs

- A new runtime dependency, `pyarrow`, is accepted because the store must round-trip dtypes and Parquet is five to ten
  times smaller than CSV. CSV was the exact bug class to close.
- The golden fixture is a fingerprint, so a failure names the drifted column but row diffs need a local capture.
- Throwaway `_DataBranchStats` and `_DataBranchOdds` sources landed the seam with zero behaviour change under the full
  suite and were deleted in Phase 4.
- Requiring `prepare()` everywhere is a breaking change to every example that builds a `SoccerDataLoader`.
- A vendor event list can hold clubs from other competitions. The roster pairing guards drop their odds without firing
  the gate.
- Historical odds are a paid tier, so the historical request shape was not tested against the live endpoint.

## Migration Plan

Each phase was independently shippable, with the suite green at each boundary.

1. Phase 0: freeze the equivalence gate. Capture the fingerprint from the mirror before it is purged.
2. Phase 1: introduce the source and store seams with the `_DataBranch*` defaults and change no behaviour. Every
   existing test passes unmodified.
3. Phase 2: relocate the transform client side. The football-data sources reproduce the fingerprint.
4. Phase 3: add the store and `prepare()`. Extraction never fetches.
5. Phase 4: default to the football-data sources and `LocalStore`, delete the scaffolding, update the docs, retire the
   `data` branch and ship as a breaking release. Then move discovery to the source.
6. Phase 5: add `OddsApi` with its cost model and credential handling.
7. Phase 6: add the resolver and the quality gate.

Users add a `prepare()` call before extraction and replace `get_all_params()` with the source's `available_params()`.

## Open Questions

- Basketball has no free source with statistics and odds. Whether it ships with no free path is a product decision left
  to the maintainer (D16).
- The historical request shape of The Odds API remains untested against the live endpoint, since the smoke-test key did
  not cover the paid tier (D17).
