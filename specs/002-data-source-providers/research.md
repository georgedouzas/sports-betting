# Research: Pluggable Data-Source Providers

Decisions taken before design. Several were settled with the maintainer in advance and are recorded here rather than re-argued; the rest come from reading the `data` branch ETL (`soccer.py`) and the current loader.

## D1. Sources declare work; they do not fetch

**Decision**: `BaseSource` exposes `required_items(params) -> list[RawItem]` and `to_snapshots(payloads) -> DataFrame`. Fetching and persistence belong to the store, orchestrated by `prepare()`. A source never opens a socket.

**Rationale**: Three requirements collapse into this one inversion.

1. `FootballDataStats` and `FootballDataOdds` read *the same upstream CSV*. If each source fetched for itself, every league-season would be downloaded twice. With declared items, the plan is a set union over sources and each item is fetched once.
2. `prepare(dry_run=True)` must cost nothing (FR-012). If planning is `set(required) - set(held)` over declared keys, it is trivially network-free. If sources fetched, "what would this cost" could only be answered by doing it.
3. Extraction must never fetch (FR-013). A source that cannot fetch cannot violate this by accident.

**Alternatives rejected**: A source that fetches and caches internally — each source then needs its own cache, its own dedup, its own dry-run, and the "don't download twice" rule becomes a cross-source handshake rather than a property of the design.

## D2. `RawItem` is the unit of fetching, caching, and cost

**Decision**: A `RawItem` is an immutable descriptor: `source` (name), `key` (stable identity), `url`/`request`, and `volatile` (bool). Two sources declaring the same `key` declare the same item. The store's manifest is keyed by `(source, key)`.

**Rationale**: It is simultaneously the dedup key (D1), the cache key, the resume unit (FR-011) and the cost unit (one item = one request = *n* credits). Making these the same object is what keeps `prepare()` simple.

## D3. Upstream URL discovery: scrape the league index pages

**Decision**: Keep the `data` branch's approach — fetch each selected league's index page and extract the `.csv` hrefs — but fetch only the index pages for the leagues in the `param_grid`, and cache the discovered manifest in the store as a volatile item.

**Rationale**: This is exactly the bug we already fixed once. The loader used to hold a hardcoded cartesian product of leagues × divisions × years, which fabricated combinations the source does not publish (`Netherlands_2` → 404). The upstream is the only authority on what exists. `beautifulsoup4` is *already* a declared runtime dependency (currently unused, left over from an earlier version), so this costs no new dependency.

**Alternatives rejected**:

- *Construct URLs deterministically* (`mmz4281/{yy}{yy}/{DIV}.csv`). Tempting and cheaper, but it reintroduces exactly the fabrication bug: nothing tells us which seasons and divisions actually exist, so a wrong guess is a 404 at fetch time rather than an honest "not available" at plan time. It also cannot enumerate `get_all_params()`.
- *Ship a static manifest in the package.* Goes stale the moment a new season starts, and the current season is the one users care about most.

## D4. Port the ETL; do not rewrite it

**Decision**: Move `_preprocess_data`, `_convert_data_types`, `_extract_features` and `_to_snapshots` from `soccer.py` into `_sources/_football_data.py` substantially as they are.

**Rationale**: Phase 0's fingerprint is an exact-equality gate, and this code has behaviour that a clean-room rewrite would not reproduce:

- `_convert_data_types` fills missing integer columns with **-1**, not NaN. `_to_snapshots` then relies on that sentinel (`home_goals.ge(0)`) to decide whether a match has been played — which is how upcoming fixtures end up with only a pre-play snapshot.
- `_extract_features` computes per-team expanding and rolling means **within a season frame that already has the fixtures rows concatenated onto it**, then shifts by one. Compute the features before appending fixtures and the fixture rows get no form features; compute them across seasons and every value changes.
- `_preprocess_data` back-fills each non-closing odds column from its `_closing` twin when the non-closing one is absent or all-NaN, then drops the closing columns. Skip that and half the historical odds vanish.

These are not incidental. They are the behaviour the fingerprint pins.

**Note for implementation**: `market_outcomes` has *already* been relocated (it lives in `_soccer/_utils.py` and is public). The port should call it rather than re-introduce the `data` branch's `_market_outcomes`.

## D5. Storage: Parquet + gzipped raw + a JSONL manifest

**Decision**: Derived snapshot tables as Parquet (zstd, partitioned by source/sport/league/season) via `pyarrow`. Raw payloads gzipped, one file per `RawItem`. A JSONL manifest as the index. Atomic writes: write to a temporary file in the same directory, then rename.

**Rationale**:

- Parquet round-trips dtypes. CSV does not, and that has already cost us: an empty `fixtures.csv` came back all-object and poisoned `division`/`year` through a concat, breaking schema validation. FR-020 exists because of that bug.
- Raw is retained forever (FR-016) because credits are money and cannot be re-spent. Derived tables are rebuildable from raw for free (FR-017), so changing the feature engineering never re-charges the user.
- Rename is atomic on POSIX and on Windows (`os.replace`), which is what makes FR-019 (never read a partial write as complete) achievable without a lock protocol.
- JSONL for the manifest, not Parquet: it is append-only, human-readable, and survives a partial write by truncating to the last complete line.

**Alternatives rejected**: DuckDB and SQLite — an engine and a query language for an access pattern that is "read a `param_grid` slice into a dataframe" plus "set-difference over ~10^4 keys". Neither needs a planner. A remote database — recreates the redistribution problem this feature exists to remove, and adds a server to a laptop-scale workload.

## D6. `prepare()` is required on *every* path, including the free one

**Decision**: `prepare()` is not conditional on the source being metered. `extract_train_data()` on an unprepared store raises `NotPreparedError` regardless of who the source is.

**Rationale**: FR-013 is a safety property, and a safety property with an exception is not one. If extraction fetches "only when it's free", then a user who swaps in a metered odds source inherits a code path that used to fetch silently. Uniformity also means the CLI, the GUI, the docs and the gallery all show the same two-step shape.

**Cost**: this is a breaking change to the user-facing flow, and every example that constructs a `SoccerDataLoader` must gain a `prepare()` call. Accepted deliberately.

## D7. The equivalence gate is a fingerprint, not the data

**Decision**: Commit per-column hashes, shapes, dtypes and the ordered column list for `stats`, `odds`, `X`, `Y` and `O` at a fixed `param_grid`. Do not commit the frames.

**Rationale**: FR-025 forbids redistributing odds; the gate has to run in CI. A column-level hash is an exact equality check that is not itself data, and when it fails it names the column that drifted. The developer regenerates the full frames locally to diff rows.

**Consequence**: the reference must be captured *before* the `data` branch is purged in Phase 4, and a local (uncommitted) copy of the frames should be kept for the duration of the migration for debugging.

## D8. Cost model for the metered source

**Decision**: The estimate is computed from the declared item count and the source's published per-request multiplier, not measured. `OddsApi.estimate(items)` returns credits; `PreparationReport` carries it. The report states plainly that it is an estimate.

**Rationale**: A dry run cannot call the API to find out what the API would charge. The published model (one request per market-region combination; historical endpoints cost a multiple of live ones) is arithmetic over the plan, which is exactly what the plan is for.

**Open**: the exact multiplier and the historical coverage start date are vendor facts to be pinned during Phase 5 against the live documentation, not guessed now.

## D9. Entity resolution: normalise, then join on a windowed kickoff

**Decision**: Resolve `(competition, kickoff, home_team, away_team)` by mapping each source's league key to the canonical `league`/`division`, normalising team names through a per-source alias table, and matching kickoff within a tolerance window. Report matched/unmatched counts; fail above `max_unmatched_rate`.

**Rationale**: The failure mode is silent, not loud: an unmatched row becomes a missing odd, a missing odd becomes a dropped bet, and the backtest comes out clean and wrong. A hard gate converts the worst kind of failure into the most obvious kind. The alias table is a maintained artifact, not a fuzzy matcher, because a fuzzy match that is 97% right is a backtest that is 3% fabricated.

**Not needed for the free path**: `FootballDataStats` and `FootballDataOdds` derive from the same upstream row, so their identities are equal by construction. The resolver only engages when the two sources differ.

## D10. No network in tests

**Decision**: A socket guard in `conftest.py` fails the suite if any test attempts a connection. Sources are exercised against recorded payloads under `tests/samples/`; the fetch layer against a local server fixture.

**Rationale**: SC-004 says extraction can never fetch. The only way to *know* that is to make fetching impossible during tests and see the suite stay green. It also removes the upstream site as a CI flake source — the existing `test_soccer.py` hits the live mirror today.

**Constraint this imposes**: `--doctest-modules` runs every `>>>` in `src/`. Network-touching classes must not carry executable doctests. `SoccerDataLoader` already has none; `FootballData*` and `OddsApi` must follow suit and document by prose plus a non-executed fenced block.

## D11. Discovery belongs to the source, and is not on the dataloader at all

**Decision**: `BaseSource.available_params()` is the public discovery API. The dataloader has **no** public discovery
method; it keeps a private `_all_params()` used only to filter `param_grid`, and that filter is the **intersection** of
what the statistics source and the odds source publish.

**Rationale**: To write a `param_grid` you must first know what exists. So discovery cannot live on an object that is
constructed *with* a `param_grid` — `SoccerDataLoader().get_all_params()` means building a throwaway dataloader to ask a
question that has nothing to do with loading data. That is a chicken-and-egg, and it is the whole reason discovery was
awkward to place.

Availability is determined by the source and its configuration: `OddsApi(key=...)` can only offer what the key's tier
covers. So it is per-instance on the source, not a class-level fact and not a dataloader fact.

The dataloader needs the available parameters for exactly one thing — refusing to request a combination the source does
not publish. That is internal. Making it public was a leftover from when the data source was hardcoded into the class,
which is the only world in which "what is available" was a property of the dataloader.

**Corrects an earlier mistake**: the invariant "a source never fetches" was over-applied. What must hold is that
*extraction* never fetches and that *planning* is free — enforced by `required_items()` and `to_snapshots()` being pure
and by extraction reading only from the store. The catalogue is free by construction and `prepare(dry_run=True)` already
resolves it, so a source resolving its own catalogue breaks nothing.

**Fixes a latent bug**: `_all_params()` asked only the statistics source. With `FootballDataStats` + `OddsApi` that would
offer seasons back to 1994, which no odds vendor has, and the missing odds would surface as a silently smaller dataset —
exactly the failure mode FR-023 exists to prevent. The intersection closes it.

**Alternatives rejected**:

- *Keep `get_all_params()` as a classmethod.* It cannot see an injected source, so it would silently answer for the
  default one. A wrong answer that looks right.
- *A classmethod on the source.* Same problem one layer down (a source's own config decides availability), and it would
  have to fetch without a store.
- *A descriptor that works on both the class and the instance.* Machinery to preserve a call site that should not exist.

## D12. Nothing upstream is immutable, and derived data is keyed by the code that derived it

**Decision**: The derived-snapshot cache key includes the library version, not just the raw content. And `prepare(refresh=True)`
re-reads everything the store holds, including seasons it considers finished.

**Rationale**: Two silent-staleness bugs, both found by exercising the store rather than by reading it.

1. *The derived cache was keyed only on the payloads.* Snapshots are produced by code as well as from data, so a release that
   fixed a bug in `_extract_features` would leave the digest unchanged and hand the user the **old** transform's output. Silent
   wrong data on upgrade — the exact failure class this design exists to rule out. Including the version over-invalidates on
   every release, which costs nothing: rebuilding from raw is free by construction (FR-017).
2. *A finished season was never re-read.* `volatile=False` was treated as "cannot change", but football-data.co.uk does correct
   historical files. The honest meaning is "not expected to change, so do not re-read it unless asked". `refresh=True` is that
   asking.

**Why `refresh` is explicit and not a schedule or a TTL**: on a metered source a re-read is charged for again. A background
freshness policy would spend the user's money without being asked, which FR-012 and FR-013 exist to prevent. A TTL would also
have to be tuned, and the whole store design deliberately has nothing to tune.

**Alternative considered**: conditional requests (`If-Modified-Since` / `ETag`) would let the store detect an upstream
correction cheaply, without a full download and without spending credits. That is the better long-term answer for free sources
and is worth doing, but it belongs in the fetch layer and is not needed to close the correctness hole.

## D13. `date` is the kick-off instant in UTC, and `date + event_time` addresses a snapshot

**Decision**: `date` carries the kick-off **instant**, tz-aware UTC, for both tables and from every source. A source resolves its
own time zone at its own boundary. The invariant this buys is `date + event_time = the wall-clock instant of the snapshot`.

**Rationale**: `event_time` is elapsed-from-kick-off, but kick-off itself was never stored — `_preprocess_data` dropped the feed's
`Time` column, so `date` was a calendar day. That is fine for pre-match odds, which do not care. It is fatal for in-play odds: The
Odds API addresses historical prices by **wall-clock timestamp**, so "the price at minute 45 of Arsenal–Chelsea" is a request for
the snapshot at `kick-off + 45min`. With a day-level date there is no such request to make. Kick-off time is therefore not a
nicety for one source; it is load-bearing for the whole in-play premise.

**The time zone was determined empirically, not assumed.** The feed's own notes define `Time` only as "Time of match kick off",
with no zone. English kick-offs land on the classic UK slots (15:00, 12:30, 17:30). Spanish ones land on 13:00 / 15:15 / 17:30 /
20:00 — exactly the standard La Liga CEST slots (14:00 / 16:15 / 18:30 / 21:00) shifted by −1 hour. So the feed publishes **every**
league in **UK time**, whatever country the match is played in. Reading it as UTC, or as the match's local time, would have shifted
every in-play join by an hour or more, silently.

**Coverage lines up**: the feed carries `Time` from 2019/20 onward; The Odds API's historical odds begin 2020-06-06. The seasons
where in-play odds are obtainable are exactly the seasons that have kick-off times. Earlier seasons keep a midnight `date`, and the
intersection rule of D11 already stops them being selected alongside a time-stamped odds source.

**What it collapses**: it removes the question of how an odds source learns *when* the matches are. With kick-off times in the
statistics, the dataloader can hand the odds source the schedule. Since the statistics source is free, a dry run can produce an
**exact** credit estimate for a metered odds source while spending **nothing** — which is strictly better than having the odds
source discover the schedule itself at ~40 credits a season.

**Verified**: the same 380 matches, with per-match feature values identical. Only `date` gains its time, and same-day matches now
sort by kick-off rather than arbitrarily.

## D14. The odds source is planned from the statistics, so a metered plan is exact and free

**Decision**: `required_items(params, schedule)`. A source declares `needs_schedule()`; the dataloader reads the statistics first,
derives the schedule of kick-offs from them, and only then plans the odds. `prepare(dry_run=True)` may read free data to build the
plan, but never a priced item.

**Rationale**: The Odds API addresses historical prices by **timestamp**, not by season. "The price at minute 45" is a request for
the snapshot at `kick-off + 45min`, so the source cannot plan anything from `{'league': 'England', 'year': 2025}` alone. It has to
know when the matches are.

Two ways to give it that. The source could discover the schedule itself, through the vendor's historical-events endpoint, at about
40 credits a season. Or the statistics could supply it — which became possible once `date` carried the kick-off instant (D13).

The second is strictly better, and the reason is worth stating: **the statistics are free.** So the dataloader can read them, count
the snapshots the odds source would need, and quote an exact price — having spent nothing. Measured on a real season: 432 snapshots,
8,642 credits, quoted for **zero** credits. Self-discovery would have cost ~40 credits to reach the same number.

**FR-012 is amended** rather than violated. It said a dry run performs "no fetch". The property that actually matters is that it
**spends nothing**. Reading a free catalogue and free statistics in order to make the estimate *exact* is worth far more to a user
than refusing to read anything and reporting "unknown".

**Credential handling**: the vendor takes its key as a query parameter, not a header, so the key cannot simply be omitted from a
`RawItem` and added by the fetch layer. Instead a source exposes `request_url(item)`, which the store calls at the moment of the
request. The key therefore never reaches a `RawItem`, never reaches the manifest, and never touches the disk — asserted by a test
that scans the whole store for it.

**Cost model, pinned against the vendor's documentation** rather than guessed: historical odds cost `10 x markets x regions`, live
odds cost `markets x regions`, and `/v4/sports` is free. Matches that kick off together share a snapshot and are paid for once,
which is what keeps a season in the thousands of credits rather than the tens of thousands.

## D15. A wrong alias is worse than an unmatched team

**Decision**: normalize away what is genuinely noise (casing, accents, punctuation, the club words leagues sprinkle differently),
then compare **exactly**. Bridge the rest with an alias table. Never accept a resemblance. Default `max_unmatched_rate=0.0`, and
raise `UnmatchedError` past it.

**Rationale**: the two sources do not merely *format* names differently, they use *different names*: `Man United` against
`Manchester United`, `Wolves` against `Wolverhampton Wanderers`, `Nott'm Forest` against `Nottingham Forest`. No normalization
bridges those, so an alias table is not optional.

The temptation is a fuzzy matcher. It is the wrong call, and the reason is asymmetric:

- An **unmatched** team is loud. The match has no odds, the rate rises, the gate fires.
- A **wrong** alias is silent. It attaches one club's odds to another club's match, and reports nothing. The dataset looks
  complete. The backtest looks clean. It is wrong.

`Manchester United` and `Manchester City` are one token apart. Any similarity threshold that bridges `Man United` will eventually
bridge those two, and nothing will say so. So the library reports resemblances as **suggestions** and applies none of them.

**What makes it usable anyway**: the failure is mechanical to fix. It names the clubs it could not place, ranks what they resemble,
and prints a paste-ready `aliases={...}` dict. The user checks it — a resemblance is not a fact — and passes it back.

**Only aliases that can be verified are shipped.** Inventing entries for leagues whose club names cannot be checked would be
inventing exactly the silent corruption above. An alias the library lacks fails loudly, which is safe; an alias the library got
wrong does not, which is not.

**The odds take the statistics' identity**, not their own: the statistics say which matches exist, the odds say what they were
priced at. So the reconciled odds carry the statistics' kick-off and spelling, and the two tables line up exactly rather than
nearly.
