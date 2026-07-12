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
