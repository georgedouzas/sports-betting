# Tasks

## 1. Phase 0: Freeze the equivalence gate

- [x] 1.1 Add `pyarrow>=17.0.0` to `[project.dependencies]` in `pyproject.toml` and refresh the lock (T001)
- [x] 1.2 Write the reference capture script at `tests/samples/capture_reference.py`: for
  `param_grid={'league': ['England'], 'division': [1], 'year': [2024]}`, build the current `SoccerDataLoader`, and emit
  `stats`, `odds`, `X`, `Y`, `O` (T002)
- [x] 1.3 Extend `tests/samples/capture_reference.py` to reduce each frame to a fingerprint (ordered column list,
  per-column dtype, row count and a per-column content hash) and write it to
  `tests/samples/reference_fingerprint.json` (T003)
- [x] 1.4 Run the capture against the live mirror and commit `tests/samples/reference_fingerprint.json`, hashes only,
  since committing the frames would redistribute odds (FR-025, research D7) (T004)
- [x] 1.5 Write `tests/datasets/test_equivalence.py`: rebuild the five frames from the loader and assert every
  fingerprint field matches, failing with the name of the drifted column (US1) (T005)
- [x] 1.6 Add the uncommitted reference-frame path to `.gitignore` so a local capture for row-level diffing can never be
  committed by accident (T006)

## 2. Phase 1: Introduce the seams, change no behaviour

- [x] 2.1 Add a session-scoped socket guard to `tests/conftest.py` that fails any test attempting a network connection,
  as the enforcement mechanism for SC-004 (T007)
- [x] 2.2 Run the suite with the guard active and confirm it is already green, since `tests/datasets/test_soccer.py`
  monkeypatches `_snapshots`/`_read_csvs` and never fetches. Migrate anything that reaches for the network to a
  recorded payload rather than exempting it (T008)
- [x] 2.3 Create `src/sportsbet/datasets/_sources/__init__.py` (T009)
- [x] 2.4 Create `src/sportsbet/datasets/_sources/_base.py` with `RawItem`, `RawPayload`, `BaseSource` (ABC:
  `index_items`, `available_params`, `required_items`, `to_snapshots`, `estimate`), and the
  `BaseStatsSource`/`BaseOddsSource` markers, per the source contract (T010)
- [x] 2.5 Create `src/sportsbet/datasets/_store.py` with `BaseStore` (ABC: `held`, `fetch`, `read`, `write_snapshots`,
  `read_snapshots`), `PreparationReport` and `NotPreparedError`, per the store contract, with no `LocalStore` yet
  (T011)
- [x] 2.6 Create `src/sportsbet/datasets/_fetch.py` by extracting `_read_url_content_async`, `_read_urls_content_async`,
  `_read_urls_content`, `_read_csvs` and `CONNECTIONS_LIMIT` verbatim from
  `src/sportsbet/datasets/_soccer/_dataloader.py` (T012)
- [x] 2.7 Add the temporary `_DataBranchStats` / `_DataBranchOdds` to `src/sportsbet/datasets/_sources/_base.py`:
  `required_items` returns the current mirrored CSV URLs and `to_snapshots` reads them. Scaffolding, deleted in Phase 4
  (US1) (T013)
- [x] 2.8 Rewire `SoccerDataLoader` in `src/sportsbet/datasets/_soccer/_dataloader.py` to accept `stats`/`odds`/`store`
  (defaulting to the `_DataBranch*` pair) and implement `_snapshots()` by composing them. Constructor parameters stored
  unmodified, no validation in `__init__` (US1) (T014)
- [x] 2.9 Moved `_all_params()` onto the stats source's `available_params()` in Phase 2 (deferred from Phase 1, where it
  would have broken the zero-test-edit rule). `get_all_params` became an instance method, since with an injectable
  source what is available depends on the configured loader (US1) (T015)
- [x] 2.10 Add a no-op `prepare()` to `BaseDataLoader` in `src/sportsbet/datasets/_base/_dataloader.py` returning an
  empty `PreparationReport`, so `DummySoccerDataLoader`, `from_snapshots` and `from_dataframe` keep working with no
  store and no source (FR-028) (US5) (T016)
- [x] 2.11 Write `tests/datasets/test_sources.py` covering the source contract: `required_items` is deterministic, two
  sources declaring the same key dedupe to one item, and a source performs no I/O (US1) (T017)

## 3. Phase 2: Relocate the ETL client side

- [x] 3.1 Create `src/sportsbet/datasets/_sources/_football_data.py` and port the constants from the `data` branch's
  `soccer.py`: `URL`, `BASE_URLS`, `LEAGUES_MAPPING`, `REMOVED_COLS`, `COLS_MAPPING`, `SCHEMA` (US1) (T018)
- [x] 3.2 Port `_preprocess_data` into `_football_data.py`, preserving the closing-odds back-fill and the schema reindex
  (US1) (T019)
- [x] 3.3 Port `_convert_data_types` into `_football_data.py`, preserving the `-1` fill for integer columns and the
  replacement of `'-'`, a backtick and `'x'` with NaN (US1) (T020)
- [x] 3.4 Port `_get_output_cols_mapping`, `_rename_modelling_columns` and `_extract_features` into `_football_data.py`,
  preserving the expanding and rolling means, the `shift(1)`, and the adjusted-goals formulas (US1) (T021)
- [x] 3.5 Port `_to_snapshots` into `_football_data.py`, calling the existing public `market_outcomes` from
  `src/sportsbet/datasets/_soccer/_utils.py` rather than reintroducing the `data` branch's `_market_outcomes` (US1)
  (T022)
- [x] 3.6 Port the index-page discovery (`extract_raw_training_urls`) into `FootballDataStats.index_items()` /
  `available_params()`, fetching only the index pages for the leagues in the `param_grid`, not all 27 (SC-010) (US1)
  (T023)
- [x] 3.7 Implement `FootballDataStats.required_items()` and `FootballDataOdds.required_items()` returning identical
  `RawItem` keys (one per league-season CSV plus the global fixtures CSV), so the shared upstream file is fetched once
  (US1) (T024)
- [x] 3.8 Implement `FootballDataStats.to_snapshots()` and `FootballDataOdds.to_snapshots()` on top of the ported
  pipeline, marking completed seasons `volatile=False` and the current season and fixtures `volatile=True` (US1) (T025)
- [x] 3.9 Changed: the upstream payloads are not committed, since they carry bookmaker odds. The transform is tested
  offline against a synthetic feed-shaped CSV in `tests/datasets/test_football_data.py`, and the real feed is checked by
  the `network` marked equivalence gate (US1) (T026)
- [x] 3.10 Extend `tests/datasets/test_sources.py` with the ETL tests against the recorded payloads: the `-1` sentinel
  survives, fixtures get a preplay-only snapshot, closing odds are back-filled, and no index page outside the
  `param_grid` is requested (US1) (T027)
- [x] 3.11 Verify the equivalence gate against the live upstream with the football-data sources and reconcile any drift
  by fixing the port, never by editing the fingerprint (US1) (T028)

## 4. Phase 3: The store and prepare

- [x] 4.1 Implement `LocalStore` in `src/sportsbet/datasets/_store.py`: `raw/<source>/<key>.gz`,
  `snapshots/<source>/<kind>/` as partitioned zstd Parquet, and `manifest.jsonl` (US3) (T029)
- [x] 4.2 Implement atomic writes in `LocalStore` (temp file in the destination directory, then `os.replace`), so a
  partial write is never readable under its final name (FR-019) (US3) (T030)
- [x] 4.3 Implement `LocalStore.held()` as a pure manifest lookup, treating `volatile` items as always stale, with no
  TTL and no manual invalidation (FR-018) (US3) (T031)
- [x] 4.4 Implement `BaseDataLoader.prepare(dry_run=False)` in `src/sportsbet/datasets/_base/_dataloader.py`: plan as
  required minus held, union over sources so shared items dedupe, `dry_run` returning the `PreparationReport` without
  fetching (FR-012) (US3) (T032)
- [x] 4.5 Populate `PreparationReport.unavailable` with `param_grid` combinations the sources do not publish, so an
  impossible request is a plan-time answer rather than a fetch-time 404 (US3) (T033)
- [x] 4.6 Make `extract_train_data` and `extract_fixtures_data` raise `NotPreparedError` on an unprepared store,
  carrying the report so the message names what is missing and what it would cost (FR-013). No flag turns this into a
  fetch (US3) (T034)
- [x] 4.7 Add progress reporting to `prepare()` via `rich` (FR-014) (US3) (T035)
- [x] 4.8 Write `tests/datasets/test_store.py`: dtype round trip (FR-020), atomicity under an interrupted write,
  volatile versus immutable refresh, and concurrent appends to the manifest (US3) (T036)
- [x] 4.9 Write the `prepare()` tests in `tests/datasets/test_soccer.py`: a dry run fetches nothing, re-preparing a
  complete store fetches nothing (SC-006), rebuilding derived tables after a transform change fetches nothing and costs
  nothing (SC-007), and extraction on an unprepared store raises (US3) (T037)
- [x] 4.10 Add a `prepare` subcommand with `--dry-run` to `src/sportsbet/cli/_data.py`, since a surface that cannot
  prepare cannot extract (US3) (T038)
- [x] 4.11 Add a preparation step to the GUI dataloader creation flow in `src/sportsbet/gui/app/`, showing the estimate
  before spending anything (US3) (T039)
- [x] 4.12 Extend `tests/test_cli.py` for the `prepare` subcommand and its dry run (US3) (T040)

## 5. Phase 4: Flip the default, purge the mirror

- [x] 5.1 Default `SoccerDataLoader` to `FootballDataStats()` / `FootballDataOdds()` / `LocalStore()` in
  `src/sportsbet/datasets/_soccer/_dataloader.py` (US1) (T041)
- [x] 5.2 Delete `_DataBranchStats` / `_DataBranchOdds` from `src/sportsbet/datasets/_sources/_base.py` and the
  mirrored-CSV URL constants from `_soccer/_dataloader.py` (US1) (T042)
- [x] 5.3 Export `FootballDataStats`, `FootballDataOdds` and `LocalStore` from `src/sportsbet/datasets/__init__.py`
  (US1) (T043)
- [x] 5.4 Update `docs/overview/user_guide/dataloader.md`: the sources model, the `prepare()` step, and the "bring your
  own data" path (US1) (T044)
- [x] 5.5 Update `docs/overview/user_guide/in_practice.md` with the prepare-then-extract flow (US1) (T045)
- [x] 5.6 Update `docs/examples/plot_soccer_data.py` and `docs/examples/modelling/plot_classifier_bettor.py` to call
  `prepare()` (US1) (T046)
- [x] 5.7 Update `README.md`, whose quickstart constructs a `SoccerDataLoader` and now needs `prepare()` (US1) (T047)
- [x] 5.8 Check `docs/overview/user_guide/bettor.md` for `SoccerDataLoader` construction and update if affected (US1)
  (T048)
- [x] 5.9 Closed with nothing to do in the tree. Not hand-editable. `CHANGELOG.md` is generated by `git-changelog` from
  commit messages at release time, so the breaking changes must be carried by the release commit message, not written
  into the file (US1) (T049)
- [x] 5.10 The `data` branch was deleted by the maintainer, along with the Prefect flow. 919 odds CSVs are no longer
  published (US1) (T050)
- [x] 5.11 Verified SC-001: the remote holds no `data` branch, and the only odds in the tree are 13 hand-authored rows
  for the offline dummy loader (US1) (T051)

## 6. Phase 4b: Move discovery to the source

- [x] 6.1 Split the source contract in `src/sportsbet/datasets/_sources/_base.py`: `catalogue(payloads)` stays pure,
  and a new public `available_params(store=None)` resolves the catalogue through the store and returns the combinations
  (FR-031, FR-032) (US1) (T069)
- [x] 6.2 Rename the football-data source's catalogue parser to `catalogue` in
  `src/sportsbet/datasets/_sources/_football_data.py`, keeping it pure (US1) (T070)
- [x] 6.3 Delete the public `get_all_params` from `src/sportsbet/datasets/_base/_dataloader.py` (FR-033) (US1) (T071)
- [x] 6.4 Make the dataloader's private `_all_params()` the intersection of the statistics and odds sources in
  `src/sportsbet/datasets/_soccer/_dataloader.py`, so a season only one source publishes is never selected (FR-034)
  (US1) (T072)
- [x] 6.5 Point the CLI at the source: the config gains `STATS`/`ODDS`/`STORE` mirroring the constructor, and
  `dataloader params` in `src/sportsbet/cli/_data.py` asks the statistics source (US1) (T073)
- [x] 6.6 Point the GUI's parameter pickers at the source in `src/sportsbet/gui/app/states.py` (US1) (T074)
- [x] 6.7 Test discovery in `tests/datasets/test_sources.py`: a source answers without a dataloader, and two sources
  with different coverage intersect rather than union (US1) (T075)
- [x] 6.8 Update `docs/overview/user_guide/dataloader.md`, `docs/examples/plot_soccer_data.py` and `README.md` for the
  new discovery entry point (US1) (T076)

## 7. Phase 5: The commercial odds source

- [x] 7.1 Create `src/sportsbet/datasets/_sources/_odds_api.py` with `OddsApi(key, markets, regions)`, per the source
  contract (US2) (T052)
- [x] 7.2 Pin the vendor's cost multipliers and historical coverage start date against the live documentation and
  implement `OddsApi.estimate()` (research D8, pinned, not guessed) (US2) (T053)
- [x] 7.3 Implement `required_items()`: historical snapshot items for completed seasons, the live endpoint for the
  current one, `volatile` only for the latter (US2) (T054)
- [x] 7.4 Implement `to_snapshots()` mapping the vendor's response into the long odds schema, with
  `event_status`/`event_time` taken from the snapshot timestamp, which makes an in-play target priceable (SC-009) (US2)
  (T055)
- [x] 7.5 Inject the key at fetch time via request headers only. It must never reach a `RawItem`, the manifest, a log
  line or an error message (FR-027) (US2) (T056)
- [x] 7.6 Handle rate limits and quota exhaustion: fail with the source named and the reason given, leaving the store
  unchanged (US2) (T057)
- [x] 7.7 Record vendor payloads into `tests/samples/odds_api/` and extend `tests/datasets/test_sources.py` (US2) (T058)
- [x] 7.8 Add a `bandit`-clean credential-handling test asserting no key appears anywhere under the store path or in a
  raised message (US2) (T059)

## 8. Phase 6: The resolver and the quality gate

- [x] 8.1 Create `src/sportsbet/datasets/_resolver.py` with the match identity, the per-source alias tables and the
  windowed kickoff match (US4) (T060)
- [x] 8.2 Implement `ReconciliationReport` with matched and unmatched counts, rates and example rows (FR-022, FR-024)
  (US4) (T061)
- [x] 8.3 Enforce `max_unmatched_rate` in `SoccerDataLoader`, defaulting strict, raising above tolerance (FR-023) (US4)
  (T062)
- [x] 8.4 Skip the resolver entirely when stats and odds come from the same source, since the free path's identities
  are equal by construction (US4) (T063)
- [x] 8.5 Write `tests/datasets/test_resolver.py`: accurate unmatched rate, a hard failure above tolerance, and
  reporting but not dropping within tolerance (US4) (T064)
- [x] 8.6 Document the resolver and the quality gate in `docs/overview/user_guide/dataloader.md` (US4) (T065)

## 9. Phase 7: Polish

- [x] 9.1 Run the full quickstart validation, all eight scenarios (T066)
- [x] 9.2 Confirm docstring coverage (`interrogate`) and that no network-touching class gained an executable doctest
  (T067)
- [x] 9.3 Confirm exactly one new runtime dependency was added (SC-012) (T068)
