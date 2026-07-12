---

description: "Task list for Pluggable Data-Source Providers"
---

# Tasks: Pluggable Data-Source Providers

**Input**: Design documents from `/specs/002-data-source-providers/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/)

**Tests**: Required. The constitution mandates tests for every behavioural change, and the whole migration hangs on an equivalence gate — so tests are not optional here.

**Organization**: Grouped by the seven implementation phases in [plan.md](plan.md), because each phase is independently shippable and verifiable. `[US*]` labels carry traceability back to the user stories in [spec.md](spec.md).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on an incomplete task)
- **[Story]**: The user story served (US1–US5)
- Every task names an exact file path

## Conventions (apply to every task)

- No explanatory inline comments, in source or in tests. Short Google-style docstrings.
- Flat test files under `tests/datasets/` — no nested test packages.
- Never import a private name from a private module in a test. Test the public API, or re-export.
- `--doctest-modules` runs every `>>>` in `src/`: **no executable doctests on network-touching classes**.
- Line length 120, `skip-string-normalization`.

---

## Phase 0: Freeze the equivalence gate (Blocking)

**Purpose**: Capture what today's mirrored data produces, so every later phase can be checked against it rather than hoped about. Nothing else may start until this exists — once the `data` branch is purged in Phase 4, the reference can never be recaptured.

**Goal**: A CI-runnable test that fails if the free path's output drifts from the mirror's by a single column.

- [X] T001 Add `pyarrow>=17.0.0` to `[project.dependencies]` in `pyproject.toml` and refresh the lock
- [X] T002 Write the reference capture script at `tests/samples/capture_reference.py`: for `param_grid={'league': ['England'], 'division': [1], 'year': [2024]}`, build the current `SoccerDataLoader`, and emit `stats`, `odds`, `X`, `Y`, `O`
- [X] T003 Extend `tests/samples/capture_reference.py` to reduce each frame to a fingerprint — ordered column list, per-column dtype, row count, and a per-column content hash — and write it to `tests/samples/reference_fingerprint.json`
- [X] T004 Run the capture against the live mirror and commit `tests/samples/reference_fingerprint.json` (hashes only — committing the frames would redistribute odds, FR-025; see research D7)
- [X] T005 [US1] Write `tests/datasets/test_equivalence.py`: rebuild the five frames from the loader and assert every fingerprint field matches, failing with the name of the drifted column
- [X] T006 Add the uncommitted reference-frame path to `.gitignore` so a local capture for row-level diffing can never be committed by accident

**Checkpoint**: `pdm tests -k equivalence` passes against the current implementation and would fail if any output column changed.

---

## Phase 1: Introduce the seams, change no behaviour (Blocking)

**Purpose**: Land the source and store abstractions under the protection of the existing suite, with a default that behaves exactly as today. Fully revertible.

**Goal**: Every existing test passes unmodified, and the Phase 0 fingerprint still matches.

### Test infrastructure

- [X] T007 Add a session-scoped socket guard to `tests/conftest.py` that fails any test attempting a network connection (this is the enforcement mechanism for SC-004, not a nicety)
- [X] T008 Run the suite with the guard active and confirm it is already green — `tests/datasets/test_soccer.py` monkeypatches `_snapshots`/`_read_csvs` and never actually fetches, so no test migration is expected. If anything does reach for the network, migrate it to a recorded payload rather than exempting it.

### Contracts

- [X] T009 [P] Create `src/sportsbet/datasets/_sources/__init__.py`
- [X] T010 Create `src/sportsbet/datasets/_sources/_base.py` with `RawItem`, `RawPayload`, `BaseSource` (ABC: `index_items`, `available_params`, `required_items`, `to_snapshots`, `estimate`), and the `BaseStatsSource`/`BaseOddsSource` markers, per [contracts/source.md](contracts/source.md)
- [X] T011 Create `src/sportsbet/datasets/_store.py` with `BaseStore` (ABC: `held`, `fetch`, `read`, `write_snapshots`, `read_snapshots`), `PreparationReport` and `NotPreparedError`, per [contracts/store.md](contracts/store.md). No `LocalStore` yet — that is Phase 3.
- [X] T012 Create `src/sportsbet/datasets/_fetch.py` by extracting `_read_url_content_async`, `_read_urls_content_async`, `_read_urls_content`, `_read_csvs` and `CONNECTIONS_LIMIT` verbatim from `src/sportsbet/datasets/_soccer/_dataloader.py`

### The reversible default

- [X] T013 [US1] Add the temporary `_DataBranchStats` / `_DataBranchOdds` to `src/sportsbet/datasets/_sources/_base.py`: `required_items` returns the current mirrored CSV URLs, `to_snapshots` reads them. Scaffolding — deleted in Phase 4.
- [X] T014 [US1] Rewire `SoccerDataLoader` in `src/sportsbet/datasets/_soccer/_dataloader.py` to accept `stats`/`odds`/`store` (defaulting to the `_DataBranch*` pair) and implement `_snapshots()` by composing them. Constructor parameters stored unmodified, no validation in `__init__` (Constitution I).
- [X] T015 [US1] Moved `_all_params()` onto the stats source's `available_params()` in Phase 2 (deferred from Phase 1, where it would have broken the zero-test-edit rule). `get_all_params` became an instance method: with an injectable source, what is available depends on the configured loader, not on its class.
- [X] T016 [US5] Add a no-op `prepare()` to `BaseDataLoader` in `src/sportsbet/datasets/_base/_dataloader.py` returning an empty `PreparationReport`, so `DummySoccerDataLoader`, `from_snapshots` and `from_dataframe` keep working with no store and no source (FR-028)
- [X] T017 [P] [US1] Write `tests/datasets/test_sources.py` covering the source contract: `required_items` is deterministic, two sources declaring the same key dedupe to one item, and a source performs no I/O

**Checkpoint**: `pdm run tests` green with **zero edits to existing tests**, and `test_equivalence.py` still passes. If an existing test needed changing, the seam changed behaviour and the design is wrong.

---

## Phase 2: Relocate the ETL, client-side (US1 — P1) 🎯 MVP

**Purpose**: Stop mirroring. The transform runs on the user's machine.

**Goal**: `FootballDataStats` + `FootballDataOdds` reproduce the Phase 0 fingerprint exactly.

**Independent Test**: `pdm tests -k equivalence` passes with the loader constructed from the football-data sources instead of the `_DataBranch*` pair.

> **Port, do not rewrite** (research D4). Three behaviours a clean-room rewrite will silently miss, each of which changes the output: the `-1`-for-missing-int sentinel that `_to_snapshots` reads to decide whether a match was played; the fixtures rows being concatenated onto the season frame *before* the expanding means are computed; and the back-fill of each odds column from its `_closing` twin before the closing columns are dropped.

- [X] T018 [US1] Create `src/sportsbet/datasets/_sources/_football_data.py` and port the constants from the `data` branch's `soccer.py`: `URL`, `BASE_URLS`, `LEAGUES_MAPPING`, `REMOVED_COLS`, `COLS_MAPPING`, `SCHEMA`
- [X] T019 [US1] Port `_preprocess_data` into `_football_data.py` — preserving the closing-odds back-fill and the schema reindex
- [X] T020 [US1] Port `_convert_data_types` into `_football_data.py` — preserving the `-1` fill for integer columns and the `('-', '`', 'x')` → NaN replacement
- [X] T021 [US1] Port `_get_output_cols_mapping`, `_rename_modelling_columns` and `_extract_features` into `_football_data.py` — preserving the expanding/rolling means, the `shift(1)`, and the adjusted-goals formulas
- [X] T022 [US1] Port `_to_snapshots` into `_football_data.py`, calling the **existing public** `market_outcomes` from `src/sportsbet/datasets/_soccer/_utils.py` rather than reintroducing the `data` branch's `_market_outcomes`
- [X] T023 [US1] Port the index-page discovery (`extract_raw_training_urls`) into `FootballDataStats.index_items()` / `available_params()`, fetching only the index pages for the leagues in the `param_grid` — not all 27 (SC-010)
- [X] T024 [US1] Implement `FootballDataStats.required_items()` and `FootballDataOdds.required_items()` returning **identical** `RawItem` keys (one per league-season CSV plus the global fixtures CSV), so the shared upstream file is fetched once, not twice
- [X] T025 [US1] Implement `FootballDataStats.to_snapshots()` and `FootballDataOdds.to_snapshots()` on top of the ported pipeline, marking completed seasons `volatile=False` and the current season and fixtures `volatile=True`
- [X] T026 [US1] **Changed**: the upstream payloads are NOT committed (they carry bookmaker odds). The transform is tested offline against a synthetic feed-shaped CSV in `tests/datasets/test_football_data.py`, and the real feed is checked by the `network` marked equivalence gate.
- [X] T027 [US1] Extend `tests/datasets/test_sources.py` with the ETL tests against the recorded payloads: the `-1` sentinel survives, fixtures get a preplay-only snapshot, closing odds are back-filled, and no index page outside the `param_grid` is requested
- [X] T028 [US1] Verify the equivalence gate against the live upstream with the football-data sources and reconcile any drift **by fixing the port, never by editing the fingerprint**

**Checkpoint**: the client-side path reproduces the mirror exactly (SC-002). This is the MVP: the redistribution problem is now solvable.

---

## Phase 3: The store and `prepare()` (US3 — P1)

**Purpose**: No data request can ever cost money or time by surprise.

**Goal**: `prepare(dry_run=True)` reports cost with zero network calls; extraction on an unprepared store fails loudly and fetches nothing.

**Independent Test**: `pdm tests -k "store or prepare"`, with the socket guard proving no request escapes.

- [X] T029 [US3] Implement `LocalStore` in `src/sportsbet/datasets/_store.py`: `raw/<source>/<key>.gz`, `snapshots/<source>/<kind>/` as partitioned zstd Parquet, and `manifest.jsonl`
- [X] T030 [US3] Implement atomic writes in `LocalStore` (temp file in the destination directory, then `os.replace`), so a partial write is never readable under its final name (FR-019)
- [X] T031 [US3] Implement `LocalStore.held()` as a pure manifest lookup, treating `volatile` items as always-stale — no TTL, no manual invalidation (FR-018)
- [X] T032 [US3] Implement `BaseDataLoader.prepare(dry_run=False)` in `src/sportsbet/datasets/_base/_dataloader.py`: plan as `required(param_grid) - held()`, union over sources so shared items dedupe, `dry_run` returning the `PreparationReport` without fetching (FR-012)
- [X] T033 [US3] Populate `PreparationReport.unavailable` with `param_grid` combinations the sources do not publish, so an impossible request is a plan-time answer rather than a fetch-time 404
- [X] T034 [US3] Make `extract_train_data` and `extract_fixtures_data` raise `NotPreparedError` on an unprepared store, carrying the report so the message names what is missing and what it would cost (FR-013). No flag may turn this into a fetch.
- [X] T035 [US3] Add progress reporting to `prepare()` via `rich` (FR-014)
- [X] T036 [P] [US3] Write `tests/datasets/test_store.py`: dtype round-trip (FR-020), atomicity under an interrupted write, volatile-vs-immutable refresh, and concurrent appends to the manifest
- [X] T037 [P] [US3] Write the `prepare()` tests in `tests/datasets/test_soccer.py`: dry run fetches nothing, re-preparing a complete store fetches nothing (SC-006), rebuilding derived tables after a transform change fetches nothing and costs nothing (SC-007), and extraction on an unprepared store raises
- [X] T038 [US3] Add a `prepare` subcommand with `--dry-run` to `src/sportsbet/cli/_data.py` (Constitution I: a surface that cannot prepare cannot extract)
- [X] T039 [US3] Add a preparation step to the GUI dataloader-creation flow in `src/sportsbet/gui/app/`, showing the estimate before spending anything
- [X] T040 [P] [US3] Extend `tests/test_cli.py` for the `prepare` subcommand and its dry run

**Checkpoint**: SC-004, SC-005, SC-006 and SC-007 all hold.

---

## Phase 4: Flip the default, purge the mirror (US1 — P1)

**Purpose**: Ship the legal fix. This is the phase that actually removes the exposure.

**Goal**: No branch and no published artifact contains third-party odds.

- [X] T041 [US1] Default `SoccerDataLoader` to `FootballDataStats()` / `FootballDataOdds()` / `LocalStore()` in `src/sportsbet/datasets/_soccer/_dataloader.py`
- [X] T042 [US1] Delete `_DataBranchStats` / `_DataBranchOdds` from `src/sportsbet/datasets/_sources/_base.py` and the mirrored-CSV URL constants from `_soccer/_dataloader.py`
- [X] T043 [US1] Export `FootballDataStats`, `FootballDataOdds` and `LocalStore` from `src/sportsbet/datasets/__init__.py`
- [X] T044 [P] [US1] Update `docs/overview/user_guide/dataloader.md`: the sources model, the `prepare()` step, and the "bring your own data" path
- [X] T045 [P] [US1] Update `docs/overview/user_guide/in_practice.md` with the prepare-then-extract flow
- [X] T046 [P] [US1] Update `docs/examples/plot_soccer_data.py` and `docs/examples/modelling/plot_classifier_bettor.py` to call `prepare()`
- [X] T047 [P] [US1] Update `README.md` — the quickstart constructs a `SoccerDataLoader` and now needs `prepare()`
- [X] T048 [P] [US1] Check `docs/overview/user_guide/bettor.md` for `SoccerDataLoader` construction and update if affected
- [ ] T049 [US1] **Not hand-editable.** `CHANGELOG.md` is generated by `git-changelog` from commit messages at release time, so the breaking changes must be carried by the release commit message, not written into the file.
- [ ] T050 [US1] **Blocked, needs the maintainer.** `git push origin --delete data` is an irreversible remote action and was refused by the permission guard. 919 odds CSVs are still published on that branch.
- [ ] T051 [US1] Verify SC-001: no branch, tag or release artifact of the repository contains third-party odds

**Checkpoint**: the redistribution problem is gone, and the library works end to end from a clean checkout.

---

## Phase 4b: Move discovery to the source (US1 — P1)

**Purpose**: Writing a `param_grid` requires knowing what exists, so discovery cannot live on an object constructed
*with* a `param_grid`. See research D11.

**Goal**: `FootballDataStats().available_params()` answers the question, no dataloader involved. The dataloader has no
public discovery method.

**Independent Test**: `pdm tests -k "sources or soccer"`, plus a check that `get_all_params` is gone from the public API.

- [X] T069 [US1] Split the source contract in `src/sportsbet/datasets/_sources/_base.py`: `catalogue(payloads)` stays pure, and a new public `available_params(store=None)` resolves the catalogue through the store and returns the combinations (FR-031, FR-032)
- [X] T070 [US1] Rename the football-data source's catalogue parser to `catalogue` in `src/sportsbet/datasets/_sources/_football_data.py`, keeping it pure
- [X] T071 [US1] Delete the public `get_all_params` from `src/sportsbet/datasets/_base/_dataloader.py` (FR-033)
- [X] T072 [US1] Make the dataloader's private `_all_params()` the **intersection** of the statistics and odds sources in `src/sportsbet/datasets/_soccer/_dataloader.py`, so a season only one source publishes is never selected (FR-034)
- [X] T073 [US1] Point the CLI at the source: the config gains `STATS`/`ODDS`/`STORE` mirroring the constructor, and `dataloader params` in `src/sportsbet/cli/_data.py` asks the statistics source
- [X] T074 [US1] Point the GUI's parameter pickers at the source in `src/sportsbet/gui/app/states.py`
- [X] T075 [P] [US1] Test discovery in `tests/datasets/test_sources.py`: a source answers without a dataloader, and two sources with different coverage intersect rather than union
- [X] T076 [P] [US1] Update `docs/overview/user_guide/dataloader.md`, `docs/examples/plot_soccer_data.py` and `README.md` for the new discovery entry point

**Checkpoint**: no public `get_all_params` anywhere, and a user can discover parameters before constructing anything.

---

## Phase 5: The commercial odds source (US2 — P1)

**Purpose**: Make in-play betting real rather than notional. This is the first time we can know the price that was actually available at minute 45.

**Goal**: A user's own key buys time-stamped odds; nothing about that key or its data leaves their machine.

**Independent Test**: `pdm tests -k odds_api` against recorded payloads — no network, no credits.

- [X] T052 [US2] Create `src/sportsbet/datasets/_sources/_odds_api.py` with `OddsApi(key, markets, regions)`, per [contracts/source.md](contracts/source.md)
- [X] T053 [US2] Pin the vendor's cost multipliers and historical-coverage start date against the live documentation and implement `OddsApi.estimate()` (research D8 — pinned, not guessed)
- [X] T054 [US2] Implement `required_items()`: historical snapshot items for completed seasons, the live endpoint for the current one, `volatile` only for the latter
- [X] T055 [US2] Implement `to_snapshots()` mapping the vendor's response into the long odds schema, with `event_status`/`event_time` taken from the snapshot timestamp — this is what makes an in-play target priceable (SC-009)
- [X] T056 [US2] Inject the key at fetch time via request headers only. It must never reach a `RawItem`, the manifest, a log line or an error message (FR-027).
- [X] T057 [US2] Handle rate limits and quota exhaustion: fail with the source named and the reason given, leaving the store unchanged
- [X] T058 [P] [US2] Record vendor payloads into `tests/samples/odds_api/` and extend `tests/datasets/test_sources.py`
- [X] T059 [P] [US2] Add a `bandit`-clean credential-handling test asserting no key appears anywhere under the store path or in a raised message

**Checkpoint**: US2 and US3 acceptance scenarios pass; an in-play target can be priced at the moment it occurred.

---

## Phase 6: The resolver and the quality gate (US4 — P2)

**Purpose**: A failed join is the most dangerous thing in this feature. It does not look like an error — it looks like a slightly smaller dataset and a suspiciously clean backtest.

**Goal**: Mixing sources either reconciles, or fails loudly. It never quietly emits missing odds.

**Independent Test**: `pdm tests -k resolver` — deliberately mismatched naming produces an accurate unmatched rate and a raised error, not NaN odds.

- [ ] T060 [US4] Create `src/sportsbet/datasets/_resolver.py` with the match identity, the per-source alias tables and the windowed kickoff match
- [ ] T061 [US4] Implement `ReconciliationReport` with matched/unmatched counts, rates and example rows (FR-022, FR-024)
- [ ] T062 [US4] Enforce `max_unmatched_rate` in `SoccerDataLoader`, defaulting strict, raising above tolerance (FR-023)
- [ ] T063 [US4] Skip the resolver entirely when stats and odds come from the same source — the free path's identities are equal by construction
- [ ] T064 [P] [US4] Write `tests/datasets/test_resolver.py`: accurate unmatched rate, a hard failure above tolerance, and reporting-but-not-dropping within tolerance
- [ ] T065 [P] [US4] Document the resolver and the quality gate in `docs/overview/user_guide/dataloader.md`

**Checkpoint**: all user stories functional.

---

## Phase 7: Polish

- [ ] T066 [P] Run the full [quickstart.md](quickstart.md) validation, all eight scenarios
- [ ] T067 [P] Confirm docstring coverage (`interrogate`) and that no network-touching class gained an executable doctest
- [ ] T068 Confirm exactly one new runtime dependency was added (SC-012)

---

## Dependencies & Execution Order

### Phase dependencies

- **Phase 0** blocks everything. The reference cannot be recaptured once Phase 4 purges the mirror.
- **Phase 1** blocks Phases 2–6: they all build on the source and store contracts.
- **Phase 2** (US1) depends on Phase 1. Ships the MVP.
- **Phase 3** (US3) depends on Phase 1; parallel with Phase 2 in principle, but Phase 2 is the higher-value increment.
- **Phase 4** depends on Phases 2 **and** 3 — flipping the default without `prepare()` would leave extraction with nothing to read.
- **Phase 5** (US2) depends on Phase 3: a metered source without cost estimation is exactly the thing the spec forbids.
- **Phase 6** (US4) depends on Phase 5 — until two *different* sources exist, there is nothing to reconcile.

### Story dependencies

- **US1** (free path): Phases 0 → 1 → 2 → 4
- **US3** (prepare/cost): Phase 3 — needed by US2, but independently valuable
- **US2** (BYO odds): Phase 5, after US3
- **US4** (reconciliation): Phase 6, after US2
- **US5** (offline entry points): T016, then verified continuously — every phase's checkpoint requires `test_dummy.py` green and unmodified

### Parallel opportunities

- T009–T012 (the four new modules) are independent files
- T019–T022 are separate functions but land in one file — sequential
- T044–T048 (docs) are all independent
- T036, T037, T040 (Phase 3 tests) are independent files

---

## Implementation Strategy

**MVP = Phases 0 + 1 + 2.** At that point the library can produce its data client-side and we have proved it matches. Phases 3 and 4 make it shippable; Phase 5 delivers the capability that motivated the architecture; Phase 6 makes mixing sources safe.

**Immediate work: Phases 0 and 1.** They are the safe, reversible foundation — Phase 0 costs nothing and can never be done later, and Phase 1 is verified by the existing suite passing *unmodified*.

**Gate at every phase boundary** (Constitution IV):

```bash
pdm run formatting
pdm run checks
pdm run tests
```

A phase is not done until all three are green.
