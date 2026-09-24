# Implementation Plan: Pluggable Data-Source Providers

**Branch**: `002-data-source-providers` | **Date**: 2026-07-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-data-source-providers/spec.md`

## Summary

Replace the mirrored dataset with injectable data sources. `SoccerDataLoader` keeps its shape but gains three constructor parameters — `stats`, `odds`, `store` — each an object owning its own configuration. The football-data.co.uk ETL that currently runs on the `data` branch is relocated into the library so it runs on the user's machine; a keyed `OddsApi` source adds time-stamped in-play prices. Fetching happens only in an explicit `prepare()` step that is incremental, resumable and (for metered sources) cost-estimating; extraction never fetches and fails loudly on an unprepared store.

The architecture rests on one inversion: **sources do not fetch**. A source declares *what raw items* a `param_grid` needs (`required_items`) and *how to turn raw payloads into long snapshots* (`to_snapshots`). The store fetches and persists. This is what lets `FootballDataStats` and `FootballDataOdds` — which read the same upstream CSV — share a single download instead of pulling it twice, and it is what makes `prepare(dry_run=True)` possible at all: the plan is a pure set-difference computed with zero network access.

`BaseDataLoader._snapshots()` is unchanged. Everything here lives behind that existing seam.

## Technical Context

**Language/Version**: Python >=3.11, <3.14 (targets py311)

**Primary Dependencies**: pandas, numpy, scikit-learn, pandera, aiohttp (fetching), beautifulsoup4 (already declared, currently unused — will be used for source index discovery), click/rich (CLI). **New**: `pyarrow` (Parquet).

**Storage**: Local filesystem. Parquet (zstd) for derived snapshot tables, gzipped raw payloads, a JSONL manifest index. Atomic writes (write-temp-then-rename). No embedded or remote database.

**Testing**: pytest with `--doctest-modules`, branch coverage, `pytest-randomly`, `pytest-xdist`. **Hard constraint: no test may touch the network.** Sources are tested against recorded payloads in `tests/samples/`; the fetch layer is tested against a local server fixture, not the internet.

**Target Platform**: Linux/macOS/Windows (CI runs all three on 3.11–3.13).

**Project Type**: Python library with a CLI and an optional GUI. Three delivery surfaces must stay in parity (Constitution I) — `prepare()` therefore needs a CLI subcommand and a GUI step.

**Performance Goals**: Preparing one league-season from the free source completes in seconds. Reading a `param_grid`-scoped slice out of a fully populated store (all leagues, all seasons, 10^5–10^6 snapshot rows) into pandas completes in a few seconds. The incremental plan (set-difference over ~10^4 keys) is effectively instant.

**Constraints**:

- `prepare(dry_run=True)` and all planning must perform **zero** network calls and incur **zero** cost.
- `extract_train_data` / `extract_fixtures_data` must perform **zero** network calls, ever.
- Credentials never written to the store, to logs, or to error messages.
- Exactly one new runtime dependency.

**Scale/Scope**: ~900 league-seasons upstream, ~10^5 matches, gigabyte-scale at the extreme (once five-minute in-play snapshots multiply the row count). Laptop-scale by design.

## Constitution Check

*GATE: evaluated before Phase 0 research, re-evaluated after Phase 1 design.*

| Principle | Status | Notes |
| --- | --- | --- |
| **I. scikit-learn-Compatible API** | PASS (with obligations) | Sources and store are constructor parameters stored unmodified; no validation in `__init__`; state produced by `prepare()` lands on trailing-underscore attributes. **Obligation**: `prepare()` is a new capability and MUST be reachable from the CLI and the GUI, not only the Python API. Tasks cover both. |
| **II. Type Safety & Schema Validation** | PASS | Full annotations; the source and store contracts are ABCs with typed signatures. Snapshots crossing the `_snapshots()` boundary keep their existing pandera validation. The store must round-trip dtypes (FR-020) — Parquet is chosen precisely because it does. |
| **III. Test Coverage & Doctest Discipline** | PASS (with constraint) | `--doctest-modules` executes every `>>>` in `src/`. Network-touching classes (`SoccerDataLoader`, `FootballData*`, `OddsApi`) therefore MUST NOT carry executable doctests that fetch — the existing `SoccerDataLoader` already has none, and that precedent is preserved. Offline classes (`LocalStore`, the reports) get real doctests. Bug fixes get regression tests. |
| **IV. Automated Quality Gates** | PASS | No new gate exemptions sought. `bandit` will scrutinise the credential handling and the decompression path; `pip-audit` covers `pyarrow`. |
| **V. Documentation as a First-Class Artifact** | PASS | The user guide, `in_practice.md`, the gallery examples and the README all construct a `SoccerDataLoader` and all must gain the `prepare()` step. A CHANGELOG entry is required (breaking change). |

**Result**: PASS. Three deviations require justification; they are recorded in Complexity Tracking below.

## Project Structure

### Documentation (this feature)

```text
specs/002-data-source-providers/
├── plan.md              # This file
├── research.md          # Phase 0 output — decisions and rejected alternatives
├── data-model.md        # Phase 1 output — entities and their fields
├── quickstart.md        # Phase 1 output — how to validate the feature end to end
├── contracts/           # Phase 1 output — the public interfaces
│   ├── source.md
│   ├── store.md
│   └── dataloader.md
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
src/sportsbet/datasets/
├── __init__.py                 # re-exports: + FootballDataStats, FootballDataOdds, OddsApi, LocalStore
├── _base/
│   ├── _dataloader.py          # BaseDataLoader — _snapshots() contract UNCHANGED; gains prepare() orchestration
│   ├── _factory.py             # from_snapshots / from_dataframe — untouched
│   └── _schema.py              # untouched
├── _sources/                   # NEW
│   ├── __init__.py
│   ├── _base.py                # BaseSource, BaseStatsSource, BaseOddsSource, RawItem
│   ├── _football_data.py       # FootballDataStats, FootballDataOdds + the relocated ETL
│   └── _odds_api.py            # OddsApi
├── _store.py                   # NEW — BaseStore, LocalStore, PreparationReport, NotPreparedError
├── _resolver.py                # NEW — match-identity resolution, ReconciliationReport
├── _fetch.py                   # NEW — the async fetch layer, extracted from _soccer/_dataloader.py
└── _soccer/
    ├── _dataloader.py          # SoccerDataLoader — becomes source composition, not download logic
    └── _dummy.py               # untouched

src/sportsbet/cli/_data.py      # gains a `prepare` subcommand
src/sportsbet/gui/app/          # gains a prepare step in the dataloader creation flow

tests/datasets/
├── test_base.py                # existing
├── test_schema.py              # existing
├── test_dummy.py               # existing — must keep passing untouched
├── test_soccer.py              # existing — rewritten to the new construction
├── test_sources.py             # NEW — the source contract and the relocated ETL, against recorded payloads
├── test_store.py               # NEW — round-trip, atomicity, cache policy
├── test_resolver.py            # NEW — reconciliation and the quality gate
└── test_equivalence.py         # NEW — the golden-fingerprint gate (Phase 0)

tests/samples/                  # recorded upstream payloads + the golden fingerprint
```

**Structure Decision**: Flat test files under `tests/datasets/`, one per source module — the maintainer's convention; no nested test packages. `_sources/` is a package because it holds one module per provider and will grow; `_store.py`, `_resolver.py` and `_fetch.py` are single modules because they are provider-agnostic and small. Nothing new is added under `_soccer/`: sources are sport-agnostic in shape, and the soccer specialisation is simply *which* sources a soccer loader defaults to.

## Implementation Phases

Each phase is independently shippable and independently verifiable. A phase boundary is a point at which the suite is green and the library is releasable.

### Phase 0 — Freeze the equivalence gate

Capture, from the *current* mirrored data, a fingerprint of the `stats`/`odds` long tables and the extracted `X`/`Y`/`O` for a fixed `param_grid` (England, division 1, one completed season). Commit the **fingerprint** — per-column hashes, shapes, dtypes and the ordered column list — not the frames. Any later change that alters the free path's output changes the fingerprint and fails the test, and the failure localises to a named column.

Committing the frames themselves would redistribute odds (FR-025). A fingerprint gates just as tightly and is not data. See Complexity Tracking.

**Done when**: `tests/datasets/test_equivalence.py` passes against the current implementation and would fail if any column of the output changed.

### Phase 1 — Introduce the seams, change no behaviour

Add `BaseSource`/`BaseStatsSource`/`BaseOddsSource`, `BaseStore`, and the `_fetch.py` layer extracted verbatim from `_soccer/_dataloader.py`. Add a temporary `_DataBranchStats`/`_DataBranchOdds` pair whose `required_items` are the current mirrored CSV URLs and whose `to_snapshots` is a `read_csv`. Rewire `SoccerDataLoader._snapshots()` to compose them. Default construction behaves exactly as it does today.

This pair is scaffolding, deleted in Phase 4. It exists so the seam lands under the protection of the full existing test suite and can be reverted cleanly if the shape proves wrong.

**Done when**: every existing test passes unmodified and the Phase 0 fingerprint still matches.

### Phase 2 — Relocate the ETL, client-side

Port `_preprocess_data`, `_convert_data_types`, `_extract_features`, `_to_snapshots` and the index-page URL discovery from the `data` branch's `soccer.py` into `_sources/_football_data.py`. **Port, do not rewrite** — the fingerprint is the gate and a rewrite will not reproduce it. `_extract_features` in particular depends on the `-1`-for-missing-int sentinel that `_convert_data_types` introduces, on row order, and on the fixtures rows being concatenated into the season frame *before* the expanding means are computed.

`FootballDataStats` and `FootballDataOdds` declare the *same* `RawItem` keys (one per league-season CSV, plus the global fixtures CSV). The plan dedupes them, so each CSV is fetched once and both sources transform the same cached payload.

**Done when**: `FootballDataStats` + `FootballDataOdds`, fetching from the live upstream, reproduce the Phase 0 fingerprint exactly.

### Phase 3 — The store and `prepare()`

`LocalStore`: raw payloads (gzipped, keyed by item), derived snapshot tables (Parquet, partitioned), and a JSONL manifest recording what is held, from which source, and whether it is immutable or volatile. Atomic writes throughout. `prepare(dry_run=True)` returns a `PreparationReport`; `prepare()` executes it with progress. `extract_train_data`/`extract_fixtures_data` raise `NotPreparedError` naming exactly what is missing.

**Done when**: SC-004 (extraction never fetches — enforced by a socket guard that fails the suite if a connection is attempted), SC-005, SC-006 and SC-007 all hold.

### Phase 4 — Flip the default, purge the mirror

`SoccerDataLoader` defaults to `FootballDataStats()` / `FootballDataOdds()`. Delete `_DataBranchStats`/`_DataBranchOdds`. Update the user guide, `in_practice.md`, the gallery examples, the README, the CLI and the GUI. Retire the `data` branch and remove the redistributed odds from it. Ship as a breaking release.

**Done when**: no published artifact or branch of the repository contains third-party odds (SC-001), and `prepare()` plus extraction works end to end from a clean checkout.

### Phase 5 — `OddsApi`

The keyed source: historical snapshots (five-minute granularity), upcoming and in-play prices. A cost model for the estimate. Raw responses retained forever (FR-016). Rate limiting and quota-exhaustion handling. No network in tests — recorded payloads only.

**Done when**: the US2 and US3 acceptance scenarios pass against recorded payloads, and an in-play target can be priced at the moment it occurred (SC-009).

### Phase 6 — The resolver and the quality gate

Match-identity resolution across sources: normalised team names via a per-source alias table, competition mapping, and kickoff within a tolerance window. `ReconciliationReport` with matched/unmatched counts and examples. Hard failure above `max_unmatched_rate`. Engaged only when stats and odds come from different sources.

**Done when**: the US4 acceptance scenarios pass, and a deliberately mismatched pair of sources fails loudly rather than emitting NaN odds.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
| --- | --- | --- |
| New runtime dependency: `pyarrow` | The store must round-trip column types (FR-020). We have already been bitten by CSV's dtype amnesia: an empty `fixtures.csv` read back as all-object poisoned `division`/`year` and broke schema validation. Parquet also stores 5–10× smaller, which starts to matter once in-play snapshots multiply the row count. | CSV: the exact bug class we are trying to close. Pickle: not a durable archive format and unsafe to read back. SQLite/DuckDB: an engine and a query language we do not need — the access pattern is "read a `param_grid` slice into a dataframe" and "set-difference over ~10^4 keys". |
| The golden fixture is a **fingerprint**, not the frames | FR-025 forbids redistributing odds, and the equivalence gate has to live in CI. Column-level hashes gate equivalence exactly, are not data, and still localise a failure to a named column. | Committing the frames: redistributes odds — precisely what this feature exists to stop. Not gating in CI: makes the migration hopeful rather than verified, which is the entire point of Phase 0. |
| Throwaway `_DataBranch*` sources in Phase 1 | Lets the source/store seam land with provably zero behaviour change, under the full existing test suite, and be reverted cleanly. Roughly 40 lines, deleted in Phase 4. | Landing the seam and the client-side ETL together: two large changes at once, with no way to attribute a regression to either. |

## Post-Design Constitution Re-Check

Re-evaluated after the Phase 1 artifacts (`data-model.md`, `contracts/`):

- **I**: The design keeps `__init__` free of validation and puts all derived state behind `prepare()`. The CLI/GUI parity obligation is captured as explicit tasks rather than left implicit. PASS.
- **II**: `RawItem`, `PreparationReport` and `ReconciliationReport` are typed; the store's Parquet round-trip is the mechanism by which the existing pandera schemas stay satisfiable. PASS.
- **III**: "No network in tests" becomes a testable invariant (a socket guard in `conftest.py`) rather than an aspiration. PASS.
- **IV, V**: unchanged. PASS.

No new violations. The three Complexity Tracking entries stand.
