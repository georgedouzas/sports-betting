# Implementation Plan: Basketball, starting with the EuroLeague

**Branch**: `003-basketball-euroleague` | **Date**: 2026-07-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/003-basketball-euroleague/spec.md`

## Summary

Add a second sport by adding a data source. The extraction engine, the store, the preparation step, the cost estimate and the reconciliation all came from `002` and are **not changed**. What is new is `EuroLeagueStats` — a free, key-less source over the EuroLeague's official public API — and a `BasketballDataLoader` that is the soccer loader with different default sources.

The plan is shaped by two findings from hitting the live API, both of which changed the scope:

- **The API publishes every game in CET**, whatever country it is played in. Not UTC, not the venue's local time. Determined from the data (research D1); assuming otherwise would silently shift every odds snapshot the user buys by one to two hours.
- **Basketball has no totals line**, so the first cut is head-to-head only (research D2). Total points run 125–229 and a bookmaker sets a different line per game. The library expresses a market as a *column*, and a moving line is not a column.

The real engineering work is not the source — it is **hoisting** the sport-agnostic half of `SoccerDataLoader` onto the base so that `BasketballDataLoader` is a class name and two defaults. If it needs to be more than that, the abstraction is wrong.

## Technical Context

**Language/Version**: Python >=3.11, <3.14 (targets py311)

**Primary Dependencies**: unchanged. XML parsed with the standard library's `xml.etree.ElementTree`. **No new runtime dependency** (SC-010).

**Storage**: unchanged — the existing `LocalStore`.

**Testing**: pytest, `--doctest-modules`, branch coverage, `pytest-randomly`, `pytest-xdist`. **No test may touch the network** — the socket guard enforces it. The EuroLeague transform is tested against payloads shaped like the real XML.

**Target Platform**: Linux/macOS/Windows on 3.11–3.13.

**Project Type**: Python library with a CLI and an optional GUI.

**Performance Goals**: a EuroLeague season is **one** request (research D3), so preparing one is a couple of seconds.

**Constraints**:

- Sources never fetch; only the store does.
- Extraction never fetches; only `prepare()` does.
- `date` is the kick-off instant in **UTC** — every source resolves its own zone at its own boundary.
- No new runtime dependency.
- No basketball data is redistributed.

**Scale/Scope**: ~330 games per EuroLeague season; a handful of seasons. Trivially small next to soccer's ~900 league-seasons.

## Constitution Check

*GATE: evaluated before Phase 0 research, re-evaluated after Phase 1 design.*

| Principle | Status | Notes |
| --- | --- | --- |
| **I. scikit-learn-Compatible API** | PASS | `BasketballDataLoader` takes the same constructor parameters as the soccer one, stored unmodified, no validation in `__init__`. The CLI and GUI reach it through the same `DATALOADERS` mapping, so all three surfaces gain the sport at once. |
| **II. Type Safety & Schema Validation** | PASS | Full annotations. The new source obeys the existing `BaseSource` contract and its snapshots the existing pandera schemas — that is the point of the exercise. |
| **III. Test Coverage & Doctest Discipline** | PASS (with constraint) | `--doctest-modules` runs every `>>>` in `src/`, so `EuroLeagueStats` and `BasketballDataLoader` carry **no executable doctests** (they are network-touching), matching `SoccerDataLoader`. Every behavioural change gets a test; the two bugs found (D2's totals, D7's empty-odds error) get regression tests. |
| **IV. Automated Quality Gates** | PASS | No new exemptions. `pip-audit` is unaffected — no new dependency. |
| **V. Documentation as a First-Class Artifact** | PASS | The user guide currently says the library is soccer-only. It must gain the sport, the source, and the honest statement that basketball needs a paid odds key. CHANGELOG entry via the commit message. |

**Result**: PASS. Two deviations are recorded in Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/003-basketball-euroleague/
├── plan.md              # This file
├── research.md          # Phase 0 — the live-API findings that shaped the scope
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/
│   └── euroleague.md
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
src/sportsbet/datasets/
├── _base/
│   └── _dataloader.py        # GAINS the hoisted, sport-agnostic source/store/prepare/resolve logic
├── _utils.py                 # NEW — `market_outcomes` moves here; it is not a soccer thing any more
├── _sources/
│   ├── _euroleague.py        # NEW — EuroLeagueStats
│   └── _odds_api.py          # gains the basketball_* league mapping
├── _soccer/
│   ├── _dataloader.py        # SHRINKS to its default sources
│   └── _utils.py             # re-exports market_outcomes, so the public name keeps working
└── _basketball/              # NEW
    ├── __init__.py
    └── _dataloader.py        # BasketballDataLoader — a class name and two defaults

tests/datasets/
├── test_euroleague.py        # NEW — the source, against payloads shaped like the real XML
├── test_basketball.py        # NEW — the loader, and the no-draw markets
└── test_equivalence.py       # gains a EuroLeague fingerprint alongside the soccer one
```

**Structure Decision**: `_basketball/` mirrors `_soccer/` so a third sport is obvious. `_utils.py` is promoted out of `_soccer/` because `market_outcomes` now serves two sports — but `sportsbet.datasets.market_outcomes` keeps working, because that is public API.

## Implementation Phases

### Phase 0 — Hoist the engine, change no behaviour

Move the sport-agnostic half of `SoccerDataLoader` (`_resolved`, `sources`, `_catalogue`, `_params`, `_all_params`, `_items`, `_report`, `_unavailable`, `prepare`, `_snapshots`, `_derive`, `_authorize`, `_schedule`, `_unique`) onto `BaseDataLoader`. `SoccerDataLoader` keeps only its default sources.

**Done when**: every existing test passes **unmodified**, and the soccer equivalence gate still matches. If a test needs editing, the hoist changed behaviour and is wrong.

### Phase 1 — `EuroLeagueStats`

The source, per [contracts/euroleague.md](contracts/euroleague.md). One item per season, catalogue from the API, XML by standard library, kick-off in CET → UTC, form from points for/against.

**Done when**: it produces long snapshots from a recorded season payload, offline, with the pre-play/post-play split correct and an unplayed game appearing only as a fixture.

### Phase 2 — `BasketballDataLoader` and the odds mapping

The loader (a class name and two defaults), `OddsApi` gains the `basketball_*` keys, `market_outcomes` moves to `_utils.py`.

**Done when**: a EuroLeague season extracts end to end against the live API, with `Y` carrying a home win and an away win and **no draw**.

### Phase 3 — Reconcile the two rosters

Verify the roster pairing on the **real** names of both sources. Add aliases only where the pairing genuinely fails, each with a reason.

**Done when**: the rosters pair, and an unplaceable club raises rather than silently dropping its game.

### Phase 4 — The honest failure, the gate, the docs

The empty-odds error (D7). A frozen EuroLeague fingerprint. The user guide, which currently says the library is soccer-only.

**Done when**: the full gate is green and the docs no longer lie.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
| --- | --- | --- |
| Hoisting the engine out of `SoccerDataLoader` touches working soccer code | `BasketballDataLoader` would otherwise duplicate the entire preparation, cost-estimate and reconciliation contract — ~200 lines with nothing sport-specific in them — and the two copies would drift apart on the first bug fix. The hoist is protected by the existing suite plus the soccer equivalence gate. | Copy-pasting the loader: two engines to fix every bug in, and a guarantee they diverge. Subclassing `SoccerDataLoader` from `BasketballDataLoader`: a basketball loader that *is-a* soccer loader is a lie. |
| Basketball ships with **no totals market** | The library expresses a market as a column and basketball's line moves per game (125–229 points, different every night). There is no column to write. Faking one — a season median, say — would produce a market no bookmaker offers and a backtest against prices that never existed. | Picking a fixed line: fabricates a market. Extending the data model to a per-row line: real, worth doing, and would change soccer too — so it is its own feature, not a rider on this one. |

## Post-Design Constitution Re-Check

- **I**: The hoist keeps `__init__` free of validation and leaves the fitted state where it was. Both sports reach the CLI and GUI through the same mapping. PASS.
- **II**: The new source implements the existing contract with no new types. If it had needed a new one, the abstraction from `002` would have been wrong. PASS.
- **III**: "No network in tests" holds — the season payload is recorded, and the socket guard proves it. PASS.
- **IV, V**: unchanged. PASS.

No new violations. The two Complexity Tracking entries stand.
