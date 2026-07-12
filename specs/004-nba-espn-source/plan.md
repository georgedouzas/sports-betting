# Implementation Plan: The NBA, as a second basketball league

**Branch**: `004-nba-espn-source` | **Date**: 2026-07-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/004-nba-espn-source/spec.md`

## Summary

Add `NBAStats`, a free, key-less statistics source backed by ESPN, and nothing else.

The NBA is not a new sport. It is a new **league** inside a sport the library already plays, so it reaches the user
through the dataloader that already exists:

```python
BasketballDataLoader(param_grid={'league': ['NBA'], 'year': [2025]}, stats=NBAStats(), odds=OddsApi(key=...))
```

The odds side is already finished — `basketball_nba` has been mapped since feature 003. The engine is already
sport-agnostic — feature 003 hoisted it. So the entire feature is **one new source file, one export, one test file and
some docs**, and that is the point: SC-007 says nothing else may change. If the implementation finds itself editing the
engine, the fetch layer or `RawItem`, the abstraction was wrong, and it must stop and report rather than quietly widen
the seam.

The technical work is therefore small, and the *research* is where the difficulty lived. Two findings shaped everything
and are recorded in [research.md](research.md): the league's own official feed is an annually back-filled archive that
cannot see a season in progress (**D1** — which is why an unofficial feed wins), and the chosen feed files its all-star
exhibition under `regular-season` while filing the playoffs under something other than `STD` (**D5** — which is why the
obvious filter is wrong in both directions at once).

## Technical Context

**Language/Version**: Python `>=3.11, <3.14`, targeting `py311`

**Primary Dependencies**: `pandas`, `pandera`, `aiohttp` — **all already present**. This feature adds none (SC-006). The
payload is JSON, parsed with the standard library's `json`, exactly as `EuroLeagueStats` does.

**Storage**: The existing `LocalStore` — raw payloads gzipped under `raw/<source>/<key>.gz`, derived snapshots as
Parquet keyed by a digest of the payloads and the transform. Nothing new.

**Testing**: `pytest`, branch coverage, `pytest-randomly`, `--doctest-modules`. The socket guard in `tests/conftest.py`
makes any network access from a test an error. New tests are flat, in `tests/datasets/test_nba.py`, and run against
payloads shaped like the real response.

**Target Platform**: Library, plus the `sportsbet` CLI and the `sportsbet-gui` GUI. All three must reach the NBA
(FR-017, Constitution I).

**Project Type**: Library (`src/`-based).

**Performance Goals**: A season is ~1,400 events across 11 monthly requests, fetched concurrently by the existing async
fetcher. Preparation is I/O-bound; there is no computational goal. `prepare(dry_run=True)` must remain free of credit
spend.

**Constraints**:

- Sources never fetch. `index_items`, `catalogue`, `required_items` and `to_snapshots` are pure functions of their
  inputs; only the store touches the network (FR-015).
- Extraction never fetches. `prepare()` is the only thing that downloads.
- No test touches the network. No executable doctest on a network-touching class.
- Ship no data (FR-018, SC-010).
- **Every request must be provably below the feed's 1,000-event cap** (FR-012), because the feed truncates silently.

**Scale/Scope**: One new source (~150 lines), one test module, three doc touches. 30 clubs, ~1,400 games per season, 81
seasons published — of which only those the odds vendor also covers are selectable.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design — still passing, no new violations.*

| Principle | Status | How |
| --- | --- | --- |
| **I. scikit-learn-compatible API** | PASS | `NBAStats` is a source, not an estimator, and is passed as a constructor argument (`stats=`) stored unmodified. No new estimator and no new `param_grid` semantics. The CLI and GUI gain the NBA through the same registry as `EuroLeagueStats`, so no surface holds logic the others cannot reach (FR-017). |
| **II. Type safety & schema validation** | PASS | Full annotations; `mypy` clean. The source emits the long snapshots the existing statistics schema already validates — no new schema, because the NBA is the same shape as the EuroLeague. |
| **III. Test coverage & doctest discipline** | PASS | `tests/datasets/test_nba.py` covers every functional requirement, including two regression tests for the D5 trap. `NBAStats` touches the network, so it gets **no executable doctest** — consistent with every other source. |
| **IV. Automated quality gates** | PASS | `black`, `docformatter`, `ruff` (120), `interrogate`, `bandit`, `pip-audit`. Parsing JSON needs no suppression — which is exactly why the EuroLeague source abandoned XML, whose parser `bandit` flags (B405/B314). |
| **V. Documentation first-class** | PASS | `sources.md` gains the NBA in its source table; the basketball user guide gains its second league; `CHANGELOG.md` gains an entry. The public-API docs audit must still find an example for every public name. |

**No violations, so the Complexity Tracking table is empty and omitted.**

The strongest gate here is not in the constitution at all — it is **SC-007**: *no existing file's behaviour changes, and
every existing test passes unmodified.* That is the falsifiable claim the 003 hoist made, and this feature is its first
independent test.

## Project Structure

### Documentation (this feature)

```text
specs/004-nba-espn-source/
├── plan.md              # This file
├── spec.md              # The requirements
├── research.md          # D1-D9: what the live APIs actually said
├── data-model.md        # The entities and the transform
├── quickstart.md        # How to prove it works
├── contracts/
│   └── nba.md           # The source contract: items in, snapshots out
├── checklists/
│   └── requirements.md
└── tasks.md             # /speckit-tasks output, not this command's
```

### Source Code (repository root)

The complete set of files this feature touches. **The shortness of this list is the deliverable.**

```text
src/sportsbet/datasets/
├── _sources/
│   └── _nba.py                 # NEW. The whole feature.
└── __init__.py                 # +1 export: NBAStats

src/sportsbet/
├── cli/                        # register NBAStats where EuroLeagueStats is registered
└── gui/                        # ditto

tests/datasets/
└── test_nba.py                 # NEW

docs/overview/user_guide/
├── sources.md                  # + the NBA row in the source table
└── dataloader.md               # + basketball's second league

CHANGELOG.md                    # + the entry
```

**Explicitly NOT touched.** A diff against any of these means the design failed:

```text
src/sportsbet/datasets/_fetch.py               # ESPN needs no headers (research D2)
src/sportsbet/datasets/_store.py               # no probe concept needed (research D2)
src/sportsbet/datasets/_sources/_base.py       # RawItem is sufficient as it stands
src/sportsbet/datasets/_base/_sourced.py       # the engine
src/sportsbet/datasets/_basketball/_dataloader.py  # NBAStats is passed in, never defaulted
src/sportsbet/datasets/_sources/_odds_api.py   # basketball_nba is already mapped (research D9)
src/sportsbet/datasets/_sources/_euroleague.py
src/sportsbet/datasets/_sources/_football_data.py
src/sportsbet/datasets/_resolver.py            # unless a club genuinely cannot be placed (research D8)
```

**Structure Decision**: The existing `src/`-based layout, unchanged. A source lives in
`src/sportsbet/datasets/_sources/`, one module per feed, beside `_football_data.py`, `_odds_api.py` and
`_euroleague.py`. The NBA is a *league*, not a sport, so it gets **no** package under
`src/sportsbet/datasets/_basketball/`: that package holds the dataloader for the sport, and the sport already exists.

## Implementation Phases

Every phase ends with the full gate green: `pdm run formatting`, `pdm run checks`, `pdm run tests`.

### Phase 1 — `NBAStats`, offline (US1, US2, US3, US4 — all P1)

The source and its tests, against payloads shaped like the real response. No network, in the code or in the tests.

The two traps get regression tests **first**, because each produces a plausible, quiet, wrong answer:

- an exhibition labelled `regular-season` must still be excluded (D5),
- a playoff game not labelled `STD` must still be included (D5),

and the monthly window gets a guard test, so a later "optimisation" that widens it to save requests fails loudly instead
of silently truncating a season (D4, FR-012).

**Checkpoint**: the source turns a recorded payload into snapshots, offline.

### Phase 2 — The league reaches the user (US1; FR-017)

Export `NBAStats`, and register it with the CLI and the GUI beside `EuroLeagueStats`. Verify end to end against the live
feed that a season prepares and extracts with `X`, `Y` and `O` aligned.

**Checkpoint**: `BasketballDataLoader(stats=NBAStats(), odds=OddsApi(key=...))` produces an NBA dataset.

### Phase 3 — Reconciliation (US5 — P2)

Verify live, against the vendor with the real key, that the roster-bijection resolver pairs all 30 clubs. Add an alias
only where a club genuinely cannot be placed, and record the reason. Expect zero (D8).

**Checkpoint**: the two sources reconcile, or fail loudly and name the club.

### Phase 4 — Docs and the gate

`sources.md`, the user guide, `CHANGELOG.md`. Re-run the public-API docs audit; it must still find an example for every
public name. Confirm SC-006 (no new dependency), SC-007 (no engine change, no test edited) and SC-010 (no data shipped).

## Risks

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| **The feed truncates a request and says nothing.** | A season quietly missing games, and a backtest confidently wrong. | Monthly windows: peak 239 events against a 1,000 cap, a 4× margin (D4). A guard test pins the window so nobody widens it. |
| **An exhibition enters the dataset.** | Invented teams poison the roster bijection, breaking the odds pairing for the *whole* league, and a model trains on a game nobody was trying to win. | The filter excludes an *unrecognised* label rather than admitting it (FR-011). Two regression tests, one per direction of the trap. |
| **ESPN is unofficial, and could change or vanish.** | The NBA source breaks. | Accepted deliberately, and recorded (D1): the official alternative cannot see a season in progress, failing FR-014 outright. The source is one file behind a stable contract, so replacing the feed is a rewrite of one file, not of the library. |
| **Someone "simplifies" onto the official NBA feed later.** | The NBA silently becomes backtest-only. | D1 is written down with its evidence, and FR-014 exists precisely to forbid it. |
| **Someone "fixes" the season year by adding one.** | Every NBA season off by one, and the catalogue intersection with the odds vendor empties. | ESPN's `season.year` already means the year the season ends (D3). Stated in the research and pinned by a test. |
