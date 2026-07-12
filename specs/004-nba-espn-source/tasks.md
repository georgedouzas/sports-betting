---

description: "Task list for the NBA, as a second basketball league"
---

# Tasks: The NBA, as a second basketball league

**Input**: Design documents from `/specs/004-nba-espn-source/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [contracts/nba.md](contracts/nba.md)

**Tests**: Required. Two traps were found while researching (the all-star exhibition filed under `regular-season`, the
playoffs filed under something other than `STD`) and **each produces a plausible, silent, wrong answer**. Both get a
regression test, and the request window gets a guard test.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelisable (different files, no dependency on an incomplete task)
- **[Story]**: US1–US5 from [spec.md](spec.md)

## The acceptance criterion

**SC-007 is the point of this feature.** The NBA is a *new file*. These files must show a **zero diff**:

```text
src/sportsbet/datasets/_fetch.py                   src/sportsbet/datasets/_base/_sourced.py
src/sportsbet/datasets/_store.py                   src/sportsbet/datasets/_basketball/_dataloader.py
src/sportsbet/datasets/_sources/_base.py           src/sportsbet/datasets/_sources/_odds_api.py
src/sportsbet/datasets/_sources/_euroleague.py     src/sportsbet/datasets/_sources/_football_data.py
```

**If an existing test must be edited to stay green, STOP AND REPORT.** Do not edit the test. A test that needs changing
means the abstraction leaked, and that is a finding, not a chore.

## Conventions

- Sources never fetch. `index_items` / `catalogue` / `required_items` / `to_snapshots` are pure; the store fetches.
- No test touches the network (socket guard). No executable doctest on a network-touching class.
- No new runtime dependency. Ship no data.
- No explanatory inline comments. Short Google-style docstrings.
- Flat test files under `tests/datasets/`. Never import a private name from a private module in a test.
- Line length 120, `skip-string-normalization`.

---

## Phase 1: `NBAStats`, offline (US1, US2, US3, US4 — all P1)

**Purpose**: the whole feature. One file, plus the tests that prove it is not quietly wrong.

**Goal**: a free, key-less source that turns a recorded ESPN payload into long snapshots, with no network anywhere.

- [X] T001 [US1] Create `src/sportsbet/datasets/_sources/_nba.py` with `NBAStats(BaseStatsSource)`, `name = 'nba'`. Free, key-less, **no constructor arguments**. It is a league, not a sport — no package under `_basketball/`.
- [X] T002 [US1] `index_items()` → **one** item: the seasons index `https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons?limit=100`. Volatile (a new season appears each year), free.
- [X] T003 [US1] `catalogue(payloads)` → parse the season years out of the `$ref` URLs → `{'league': 'NBA', 'division': 1, 'year': <year>}`. **NO `+1` CONVERSION.** ESPN's season year is *already* the year the season ends, unlike the EuroLeague's `E2024`. The docstring must say so, because this is exactly what a later reader "fixes" by analogy (research D3).
- [X] T004 [US3] `required_items(params)` → **one item per month**, September of `year-1` through July of `year`, each a scoreboard call `?dates=YYYYMMDD-YYYYMMDD&limit=1000`. The docstring must say **why** monthly: the feed truncates at 1,000 events **silently**, a season is ~1,400, and a month peaks at 239 (research D4). Not "it is monthly" — *why* it is monthly.
- [X] T005 [US2] `to_snapshots(payloads)`: the filter, written as an **exclusion** — drop `season.type == 1` (pre-season) and drop `competition.type.abbreviation == 'ALLSTAR'`. Keep everything else, **including** the playoffs (`RD16`/`QTR`/`SEMI`/`FINAL`) and the play-in. An **unrecognised** label must be excluded, never admitted (FR-011): a missing game is a visible bug, an invented team is silent corruption.
- [X] T006 [US1] Read the tip-off from the event's `date` (ISO-8601, UTC, ends in `Z`). **Read, never inferred** — this is the first feed of the three that gives it straight (research D6).
- [X] T007 [US4] Take *played* from `competitions[0].status.type.completed`, **per game**. Never infer it from the season being over: a **completed** 2023-24 season still contains 2 games that were never played (research D7).
- [X] T008 [US1] Split a game into a pre-play snapshot (form) and a post-play snapshot (score + outcome). An unplayed game gets the pre-play snapshot **only**, so it is a fixture and never a training row.
- [X] T009 [US1] Derive the outcome with the existing public `market_outcomes`: `home_win`, `away_win`. **No draw** (a tie goes to overtime) and **no totals** (the line moves per game — settled for the EuroLeague, not re-litigated).
- [X] T010 [US1] Build the form features by porting the **shape** of `EuroLeagueStats._form`: per-team expanding and rolling(3) means over `points_for` / `points_against` / `wins`, **shifted by one**, on a frame with the fixtures **already appended**. Invent no basketball statistics from a score line.
- [X] T011 [P] [US1] Write `tests/datasets/test_nba.py` against a payload shaped like the real ESPN response: the catalogue comes from the feed and a season is named by the year it ends in **with no `+1`**; the UTC tip-off is read; the two-way outcome has no `draw` and no `over_`/`under_` column; the played/unplayed split; form never sees its own game.
- [X] T012 [P] [US2] **Regression, trap half 1**: an all-star exhibition labelled `season.type: 2, slug: 'regular-season'` whose competitors are `Team Stars` / `World` is **EXCLUDED**, and no invented team appears among the teams. *(Filtering on season type alone admits it.)*
- [X] T013 [P] [US2] **Regression, trap half 2**: a playoff game whose competition type is `RD16` / `SEMI` / `FINAL` — **not** `STD` — is **INCLUDED**. *(Filtering on `STD` alone drops 85 real games a season, and looks entirely plausible.)*
- [X] T014 [P] [US2] The pre-season is excluded, and an **unrecognised** competition label is excluded rather than admitted (FR-011).
- [X] T015 [P] [US4] A postponed game in a **finished** season is a fixture, not a training row.
- [X] T016 [P] [US3] **Guard test**: `required_items` produces one item per month, each window spanning a single calendar month. This pins the window below the feed's cap, so a later "optimisation" that widens it to save requests **fails the suite** instead of silently truncating a season (FR-012, SC-003).

**Checkpoint**: the source produces snapshots offline, with no network. Full gate green.

---

## Phase 2: The league reaches the user (US1 — P1; FR-017)

- [X] T017 [US1] Export `NBAStats` from `sportsbet.datasets` (`src/sportsbet/datasets/__init__.py`).
- [X] T018 [US1] **No registration point exists, and none was invented.** `EuroLeagueStats` is not registered anywhere: the CLI loads a user config module that names a `DATALOADER_CLASS`, so the NBA is reachable from it exactly as the EuroLeague is — by subclassing the basketball dataloader — and needs no change. The **GUI** maps only `{'Soccer': SoccerDataLoader}`: it has **no basketball at all**, because it cannot supply a paid odds key. That is a pre-existing gap from feature 003, not one this feature introduces, and closing it means solving the key problem in the GUI. Reported, not papered over.
- [~] T019 [US1] **Statistics verified live; the odds half could not be bought.** Against the live feed, the 2025-26 season gives **1326 games, exactly 30 teams, zero exhibitions**, 4 postponed games correctly left as fixtures, UTC tip-offs from 2025-10-21 to 2026-06-14, and the two-way market — every number matching research D5. `prepare(dry_run=True)` quotes the odds at **17,722 credits for zero spent**, which is the two-stage plan working exactly as designed. But 17,722 is what an NBA season's time-stamped odds *cost*, and the free-tier key holds **498**, so `X`/`Y`/`O` alignment is **not** verified end to end. It is the same code path the EuroLeague already exercises; it is still unverified for the NBA, and is not claimed.

**Checkpoint**: `BasketballDataLoader(stats=NBAStats(), odds=OddsApi(key=...))` produces an NBA dataset.

---

## Phase 3: Reconcile the two rosters (US5 — P2)

**Purpose**: basketball **always** mixes two sources — no single feed carries both games and odds — so this is the normal
path, not an edge case. An unplaceable club means a game with no price, which means a backtest that is confidently wrong.

- [ ] T020 [US5] **BLOCKED, and not faked.** The NBA is out of season, so the vendor lists **zero** NBA events — there is no roster to pair against. Its **historical** endpoint, which would have one, is **paid-only** (`401`, zero credits spent). The pairing is therefore **unverified**. Established instead, live and free: the vendor spells basketball clubs `City Nickname` (checked on the WNBA), exactly as ESPN spells the NBA — so zero aliases is *expected*. Expected is not verified. **Re-run in October, or with a paid key.** See research D8.
- [X] T021 [US5] **No alias added**, which is the correct outcome of a pairing that could not be run: adding one blind would attach one club's prices to another club's game and say nothing about it.

**Checkpoint**: the two sources reconcile, or fail loudly and name the club.

---

## Phase 4: Docs and the gate

- [X] T022 [P] Add the NBA to the source table in `docs/overview/user_guide/sources.md`.
- [X] T023 [P] Update `docs/overview/user_guide/dataloader.md`: basketball now has a **second league**. Show `BasketballDataLoader(param_grid={'league': ['NBA']}, stats=NBAStats(), odds=OddsApi(key=...))`, and say plainly that the NBA needs a paid odds key because no free basketball odds feed exists.
- [X] T024 [P] Add the `CHANGELOG.md` entry.
- [X] T025 Re-run the public-API docs audit: **every** public name must still have a runnable example, `NBAStats` included.
- [X] T026 Confirm **SC-006** (no new runtime dependency), **SC-007** (`git diff` clean on every file listed above, and **no existing test edited**) and **SC-010** (no NBA data committed).

---

## Dependencies

- **Phase 1 is the feature.** Phases 2–4 are short and follow it.
- T001 → T002–T010 (the same file). T011–T016 are **[P]** with each other once the source exists.
- Phase 2 → Phase 3 (reconciliation needs a working dataloader) → Phase 4.
- **Soccer and the EuroLeague must not move.** Their tests are checked at every phase boundary (SC-009).

## Gate at every phase boundary

```bash
pdm run formatting
pdm run checks
pdm run tests
```
