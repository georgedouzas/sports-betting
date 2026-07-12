---

description: "Task list for Basketball, starting with the EuroLeague"
---

# Tasks: Basketball, starting with the EuroLeague

**Input**: Design documents from `/specs/003-basketball-euroleague/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [contracts/euroleague.md](contracts/euroleague.md)

**Tests**: Required. Two bugs were found while planning (the totals line, the empty-odds error) and both get regression tests.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelisable (different files, no dependency on an incomplete task)
- **[Story]**: US1–US4 from [spec.md](spec.md)

## Conventions

- No explanatory inline comments. Short Google-style docstrings.
- Flat test files under `tests/datasets/`.
- Never import a private name from a private module in a test.
- No executable doctests on network-touching classes.
- Line length 120, `skip-string-normalization`.

---

## Phase 0: Hoist the engine (Blocking)

**Purpose**: `SoccerDataLoader` is 282 lines and the **only** soccer-specific thing in it is two class names. Everything else — the store plumbing, the catalogue intersection, `prepare`, the cost estimate, the schedule, the reconciliation — is the engine. Basketball must inherit it, not copy it.

**Goal**: every existing test passes **unmodified** and the soccer equivalence gate still matches. If a test needs editing, the hoist changed behaviour and is wrong.

- [X] T001 Create `src/sportsbet/datasets/_base/_sourced.py` with `SourcedDataLoader(BaseDataLoader)`, holding everything sport-agnostic moved out of `SoccerDataLoader`: `__init__`, `_resolved`, `sources`, `_authorize`, `_unique`, `_catalogue`, `_params`, `_all_params`, `_schedule`, `_items`, `_report`, `_unavailable`, `prepare`, `_snapshots`, `_derive`
- [X] T002 Give `SourcedDataLoader` the class attributes `DEFAULT_STATS` and `DEFAULT_ODDS`, so a sport is its defaults and nothing else
- [X] T003 Reduce `src/sportsbet/datasets/_soccer/_dataloader.py` to a class name and two defaults (`FootballDataStats`, `FootballDataOdds`)
- [X] T004 Move `market_outcomes` from `_soccer/_utils.py` to a sport-neutral `src/sportsbet/datasets/_utils.py`, keeping `sportsbet.datasets.market_outcomes` working — it now serves two sports
- [X] T005 Run the full suite and the soccer equivalence gate. **Zero test edits allowed.**

**Checkpoint**: `pdm run tests` green with no test touched, and `pdm run pytest tests/datasets/test_equivalence.py -m network` still passes.

---

## Phase 1: `EuroLeagueStats` (US1 — P1)

**Goal**: a free, key-less source producing long snapshots from the EuroLeague's official API.

- [X] T006 [US1] Create `src/sportsbet/datasets/_sources/_euroleague.py`: `index_items()` → `/v2/competitions/E/seasons` (free); `catalogue(payloads)` → the seasons the API publishes, with `year` as the year the season **ends** (`E2024` → 2025), never a fabricated range
- [X] T007 [US1] `required_items(params)` → **one** item per season (`/v2/competitions/E/seasons/E{year}/games`), because the endpoint returns the whole season in one response — verified: 330 games
- [X] T008 [US1] **Changed to JSON.** `bandit` flags `xml.etree.ElementTree` as entity-expansion-vulnerable, and the fix (`defusedxml`) would be a new dependency. The v2 endpoint returns JSON, which needs nothing — and gives better team names. See research D4.
- [X] T009 [US1] **Take `utcDate`.** The API does publish every game in its own head-office zone whatever the country — an Istanbul game reads `18:30` and tips off at 20:30 local — but v2 also gives UTC outright, so the conversion is not a guess at all. The CET finding is what proves reading the other field would have been wrong. See research D1.
- [X] T010 [US1] Build the form features: per-team expanding and rolling(3) means over **points for/against**, shifted by one, on a frame with the fixtures already appended. Port the *shape* of the soccer version; invent no basketball statistics from a score line.
- [X] T011 [US1] Split a game into a pre-play snapshot (form) and a post-play snapshot (score + outcome), using `<played>` so an unplayed game becomes a fixture and never a training row
- [X] T012 [US1] Derive the outcome with `market_outcomes`: `home_win`, `away_win`. **No draw** (a tie goes to overtime) and **no totals** (the line moves per game — research D2)
- [X] T013 [P] [US1] Write `tests/datasets/test_euroleague.py` against a payload shaped like the real response: the UTC tip-off, the played/unplayed split, the two-way outcome, form that never sees its own game

**Checkpoint**: the source produces snapshots offline, with no network.

---

## Phase 2: `BasketballDataLoader` and the odds (US1, US2 — P1)

- [X] T014 [US1] Create `src/sportsbet/datasets/_basketball/_dataloader.py`: `BasketballDataLoader(SourcedDataLoader)` with `DEFAULT_STATS = EuroLeagueStats` and `DEFAULT_ODDS = None`. **If it needs more than that, the abstraction is wrong** — fix the base, not this.
- [X] T015 [US1] Extend `OddsApi.LEAGUES_MAPPING` with `basketball_euroleague` → `('Euroleague', 1)` and the other `basketball_*` keys
- [X] T016 [US1] Export `BasketballDataLoader` and `EuroLeagueStats` from `sportsbet.datasets`, and add basketball to the CLI/GUI dataloader mapping so all three surfaces gain the sport at once
- [X] T017 [P] [US2] Write `tests/datasets/test_basketball.py`: `Y` carries a home win and an away win and **no draw**, and a bettor's probabilities for the two sum to one — while soccer keeps its draw
- [X] T018 [US1] Verify end to end against the live API: a EuroLeague season prepares and extracts, `X`/`Y`/`O` aligned

**Checkpoint**: the library has a second sport.

---

## Phase 3: Reconcile the two rosters (US3 — P1)

**Purpose**: basketball **always** mixes two sources — there is no single feed with both. So this is the normal path, not an edge case.

- [X] T019 [US3] Verified: **17 of the 18 clubs pair with no aliases**, sponsor names and all — `Maccabi Tel Aviv` → `Maccabi Playtika Tel Aviv`, `Partizan` → `Partizan Mozzart Bet Belgrade`, `ASVEL` → `LDLC ASVEL Villeurbanne`.
- [X] T020 [US3] **One** alias added: `Olimpia Milano` → `EA7 Emporio Armani Milan`. The vendor uses the club's historic name and the feed uses its sponsor's; they share nothing but the city, so nothing can pair them and nothing should try. The resolver correctly refused and suggested it.
- [X] T021 [P] [US3] Test that an unplaceable club raises and names itself, rather than silently dropping its game

**Checkpoint**: the two sources reconcile, or fail loudly.

---

## Phase 4: The honest failure, the gate, the docs (US4)

- [X] T022 [US4] Reproduce the empty-odds failure, then make it a clear error: a dataloader with no odds source has no markets and nothing to predict. It must say that, not `expected series 'event_status' to have type string[pyarrow], got object`.
- [X] T023 [P] [US4] Test both shapes of it: no odds **source**, and an odds source carrying no **markets**
- [ ] T024 Freeze a EuroLeague fingerprint alongside the soccer one. **Deferred**: the EuroLeague API is volatile by design (a season is re-read on every prepare), so a fingerprint of it would drift as the season progresses. It is worth doing for a *completed* season only.
- [X] T025 [P] Update `docs/overview/user_guide/dataloader.md`: it says the library is soccer-only. Add the sport, the source, the two-way market, and the honest statement that basketball needs a paid odds key.
- [X] T026 [P] Update the README and `docs/overview/user_guide/index.md`
- [X] T027 Confirm no new runtime dependency (SC-010) and that no basketball data is published (SC-007)

---

## Dependencies

- **Phase 0 blocks everything.** Basketball inherits the engine; it must exist first.
- Phase 1 → Phase 2 → Phase 3. Phase 4 last.
- **Soccer must not move.** Its equivalence gate is checked at every phase boundary (SC-009).

## Gate at every phase boundary

```bash
pdm run formatting
pdm run checks
pdm run tests
```
