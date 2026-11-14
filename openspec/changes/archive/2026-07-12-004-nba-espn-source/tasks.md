# Tasks

## 1. NBAStats, offline

- [x] 1.1 Create `src/sportsbet/datasets/_sources/_nba.py` with `NBAStats(BaseStatsSource)` and `name = 'nba'`. Free,
  needs no key, takes no constructor arguments. It is a league, not a sport, so it gets no package under
  `_basketball/` (T001)
- [x] 1.2 `index_items()` returns one item: the seasons index
  `https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons?limit=100`. Volatile, because a new season
  appears each year, and free (T002)
- [x] 1.3 `catalogue(payloads)` parses the season years out of the `$ref` URLs into
  `{'league': 'NBA', 'division': 1, 'year': <year>}`. No `+1` conversion, because ESPN's season year is already the year
  the season ends. The docstring says so (T003)
- [x] 1.4 `required_items(params)` returns one item per month, September of `year-1` through July of `year`, each a
  scoreboard call `?dates=YYYYMMDD-YYYYMMDD&limit=1000`. The docstring says why: the feed truncates at 1,000 events
  silently, a season is about 1,400, and a month peaks at 239 (T004)
- [x] 1.5 `to_snapshots(payloads)` applies the filter as an exclusion: drop `season.type == 1` (pre-season) and drop
  `competition.type.abbreviation == 'ALLSTAR'`. Keep everything else, including the playoffs (`RD16`, `QTR`, `SEMI`,
  `FINAL`) and the play-in. An unrecognised label is excluded, never admitted (FR-011) (T005)
- [x] 1.6 Read the tip-off from the event's `date` (ISO-8601, UTC, ends in `Z`). Read it, never infer it (T006)
- [x] 1.7 Take played from `competitions[0].status.type.completed`, per game. Never infer it from the season being over,
  because the completed 2023-24 season still contains 2 games that were never played (T007)
- [x] 1.8 Split a game into a pre-play snapshot (form) and a post-play snapshot (score and outcome). An unplayed game
  gets the pre-play snapshot only, so it is a fixture and never a training row (T008)
- [x] 1.9 Derive the outcome with the public `market_outcomes`: `home_win`, `away_win`. No draw, because a tie goes to
  overtime, and no totals, because the line moves per game (T009)
- [x] 1.10 Build the form features with the shape of `EuroLeagueStats`: per-team expanding and rolling(3) means over
  `points_for`, `points_against` and `wins`, shifted by one, on a frame with the fixtures already appended. Invent no
  basketball statistics from a score line (T010)
- [x] 1.11 Write `tests/datasets/test_nba.py` against a payload shaped like the real ESPN response: the catalogue comes
  from the feed and a season is named by the year it ends in with no `+1`, the UTC tip-off is read, the two-way outcome
  has no `draw` and no `over_` or `under_` column, the played and unplayed split, and form never sees its own game
  (T011)
- [x] 1.12 Regression test, first half of the trap: an all-star exhibition labelled
  `season.type: 2, slug: 'regular-season'` with competitors `Team Stars` and `World` is excluded, and no invented team
  appears among the teams (T012)
- [x] 1.13 Regression test, second half of the trap: a playoff game whose competition type is `RD16`, `SEMI` or
  `FINAL`, not `STD`, is included (T013)
- [x] 1.14 The pre-season is excluded, and an unrecognised competition label is excluded rather than admitted (FR-011)
  (T014)
- [x] 1.15 A postponed game in a finished season is a fixture, not a training row (T015)
- [x] 1.16 Guard test: `required_items` produces one item per month, each window spanning a single calendar month, so a
  later change that widens the window fails the suite (FR-012, SC-003) (T016)

## 2. The league reaches the user

- [x] 2.1 Export `NBAStats` from `sportsbet.datasets` (`src/sportsbet/datasets/__init__.py`) (T017)
- [x] 2.2 No registration point exists, and none was invented. `EuroLeagueStats` is not registered anywhere. The CLI
  loads a user configuration module that names a `DATALOADER_CLASS`, so the NBA is reachable from it as the EuroLeague
  is, through a subclass of the basketball dataloader, with no change. The GUI maps only
  `{'Soccer': SoccerDataLoader}` and has no basketball, because it cannot supply a paid odds key. That gap predates this
  feature and is reported (T018)
- [x] 2.3 Moved to change `verify-basketball-sources`, task 3. Partly done. The statistics were verified live: the
  2025-26 season gives 1326 games, exactly 30 teams, zero exhibitions, 4 postponed games left as fixtures, UTC tip-offs
  from 2025-10-21 to 2026-06-14, and the two-way market. `prepare(dry_run=True)` quotes the odds at 17,722 credits with
  zero spent. The free key holds 498 credits, so the alignment of `X`, `Y` and `O` is not verified end to end for the
  NBA (T019)

## 3. Reconcile the two rosters

- [x] 3.1 Moved to change `verify-basketball-sources`, task 2. Blocked. The NBA is out of season, so the vendor lists
  zero NBA events and there is no roster to pair against. The historical endpoint is paid-only (`401`, zero credits
  spent). The pairing is unverified. The vendor spells basketball clubs as city plus nickname, checked on the WNBA, as
  ESPN spells the NBA, so zero aliases is expected. Re-run in October or with a paid key (T020)
- [x] 3.2 No alias added, which is the correct outcome of a pairing that could not run. An alias added blind could
  attach one club's prices to another club's game silently (T021)

## 4. Docs and the gate

- [x] 4.1 Add the NBA to the source table in `docs/overview/user_guide/sources.md` (T022)
- [x] 4.2 Update `docs/overview/user_guide/dataloader.md`: basketball has a second league. Show
  `BasketballDataLoader(param_grid={'league': ['NBA']}, stats=NBAStats(), odds=OddsApi(key=...))`, and state that the
  NBA needs a paid odds key because no free basketball odds feed exists (T023)
- [x] 4.3 Add the `CHANGELOG.md` entry (T024)
- [x] 4.4 Re-run the public-API docs audit: every public name, `NBAStats` included, still has a runnable example (T025)
- [x] 4.5 Confirm SC-006 (no new runtime dependency), SC-007 (`git diff` clean on every listed file, and no existing
  test edited) and SC-010 (no NBA data committed) (T026)
