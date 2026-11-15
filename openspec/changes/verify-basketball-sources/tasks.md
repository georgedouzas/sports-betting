# Tasks

Each group ends green on `pdm run formatting`, `pdm run checks`, `pdm run docs build` and `pdm run tests`, read from
the nox session summary line.

## 1. EuroLeague sample (from change 003, task 5.3)

- [x] 1.1 Record a small set of games from the completed 2025-26 EuroLeague season into `tests/sources/samples/`, as
  the soccer sample is recorded.
- [x] 1.2 Add a test to `tests/sources/stats/test_euroleague.py` that transforms the sample offline and checks the
  moments, the teams and the two-way market. Verify that it passes with the network off.

## 2. NBA roster pairing (from change 004, task 3.1)

- [x] 2.1 Read the current `OddsApi` NBA event list, which costs no credits, and pair it with the ESPN roster of the
  2026-27 season through `resolve_odds`. Record how many teams pair and which do not.
- [x] 2.2 Add an alias for each club the pairing cannot place, with a test, or record that zero aliases are needed.

Result of 2.1, run on 2026-11-15. The `OddsApi` event list held 9 games and 18 teams. The ESPN 2026-27 roster held 30
teams and a `TBD` placeholder. 17 of the 18 teams paired exactly. `Los Angeles Clippers` did not pair with ESPN's
`LA Clippers`, so 2.2 adds that alias to `ALIASES`. The other 12 teams had no game in the event list that day and were
not checked live.

## 2b. ESPN defects found by 2.1

- [x] 2.3 ESPN answers `400` to the `YYYYMMDD-YYYYMMDD` range `NBAStats` asked for, so no NBA season could be read.
  Ask for each month as `YYYYMM`, which returns the whole month, with a test of the URL. Item keys are unchanged.
- [x] 2.4 ESPN lists games whose teams are not known yet as `TBD` against `TBD`, under negative team ids. 72 such
  games were in the 2026-27 season. Leave them out, with a test, so `TBD` never becomes a team or a fixture.

## 3. NBA alignment (from change 004, task 2.3)

- [ ] 3.1 Quote the odds of a window of the 2026-27 season with `prepare(dry_run=True)`, and choose a window the
  available credits cover.
- [ ] 3.2 Extract training data over that window and check that `X`, `Y` and `O` share their index, that every game
  with odds has its outcome, and that the home and away prices belong to the right teams. Record the result in this
  file.

Result of 3.1, run on 2026-11-15. `prepare(dry_run=True)` no longer exists, so the window was priced from the items
`OddsApi.list_required_items` returns. With `h2h`, one region and the pre-play moment only, 11 and 12 November 2026
held 16 games at 10 tip-off times, so 10 historical requests at 10 credits each and one live request. The free key
answered the first historical request with `401 Unauthorized`, since the vendor sells that endpoint on paid plans only.
One credit was spent, on the live request. 3.1 and 3.2 stay open until a paid key is available.

The `401` also showed that a failed `OddsApi` request raises an error whose message holds the full URL, key included,
so a traceback or a log line leaks the key. That defect gets its own change.
