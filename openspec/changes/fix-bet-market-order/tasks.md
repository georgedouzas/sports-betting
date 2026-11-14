# Tasks

Each group ends green on `pdm run formatting`, `pdm run checks`, `pdm run docs build` and `pdm run tests`, read from
the nox session summary line.

## 1. Column order

- [ ] 1.1 Write a failing test in `tests/evaluation/` for a bettor whose `COMPLEMENTARY_EVENTS` lists the over/under
  group before the result group, while `betting_markets_` lists `home_win` first. Assert that the column of `bet` for
  each market is true exactly where that market is the value bet.
- [ ] 1.2 In `src/sportsbet/evaluation/_base.py`, reorder the joined frame of `bet` to `betting_markets_` before
  returning it. Verify that the test of 1.1 passes and that `score` on the same data changes accordingly.

## 2. Market outside every group

- [ ] 2.1 Write a failing test: a bettor with `COMPLEMENTARY_EVENTS` that leaves out a selected market raises
  `ValueError` at `fit`, and the message names the market.
- [ ] 2.2 In `fit`, check that every selected market falls in a group, building the message in a variable. Verify the
  test of 2.1.

## 3. Record

- [ ] 3.1 Commit as `fix(evaluation): ...` so the changelog carries the corrected backtest results.
