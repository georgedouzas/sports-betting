# Tasks

Each group ends green on `pdm run formatting`, `pdm run checks`, `pdm run docs build` and `pdm run tests`, read from
the nox session summary line.

## 1. EuroLeague sample (from change 003, task 5.3)

- [ ] 1.1 Record a small set of games from the completed 2025-26 EuroLeague season into `tests/sources/samples/`, as
  the soccer sample is recorded.
- [ ] 1.2 Add a test to `tests/sources/stats/test_euroleague.py` that transforms the sample offline and checks the
  moments, the teams and the two-way market. Verify that it passes with the network off.

## 2. NBA roster pairing (from change 004, task 3.1)

- [ ] 2.1 Read the current `OddsApi` NBA event list, which costs no credits, and pair it with the ESPN roster of the
  2026-27 season through `resolve_odds`. Record how many teams pair and which do not.
- [ ] 2.2 Add an alias for each club the pairing cannot place, with a test, or record that zero aliases are needed.

## 3. NBA alignment (from change 004, task 2.3)

- [ ] 3.1 Quote the odds of a window of the 2026-27 season with `prepare(dry_run=True)`, and choose a window the
  available credits cover.
- [ ] 3.2 Extract training data over that window and check that `X`, `Y` and `O` share their index, that every game
  with odds has its outcome, and that the home and away prices belong to the right teams. Record the result in this
  file.
