# Tasks

Each group ends green on `pdm run formatting`, `pdm run checks`, `pdm run docs build` and `pdm run tests`, read from
the nox session summary line.

## 1. Adjusted goals

- [ ] 1.1 Write a failing test in `tests/sources/`: a team with 2 goals, 1 red card, 2 yellow cards, 5 shots on target,
  10 shots and 4 corners in its first match has `adj_goals_for` of 3.84 in that match, read through the `_avg` feature
  of its second match.
- [ ] 1.2 In `src/sportsbet/sources/_common/_football_data.py`, replace the second shots on target term of
  `adj_goals_for` and `adj_goals_against` with the shots. Verify the test of 1.1.

## 2. Spelling

- [ ] 2.1 Rename `fouls_commited` to `fouls_committed` in the feature filter and `hit_woodork` to `hit_woodwork` in the
  column mapping and the schema. Verify that grep finds neither misspelling under `src/`.

## 3. Record

- [ ] 3.1 Commit as `fix(sources): ...` so the changelog says the football-data features changed.
