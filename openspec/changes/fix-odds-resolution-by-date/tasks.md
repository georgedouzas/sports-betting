# Tasks

Each group ends green on `pdm run formatting`, `pdm run checks`, `pdm run docs build` and `pdm run tests`, read from
the nox session summary line.

## 1. Kick-off in the join

- [ ] 1.1 Write a failing test in `tests/sources/`: statistics hold `Olympiacos` at home to `Real Madrid` on two dates
  in one season, and the odds hold one row for each date with different prices. Assert that each price resolves to its
  own date.
- [ ] 1.2 Write a failing test: an odds row 3 days away from any statistics kick-off of its pairing is dropped.
- [ ] 1.3 In `src/sportsbet/sources/_resolver.py`, replace the merge on the match columns with a nearest join on the
  kick-off within 36 hours, per pairing. Keep the doctest of `resolve_odds` passing and add the repeated pairing to it.
  Verify the tests of 1.1 and 1.2.

## 2. Record

- [ ] 2.1 Commit as `fix(sources): ...`.
