# Quickstart: validating basketball

How to prove the sport actually works. Every scenario maps to acceptance criteria in [spec.md](spec.md).

## Prerequisites

```bash
pdm install -dG tests
```

No test may touch the network. The socket guard in `tests/conftest.py` fails the suite if any test opens a connection —
that guard *is* the proof of SC-008.

## Scenario 1 — The hoist changed nothing (Phase 0)

Before basketball exists, the sport-agnostic half of the soccer loader moves onto the base.

```bash
pdm run tests
pdm run pytest tests/datasets/test_equivalence.py -m network --no-cov
```

**Expects**: every existing test passes **unmodified**, and the soccer fingerprint still matches. If a test needed
editing, the hoist changed behaviour and is wrong — that is the whole point of doing it first.

## Scenario 2 — A EuroLeague dataset (US1, SC-001)

```python
from sportsbet.datasets import BasketballDataLoader, EuroLeagueStats, OddsApi

dataloader = BasketballDataLoader(
    param_grid={'league': ['Euroleague'], 'division': [1], 'year': [2025]},
    stats=EuroLeagueStats(),
    odds=OddsApi(key='...', markets=['h2h']),
)
dataloader.prepare()
X, Y, O = dataloader.extract_train_data(odds_type='pinnacle')
X_fix, _, O_fix = dataloader.extract_fixtures_data()
```

**Expects**: `X`, `Y` and `O` share the same rows and index. A season is **one** request to the statistics API.

## Scenario 3 — A game that cannot be drawn (US2, SC-003)

```python
list(Y.columns)
# ['home_win__postplay__0min', 'away_win__postplay__0min']    <- no draw
```

```bash
pdm tests -k complementary
```

**Expects**: no `draw` column, and a bettor's predicted probabilities for a home win and an away win sum to **one** —
which they do **not** in soccer, where a draw is possible. Nothing is configured to make this happen: it is derived from
the columns the data carries.

## Scenario 4 — The two rosters reconcile (US3, SC-004)

```python
dataloader.reconciliation_        # matched, unmatched_rate, unmatched_stats, unmatched_odds
```

**Expects**: the real EuroLeague rosters — `PANATHINAIKOS AKTOR ATHENS`, `MACCABI PLAYTIKA TEL AVIV` — pair with the odds
vendor's names using **no hand-written aliases**, or every exception is named and justified.

**Fails loudly when**: a club cannot be placed. It raises `UnmatchedError` and names the club, rather than emitting a game
with no odds — which would look like a slightly smaller dataset and a backtest that is clean and wrong.

## Scenario 5 — Know before you pay (US4, SC-006)

```python
report = dataloader.prepare(dry_run=True)
report.estimated_cost        # {'odds_api': ...}
```

**Expects**: an **exact** cost, and **zero** credits spent to obtain it. The statistics are free and they say when every
game tips off, so the snapshots the odds source would need are counted without asking the vendor for any of them.

## Scenario 6 — No odds, no dataset — but say so (US4, D7)

```python
BasketballDataLoader(param_grid=...).extract_train_data()
```

**Expects**: a clear error saying there are no markets to predict. **Not** `expected series 'event_status' to have type
string[pyarrow], got object`, which is what it does today and which tells the user nothing about the fact that they forgot
the odds.

## Scenario 7 — The kick-off is right (D1, the one that would fail silently)

```python
X.index[0]     # the tip-off, in UTC
```

**Expects**: a game the API publishes at `18:30` in **Istanbul** is `17:30 UTC` — because the API publishes **CET** for
every venue, and Istanbul is UTC+3, so it tips off at 20:30 local.

**Why this scenario exists**: reading the API's time as UTC, or as the venue's local time, is wrong and **nothing would
say so**. Every odds snapshot the user buys would be off by one to two hours. This is the second feed to do this —
football-data.co.uk publishes every league in UK time — so it is checked, not assumed.

## Scenario 8 — Soccer is untouched (SC-009)

```bash
pdm run pytest tests/datasets/test_equivalence.py -m network --no-cov
```

**Expects**: the soccer fingerprint still matches, exactly. A second sport must not move the first one.

## Scenario 9 — Nothing is published (SC-007, SC-010)

```bash
git ls-tree -r --name-only HEAD | grep -i euroleague     # code and tests only, no data
git diff --stat -- pyproject.toml                        # no new runtime dependency
```

## Full gate

```bash
pdm run formatting
pdm run checks
pdm run tests
```

All three must be green before any phase is done.
