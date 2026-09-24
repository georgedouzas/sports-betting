# Quickstart & Validation Guide: In-Play (Live) Betting Support

Runnable scenarios that prove the feature works end-to-end. All scenarios use the offline
`DummySoccerDataLoader` (no network) so they double as doctests under `--doctest-modules`.
Contracts: [datasets-api](./contracts/datasets-api.md), [evaluation-api](./contracts/evaluation-api.md),
[cli](./contracts/cli.md). Data shapes: [data-model](./data-model.md).

## Prerequisites

```bash
# From repo root, on branch 001-in-play-betting (git branch: development)
pdm install -G :all          # or: uv sync
```

## Setup / gates

```bash
pdm run formatting     # black, docformatter
pdm run checks         # ruff, mypy, interrogate, bandit, pip-audit
pdm run tests          # pytest: branch coverage, doctest-modules, randomized
```

Feature is complete only when all three pass with no new suppressions (Constitution IV) and
coverage does not regress (SC-006).

## Scenario 1 — Moment-aware extraction (US1 / SC-001)

```python
import pandas as pd
from sportsbet.datasets import DummySoccerDataLoader

loader = DummySoccerDataLoader(param_grid={'league': ['England']})

# Post-match (full-time) target
X, Y, O = loader.extract_train_data(odds_type='market_average', target_event_status='postplay')

# In-play target at 60 minutes: features must exclude later snapshots
X60, Y60, O60 = loader.extract_train_data(
    target_event_status='inplay', target_event_time=pd.Timedelta('60min'))
assert all('90min' not in c for c in X60.columns)   # no post-target leakage
```

**Expected**: three aligned tables; `X60` contains no information dated after 60 minutes.

## Scenario 2 — Training/fixtures column correspondence (US1/US3 / SC-002)

```python
X_train, Y_train, O_train = loader.extract_train_data(odds_type='market_average')
X_fix, Y_fix, O_fix = loader.extract_fixtures_data()
assert Y_fix is None
pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
pd.testing.assert_index_equal(O_train.columns, O_fix.columns)
```

**Expected**: identical feature/odds columns; fixtures carry no targets.

## Scenario 3 — Schema validation rejects bad data (US1 / SC-003)

```python
import pandera.pandas as pa, pytest
bad = loader  # construct a loader whose stats contain an invalid event_status
# extract_train_data MUST raise before any modelling
with pytest.raises(pa.errors.SchemaError):
    ...  # see tests/datasets/base/test_dataloader.py for the concrete fixture
```

**Expected**: invalid `event_status`/type/duplicate snapshot → `SchemaError` naming the
field; no `X`/`Y`/`O` produced.

## Scenario 4 — Backtest and value bets (US2 / SC-004)

```python
from sklearn.dummy import DummyClassifier
from sportsbet.evaluation import ClassifierBettor, backtest

X, Y, O = loader.extract_train_data(odds_type='market_average')
bettor = ClassifierBettor(DummyClassifier())
results = backtest(bettor, X, Y, O)          # per-period performance

bettor.fit(X, Y)
X_fix, _, O_fix = loader.extract_fixtures_data()
selections = bettor.bet(X_fix, O_fix)        # value-bet selections for fixtures
```

**Expected**: `backtest` returns performance results; `bet` returns selections aligned to
fixtures — using the same bettor for pre-match and in-play targets (FR-016).

## Scenario 5 — Selection interface & persistence (US3 / SC-007)

```python
DummySoccerDataLoader.get_all_params()   # discover selectable params, no download
loader.get_odds_types()                  # discover odds types
loader.save('loader.pkl')
from sportsbet.datasets import load_dataloader
reloaded = load_dataloader('loader.pkl')
Xf1, _, Of1 = loader.extract_fixtures_data()
Xf2, _, Of2 = reloaded.extract_fixtures_data()
pd.testing.assert_index_equal(Xf1.columns, Xf2.columns)
```

**Expected**: discovery works offline; reloaded loader reproduces the column structure.

## Scenario 6 — CLI / GUI parity (US4 / SC-005)

```bash
# Using a config that sets DATALOADER_CLASS=DummySoccerDataLoader and a TARGET_EVENT_STATUS
sportsbet ...   # run extraction + backtest via CLI; compare to Scenario 4 API results
```

**Expected**: CLI (and GUI) produce equivalent extraction, backtest, and value-bet results
to the API for the same config.

## Scenario 7 — Reinforcement is not a valid extraction mode (FR-011)

```python
import pytest
with pytest.raises(ValueError):
    loader.extract_train_data(learning_type='reinforcement')
```

**Expected**: `'reinforcement'` is rejected as an invalid `learning_type` (only `supervised`
and `unsupervised` are valid). RL is a separate future method (`make_env()`), designed but
not implemented this feature — see research.md R1.

## Definition of done (validation checklist)

- [X] Scenarios 1–7 pass against the sample dataloader, offline.
- [X] All docstring examples run under `--doctest-modules`.
- [X] User guide (`dataloader.md`, `bettor.md`) and examples updated to the new model.
- [X] CHANGELOG entry added.
- [X] `formatting`, `checks`, `tests` gates green; coverage not regressed.
