# Contract: the CLI configuration

The CLI reads a **Python module**. That module hands it a **dataloader that has already been built**.

## Before, and why it could not work

```python
DATALOADER_CLASS = SoccerDataLoader          # a class
PARAM_GRID = {'league': ['England'], 'year': [2025]}
```

The CLI then did `DATALOADER_CLASS(PARAM_GRID)` — constructing with **no sources**.

A class cannot carry a source. A source is what carries a credential. So this contract could express exactly one
configuration in the whole library: **soccer, with the free feed**. Basketball raises (`BasketballDataLoader` has no free
odds default), and `OddsApi` is unreachable for every sport because a key has nowhere to live.

## After

```python
import os

from sportsbet.datasets import BasketballDataLoader, NBAStats, OddsApi
from sportsbet.evaluation import OddsComparisonBettor

DATALOADER = BasketballDataLoader(
    param_grid={'league': ['NBA'], 'year': [2026]},
    stats=NBAStats(),
    odds=OddsApi(key=os.environ['ODDS_API_KEY']),
)
BETTOR = OddsComparisonBettor(alpha=0.03)
```

One name, `DATALOADER`, holding a dataloader you configured yourself. Anything you can build in Python, the CLI can now
run — every sport, every source, every credential.

| Variable | Required | Meaning |
| --- | --- | --- |
| `DATALOADER` | **yes** | A configured `BaseDataLoader`. Rejected if it is a class, or is not one. |
| `BETTOR` | for `backtest` / `bet` | A configured `BaseBettor`. |
| `ODDS_TYPE`, `DROP_NA_THRES`, `TARGET_EVENT_STATUS`, `TARGET_EVENT_TIME`, `CV`, `N_JOBS`, `VERBOSE` | no | As before. |

`PARAM_GRID` is **gone**. The selection lives inside the dataloader, where it always belonged — it is a constructor
argument, not a separate fact about the run.

## The credential

The config is Python, so the key is read from the environment:

```python
odds=OddsApi(key=os.environ['ODDS_API_KEY'])
```

It is never written into the file, so the file can be committed, shared, and reviewed. This is the reason the config is
a Python module rather than YAML: a YAML config would have to hold the key, or invent a syntax for not holding it.

## `params` must work before you have chosen anything

`sportsbet dataloader params` is how you learn **what to select**, so it cannot require you to have already selected it:

```python
DATALOADER = SoccerDataLoader()      # no param_grid yet — valid
```

It asks the **source** what exists (`available_params`), which is free and downloads nothing. `param_grid=None` is a
valid config and must stay one.

## An old config must say what to change

A config with `DATALOADER_CLASS` fails with a message naming the replacement — not with an `AttributeError`, and not by
silently doing nothing. This is a breaking change and it must behave like one.
