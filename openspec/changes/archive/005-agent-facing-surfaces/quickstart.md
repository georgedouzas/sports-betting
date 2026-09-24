# Quickstart: proving the surfaces reach the whole library

## 1. The test run excludes nothing

```bash
grep -n "ignore" pyproject.toml     # no --ignore=src/sportsbet/gui
pdm run tests
```

Before this feature, a quarter of the package was skipped by name. **That is the single clearest measure of the
change** (SC-004).

```bash
pdm run checks                      # bandit no longer skips B404/B603/B607
```

Those three were suppressed only because the GUI shelled out to `node`. The security gate gets **stricter**.

## 2. The CLI reaches something it never could

Write a config for a league the CLI has never been able to touch:

```python
# nba.py
import os

from sportsbet.datasets import BasketballDataLoader, NBAStats, OddsApi

DATALOADER = BasketballDataLoader(
    param_grid={'league': ['NBA'], 'year': [2026]},
    stats=NBAStats(),
    odds=OddsApi(key=os.environ['ODDS_API_KEY']),
)
```

```bash
sportsbet dataloader params  --config-path nba.py
sportsbet dataloader prepare --config-path nba.py --dry-run
```

**This is the proof.** Before, the second command was impossible: the CLI constructed the dataloader with no sources,
so basketball raised *"no free default for odds"* and `OddsApi`'s key had nowhere to live. The dry run also prices the
fetch **exactly, for free**.

Soccer must still work with no ceremony added:

```python
# soccer.py
from sportsbet.datasets import SoccerDataLoader
DATALOADER = SoccerDataLoader(param_grid={'league': ['England'], 'year': [2025]})
```

And `params` must work with **nothing selected yet** — it is how you learn what to select:

```python
DATALOADER = SoccerDataLoader()
```

## 3. An old config fails usefully

A config still using `DATALOADER_CLASS` must be told **what to change** — not raise an `AttributeError`, and not quietly
do nothing.

## 4. The assistant cannot spend by accident

```text
estimate_preparation(config)          -> cost: {'odds_api': 17722}
prepare(config)                       -> REFUSED, the cost was never confirmed
prepare(config, confirm_cost=100)     -> REFUSED, the real cost is 17722
prepare(config, confirm_cost=17722)   -> runs
```

A free preparation needs no confirmation. The rule exists to stop surprise spending, not to add ceremony.

## 5. The docs audit

Every public name still has a runnable example. It reported 37/37 before this feature; it must still report 100%.
