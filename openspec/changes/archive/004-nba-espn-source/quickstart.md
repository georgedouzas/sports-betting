# Quickstart: proving the NBA works

Prerequisites: the repo installed (`pdm install`), and — for the live checks only — an Odds API key in `.env` as
`odds-api-key`. The key never leaves the machine and is never printed.

## 1. The gate, which is most of the proof

```bash
pdm run formatting
pdm run checks
pdm run tests
```

**The acceptance criterion is SC-007: no existing test may be edited.** The NBA is a new file. If an existing test needs
changing to stay green, the abstraction failed and the work stops there and gets reported — it does not get patched.

## 2. The offline source

```bash
pdm run pytest tests/datasets/test_nba.py -v
```

These run against payloads shaped like the real response and touch no network (the socket guard enforces it). The ones
that matter most are the two halves of the all-star trap:

- an exhibition labelled `regular-season` is still **excluded**, and
- a playoff game not labelled `STD` is still **included**,

plus the guard test that pins the monthly request window below the feed's silent 1,000-event cap.

## 3. The catalogue, live and free

```python
from sportsbet.datasets import NBAStats

params = NBAStats().available_params()
print(len(params), params[-1])
```

Expect the seasons the feed publishes, named by the year each **ends** in. Free — the seasons index costs nothing.

## 4. A season, end to end

```python
import os
from sportsbet.datasets import BasketballDataLoader, NBAStats, OddsApi

dataloader = BasketballDataLoader(
    param_grid={'league': ['NBA'], 'year': [2025]},
    stats=NBAStats(),
    odds=OddsApi(key=os.environ['odds-api-key']),
)

report = dataloader.prepare(dry_run=True)
print(report.to_fetch, report.estimated_cost)
```

`dry_run=True` fetches the **free** statistics, derives the schedule from them, and prices the odds items exactly —
**spending zero credits**. That is the two-stage plan, and it is why the estimate is exact rather than a guess.

Then, for real:

```python
dataloader.prepare()
X, Y, O = dataloader.extract_train_data()
X_fix, _, O_fix = dataloader.extract_fixtures_data()
```

**What to check**:

| Expectation | Why it matters |
| --- | --- |
| `Y` has `home_win` and `away_win`, and **no `draw`** | a tie goes to overtime (FR-007) |
| `X`, `Y` and `O` have the same number of rows | the reconciliation placed every club (US5) |
| the teams are exactly the league's **30** clubs | no exhibition survived (SC-002) |
| the season's game count matches the real one | nothing was silently truncated (SC-003) |
| a season in progress has results for games already played | the NBA is bettable, not just backtestable (SC-004) |

## 5. Reconciliation

The roster-bijection resolver should pair all 30 clubs with **zero** aliases: ESPN writes the canonical names and so
does the vendor. If it cannot place a club it raises and **names** it, rather than dropping its games.

An alias is added only where a club genuinely cannot be placed, with its reason recorded. A wrong alias attaches one
club's prices to another club's game and says nothing about it — worse than not matching at all.

## 6. The docs audit

```bash
# every public name must still appear in a runnable example
pdm run python -c "..."   # the audit from the docs work
```

`NBAStats` is a new public name, so it needs an example. The audit must still report 100%.
