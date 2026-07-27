[scikit-learn]: <http://scikit-learn.org/stable/>
[black badge]: <https://img.shields.io/badge/%20style-black-000000.svg>
[black]: <https://github.com/psf/black>
[docformatter badge]: <https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg>
[docformatter]: <https://github.com/PyCQA/docformatter>
[ruff badge]: <https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json>
[ruff]: <https://github.com/charliermarsh/ruff>
[mypy badge]: <http://www.mypy-lang.org/static/mypy_badge.svg>
[mypy]: <http://mypy-lang.org>
[mkdocs badge]: <https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat>
[mkdocs]: <https://squidfunk.github.io/mkdocs-material>
[safety badge]: <https://img.shields.io/badge/safety-checked-green>
[safety]: <https://github.com/pyupio/safety>
[bandit badge]: <https://img.shields.io/badge/security-bandit-yellow>
[bandit]: <https://github.com/PyCQA/bandit>
[pytest badge]: <https://img.shields.io/badge/tests-pytest-blue>
[pytest]: <https://github.com/pytest-dev/pytest>
[coverage badge]: <https://img.shields.io/badge/coverage-pytest--cov-blue>
[coverage]: <https://github.com/nedbat/coveragepy>
[interrogate badge]: <https://img.shields.io/badge/docstring-interrogate-blue>
[interrogate]: <https://github.com/econchick/interrogate>
[pre-commit badge]: <https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit>
[pre-commit]: <https://github.com/pre-commit/pre-commit>
[nox badge]: <https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg>
[nox]: <https://github.com/wntrblm/nox>
[version badge]: <https://img.shields.io/pypi/v/sports-betting.svg>
[pythonversion badge]: <https://img.shields.io/pypi/pyversions/sports-betting.svg>
[downloads badge]: <https://img.shields.io/pypi/dd/sports-betting>
[discussions]: <https://github.com/georgedouzas/sports-betting/discussions>
[discussions badge]: <https://img.shields.io/github/discussions/georgedouzas/sports-betting>
[ci]: <https://github.com/georgedouzas/sports-betting/actions?query=workflow>
[ci badge]: <https://github.com/georgedouzas/sports-betting/actions/workflows/ci.yml/badge.svg?branch=main>
[doc]: <https://github.com/georgedouzas/sports-betting/actions?query=workflow>
[doc badge]: <https://github.com/georgedouzas/sports-betting/actions/workflows/doc.yml/badge.svg?branch=main>

# sports-betting

[![ci][ci badge]][ci] [![doc][doc badge]][doc]

| Category          | Tools    |
| ------------------| -------- |
| Development   | [![black][black badge]][black] [![ruff][ruff badge]][ruff] [![mypy][mypy badge]][mypy] [![docformatter][docformatter badge]][docformatter] |
| Testing       | [![pytest][pytest badge]][pytest] [![coverage][coverage badge]][coverage] [![interrogate][interrogate badge]][interrogate] |
| Security      | [![safety][safety badge]][safety] [![bandit][bandit badge]][bandit] |
| Automation    | [![nox][nox badge]][nox] [![pre-commit][pre-commit badge]][pre-commit] |
| Package       | ![version][version badge] ![pythonversion][pythonversion badge] ![downloads][downloads badge] |
| Documentation | [![mkdocs][mkdocs badge]][mkdocs]|
| Communication | [![discussions][discussions badge]][discussions] |

## Introduction

`sports-betting` is a set of tools for building, testing, and using sports betting models. You can use it from an AI
agent, from Python, or from the command line.

The library has two main parts, dataloaders and bettors.

- A dataloader downloads the data and shapes it for modelling. You build it from a statistics source and an odds source.
  You choose both, so you always know where your data came from.
- A bettor backtests a betting strategy and predicts the value bets of upcoming events. It wraps any scikit-learn
  estimator.

A bettor finds value bets. The execution part places one of them. It takes a fitted bettor, one match, and a stake. It
watches the match. At the moment the model was fitted for, it places the bet once. Some bookmakers have a betting API,
and the library places the bet through the API. Other bookmakers have no API, so your agent drives their website on your
own account. You place the bet at a bookmaker where you hold an account. This spends real money.

## Installation

### Install

`sports-betting` is on PyPI. Install it with `pip`.

```bash
pip install sports-betting
```

### MCP server

The MCP server lets you drive the library from an AI agent. Install it with the `mcp` extra.

```bash
pip install 'sports-betting[mcp]'
```

### Execution

Execution places bets. It also drives a bookmaker's website, which needs a browser. Install the extras, then install the
browser.

```bash
pip install 'sports-betting[mcp,execution]'
python -m playwright install chromium
```

### Development

For development, clone the repository and install the project with [PDM](https://github.com/pdm-project/pdm). This
installs the main and development dependencies.

```bash
git clone https://github.com/georgedouzas/sports-betting.git
cd sports-betting
pdm install
```

## Quick start

### AI agent

The agent is a first-class way to use the library. It reaches everything that Python and the command line reach. It
also does more. It explores the data, builds tables, plots results, and writes the model. A betting model is a
scikit-learn estimator. An agent can write one, run it, and tell you how well it did.

Install the MCP server and register it with your agent.

```bash
pip install 'sports-betting[mcp]'
claude mcp add sportsbet -- sportsbet-mcp
```

You can then ask the agent to work through a full task. A typical session runs in these steps.

- Ask what data is available. The agent reads the catalogue, which is free.
- Ask for a strategy on some leagues and seasons. The agent downloads the seasons and backtests models.
- Ask which league holds the edge. The agent breaks the result down.
- Ask for the value bets in the upcoming fixtures. The agent extracts the fixtures and applies the model.
- Ask it to place a bet. The agent runs the single-event execution.

The agent names the environment variable that holds your API key. It never prints the key. Extracting the data
downloads it, so a paid odds feed spends money only when you extract. Extract once and save the dataloader instead of
extracting again. Driving a bookmaker's website on your account breaches most bookmakers' terms of service and risks the
account and its balance.

### Python API

You can do the same work in code. You build a dataloader from a statistics source and an odds source.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import FootballDataOdds, FootballDataStats

dataloader = DataLoader(
    param_grid={'league': ['Germany', 'Italy', 'France'], 'division': [1, 2], 'year': [2021, 2022, 2023, 2024]},
    stats=FootballDataStats(),
    odds=FootballDataOdds(),
)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum')
X_fix, _, O_fix = dataloader.extract_fixtures_data()
```

A betting model is any scikit-learn estimator wrapped in a bettor. The next example backtests a bettor that wraps a
logistic regression classifier.

```python
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sportsbet.evaluation import ClassifierBettor, backtest

classifier = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']), remainder='passthrough'
    ),
    SimpleImputer(),
    MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=7, class_weight='balanced')),
)
bettor = ClassifierBettor(classifier, betting_markets=['draw'], init_cash=10000.0, stake=50.0)
backtest(bettor, X_train, Y_train, O_train, cv=TimeSeriesSplit(5))
```

```text
                                                       Number of betting days  Number of bets  Yield percentage per bet  ROI percentage  Final cash
Training start Training end Testing start Testing end
2020-08-21     2021-02-27   2021-02-27    2021-11-06                     625            1247                       3.6            22.3     12234.0
               2021-11-06   2021-11-06    2022-05-14                     645            1305                       3.4            22.0     12205.0
               2022-05-14   2022-05-15    2023-02-25                     639            1349                      -2.3           -15.8      8420.0
               2023-02-25   2023-02-26    2023-11-06                     637            1344                       6.7            44.9     14491.5
               2023-11-06   2023-11-06    2024-06-02                     659            1348                       8.0            54.1     15409.5
```

Fit it with `bettor.fit(X_train, Y_train, O_train)`, then `bettor.bet(X_fix, O_fix)` returns the value bets of the
upcoming matches.

`extract_fixtures_data` downloads the fixtures separately and returns the upcoming betting events. The training data and
the fixtures data share their columns, so the model trained on the history can bet on the fixtures.

### CLI

The command line mirrors the API. The `dataloader` command extracts the data. The `evaluation` command works a model on
it. `dataloader train extract` downloads the seasons once and saves the dataloader. The evaluation commands read that
file, so the download runs once and the training runs once.

```bash
# Download the seasons once and save the dataloader
sportsbet dataloader train extract --stats football-data --odds football-data \
  --league Germany --league Italy --league France --division 1 --division 2 \
  --year 2021 --year 2022 --year 2023 --year 2024 \
  --odds-type market_maximum -o dataloader.pkl

# Backtest a model on the saved data
sportsbet evaluation backtest --dataloader dataloader.pkl --model logistic \
  --betting-market home_win --betting-market draw --betting-market away_win \
  --init-cash 10000 --stake 50 --cv 5

# Fit the model once and save it
sportsbet evaluation fit --dataloader dataloader.pkl --model logistic \
  --betting-market home_win --betting-market draw --betting-market away_win \
  -o model.pkl

# Value bets for the upcoming matches, through the fitted model
sportsbet evaluation bet --dataloader dataloader.pkl --bettor model.pkl
```

The last command prints the value bets of the upcoming matches. `--stats` and `--odds` say where the data comes from.
`dataloader train extract` downloads it. The ready-made models cover the common cases. You can also write your own model
in a Python file and name it, as in `--model models.py:bettor`, where `bettor` is the object below.

```python
# models.py
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sportsbet.evaluation import ClassifierBettor

classifier = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']), remainder='passthrough'
    ),
    SimpleImputer(),
    MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=7, class_weight='balanced')),
)
bettor = ClassifierBettor(classifier, betting_markets=['draw'], init_cash=10000.0, stake=50.0)
```
