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
| **Development**   | [![black][black badge]][black] [![ruff][ruff badge]][ruff] [![mypy][mypy badge]][mypy] [![docformatter][docformatter badge]][docformatter] |
| **Testing**       | [![pytest][pytest badge]][pytest] [![coverage][coverage badge]][coverage] [![interrogate][interrogate badge]][interrogate] |
| **Security**      | [![safety][safety badge]][safety] [![bandit][bandit badge]][bandit] |
| **Automation**    | [![nox][nox badge]][nox] [![pre-commit][pre-commit badge]][pre-commit] |
| **Package**       | ![version][version badge] ![pythonversion][pythonversion badge] ![downloads][downloads badge] |
| **Documentation** | [![mkdocs][mkdocs badge]][mkdocs]|
| **Communication** | [![discussions][discussions badge]][discussions] |

## Introduction

The `sports-betting` package is a handy set of tools for creating, testing, and using sports betting models. It comes with a
Python API, a CLI, and an MCP server, so you can drive it from code, from a terminal, or from an AI assistant.

The main components of `sports-betting` are dataloaders and bettors objects:

- Dataloaders download and prepare data suitable for predictive modelling.
- Bettors provide an easy way to backtest betting strategies and predict the value bets of future events.

## Quick start

The `sports-betting` package makes it easy to download sports betting data. Soccer and basketball are supported today, and more
sports are on the way. The data is downloaded onto your own machine by an explicit `prepare` step:

```python
from sportsbet.datasets import SoccerDataLoader
dataloader = SoccerDataLoader(param_grid={'league': ['Italy'], 'year': [2020]})
dataloader.prepare()
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum')
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

`X_train` are the historical/training data and `X_fix` are the test/fixtures data. The historical data can be used to backtest the
performance of a bettor model:

```python
from sportsbet.evaluation import ClassifierBettor, backtest
from sklearn.dummy import DummyClassifier
bettor = ClassifierBettor(DummyClassifier())
backtest(bettor, X_train, Y_train, O_train)
```

We can use the trained bettor model to predict the value bets using the fixtures data:

```python
bettor.fit(X_train, Y_train)
bettor.bet(X_fix, O_fix)
```

## Sports betting in practice

A betting event is a random experiment. Every outcome has some probability of occurring, even an unlikely one, such as more than
ten goals in a soccer match and nobody knows what those probabilities actually are.

### Fair odds

The bookmaker estimates the probability $p$ of an outcome and offers odds $o$ on it. A bet of one unit returns $o$ if the outcome
occurs and nothing otherwise, so its expected profit is

$$
\mathbb{E}[\Pi] = p \, o - 1.
$$

The odds are fair when this is zero, that is when

$$
o = \frac{1}{p}.
$$

At fair odds, neither side makes profit in the long run.

### The bookmaker's margin

Bookmakers do not offer fair odds. They shorten them, so that the implied probability $1/o$ of every outcome is a little higher
than the probability they estimated. Across the $n$ mutually exclusive outcomes of an event, the implied probabilities therefore
sum to more than one:

$$
\sum_{i=1}^{n} \frac{1}{o_i} = 1 + m, \qquad m > 0.
$$

The excess $m$ is the over-round, and it is the bookmaker's margin. Note what has *not* changed: the bookmaker still has to
estimate $p$. The margin protects a good estimate, it does not replace one.

### Value bets

The bettor can estimate the probabilities too. Write the bettor's estimate as $\hat{p}$. A bet is a value bet when the bettor's
estimate exceeds the probability implied by the offered odds:

$$
\hat{p} > \frac{1}{o} \quad \Longleftrightarrow \quad \hat{p} \, o - 1 > 0,
$$

that is, when the bet has positive expected profit under the bettor's own estimate. Selecting value bets is the only betting
strategy that makes sense over the long run.

The caveat matters: neither side observes the true $p$. A value bet is a claim that $\hat{p}$ is closer to the truth than $1/o$ is
and the bettor can be wrong, the bookmaker can be wrong, or both.

### Is it hopeless?

Bookmakers have more data, more computing power and teams of analysts. It is tempting to conclude that competing with them is
pointless, but that does not follow. Bookmakers balance many concerns beyond accuracy: their exposure, their competitors, the
weight of public money, which is why the odds offered on the same event vary noticeably from one bookmaker to another. That
variation is the opening.

The goal is therefore not to build an arbitrarily accurate model of football. It is to identify value bets systematically and
backtest them honestly. A realistic aim, and the one `sports-betting` is built to serve.

## Installation

For user installation, `sports-betting` is currently available on the PyPi's repository, and you can install it via `pip`:

```bash
pip install sports_betting
```

To drive the library from an AI assistant, install the MCP server:

```bash
pip install 'sports_betting[mcp]'
```

Development installation requires to clone the repository and then use [PDM](https://github.com/pdm-project/pdm) to install the
project as well as the main and development dependencies:

```bash
git clone https://github.com/georgedouzas/sports-betting.git
cd sports-betting
pdm install
```

## Usage

You can access `sports-betting` through the Python API, the CLI, or an AI assistant via the MCP server. All three reach the same
capabilities, every sport and every data source, while all three cover the common sports betting needs: fetching historical and
fixtures data, backtesting betting strategies, and predicting value bets.

### API

Assume we would like to backtest the following scenario and use the bettor object to predict value bets:

- Selection of data
  - First and second division of German, Italian and French leagues for the years 2021-2024
  - Maximum odds of the market in order to backtest our betting strategy
- Configuration of betting strategy
  - 5-fold time ordered cross-validation
  - Initial cash of 10000 euros
  - Stake of 50 euros for each bet
  - Use match odds (home win, away win and draw) as betting markets
  - Logistic regression classifier to predict probabilities and value bets

```python
# Selection of data
from sportsbet.datasets import SoccerDataLoader

leagues = ['Germany', 'Italy', 'France']
divisions = [1, 2]
years = [2021, 2022, 2023, 2024]
odds_type = 'market_maximum'
dataloader = SoccerDataLoader({'league': leagues, 'year': years, 'division': divisions})
dataloader.prepare()
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type=odds_type)
X_fix, _, O_fix = dataloader.extract_fixtures_data()

# Configuration of betting strategy
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sportsbet.evaluation import ClassifierBettor, backtest

tscv = TimeSeriesSplit(5)
init_cash = 10000.0
stake = 50.0
betting_markets = ['home_win', 'draw', 'away_win']
classifier = make_pipeline(
  make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']), remainder='passthrough'
  ),
  SimpleImputer(),
  MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=7, class_weight='balanced', C=50)),
)
bettor = ClassifierBettor(classifier, betting_markets=betting_markets, stake=stake, init_cash=init_cash)

# Apply backtesting and get results
backtesting_results = backtest(bettor, X_train, Y_train, O_train, cv=tscv)

# Get value bets for upcoming betting events
bettor.fit(X_train, Y_train)
bettor.bet(X_fix, O_fix)
```

### CLI

Everything the Python API does, the `sportsbet` command does. It has two groups of sub-commands: `data` selects, downloads and
extracts data, and `model` backtests a betting model and predicts the value bets. Pass `--help` to any of them to see what it
takes.

The same scenario, without writing any Python. The selection of the data is the same, and so is the betting strategy:

```bash
# Selection of data
sportsbet data prepare --sport soccer \
  --league Germany --league Italy --league France \
  --division 1 --division 2 \
  --year 2021 --year 2022 --year 2023 --year 2024

# Apply backtesting and get results
sportsbet model backtest --sport soccer \
  --league Germany --league Italy --league France \
  --division 1 --division 2 \
  --year 2021 --year 2022 --year 2023 --year 2024 \
  --odds-type market_maximum \
  --model logistic \
  --betting-market home_win --betting-market draw --betting-market away_win \
  --init-cash 10000 --stake 50 --cv 5

# Get value bets for upcoming betting events
sportsbet model bet --sport soccer \
  --league Germany --league Italy --league France \
  --division 1 --division 2 \
  --year 2021 --year 2022 --year 2023 --year 2024 \
  --odds-type market_maximum \
  --model logistic \
  --betting-market home_win --betting-market draw --betting-market away_win
```

### AI assistant

Install the MCP server and an assistant can drive the library for you:

```bash
pip install 'sports_betting[mcp]'
```

Point your assistant at the `sportsbet-mcp` command. Find what data exists, price a download before it spends anything, prepare
it, backtest a model and return the value bets. Its tools take the same arguments as the CLI, so anything you can ask for on the
command line, you can ask for in plain language.

