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
[version badge]: <https://img.shields.io/pypi/v/sports-betting.svg>
[pythonversion badge]: <https://img.shields.io/pypi/pyversions/sports-betting.svg>
[downloads badge]: <https://img.shields.io/pypi/dd/sports-betting>
[gitter]: <https://gitter.im/sports-betting/community>
[gitter badge]: <https://badges.gitter.im/join%20chat.svg>
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
| **Package**       | ![version][version badge] ![pythonversion][pythonversion badge] ![downloads][downloads badge] |
| **Documentation** | [![mkdocs][mkdocs badge]][mkdocs]|
| **Communication** | [![gitter][gitter badge]][gitter] [![discussions][discussions badge]][discussions] |

## Introduction

Python sports betting toolbox.

The `sports-betting` package is a collection of tools that makes it easy to create machine learning models for sports betting and
evaluate their performance. It is compatible with [scikit-learn].

## Daily bet tips

This section will contain **daily** updated value bets of a betting strategy based on a machine learning model. You can read the
quick start guide below to understand the details or reproduce the results. Alternatively, you can visit regularly this page to
use the predictions for your own betting.

**Value bets**

| Date       | League   | Home Team          | Away Team      |   Home Win |   Draw |   Away Win |   Over 2.5 |   Under 2.5 |
|:-----------|:---------|:-------------------|:---------------|-----------:|-------:|-----------:|-----------:|------------:|
| 2023-10-14 | England  | Grimsby            | Accrington     |       2.36 |   4.08 |       3.52 |       1.86 |        2.16 |
| 2023-10-14 | England  | Forest Green       | Colchester     |       2.62 |   4.08 |       2.78 |       1.86 |        2.17 |
| 2023-10-14 | England  | Lincoln            | Burton         |       2.29 |   4.08 |       3.55 |       1.91 |        2.09 |
| 2023-10-14 | England  | Notts County       | Mansfield      |       2.41 |   3.71 |       3.5  |       1.84 |        2.19 |
| 2023-10-14 | England  | AFC Wimbledon      | Bradford       |       2.78 |   4.01 |       3.53 |       1.87 |        2.15 |
| 2023-10-14 | England  | Crewe              | Tranmere       |       2.04 |   4.08 |       3.55 |       1.87 |        2.16 |
| 2023-10-14 | England  | Doncaster          | Sutton         |       2.29 |   4.08 |       3.55 |       1.86 |        2.17 |
| 2023-10-14 | England  | Milton Keynes Dons | Barrow         |       2.36 |   4.08 |       3.33 |       1.99 |        2.01 |
| 2023-10-14 | England  | Harrogate          | Stockport      |       2.66 |   4.61 |       2.45 |       1.9  |        2.11 |
| 2023-10-14 | England  | Blackpool          | Stevenage      |       2.38 |   4.12 |       3.52 |       1.88 |        2.13 |
| 2023-10-14 | England  | Swindon            | Newport County |       2.1  |   4.14 |       3.94 |       1.83 |        2.21 |
| 2023-10-14 | England  | Walsall            | Gillingham     |       2.6  |   4.1  |       3.37 |       1.92 |        2.09 |
| 2023-10-14 | England  | Wrexham            | Salford        |       1.91 |   2.38 |       4.17 |       1.88 |        2.14 |
| 2023-10-14 | England  | Cambridge          | Shrewsbury     |       2.36 |   4.08 |       3.53 |       1.9  |        2.11 |
| 2023-10-14 | England  | Carlisle           | Leyton Orient  |       2.62 |   4.22 |       3.52 |       1.88 |        2.13 |
| 2023-10-14 | Spain    | Mirandes           | Tenerife       |       2.63 |   4.03 |       3.48 |       1.93 |        2.07 |
| 2023-10-14 | Spain    | Espanol            | Valladolid     |       2.24 |   4.08 |       3.59 |       1.85 |        2.18 |
| 2023-10-14 | Spain    | Sp Gijon           | Zaragoza       |       2.37 |   4    |       3.65 |       1.97 |        2.03 |
| 2023-10-14 | Spain    | Cartagena          | Santander      |       2.38 |   4.55 |       3.64 |       1.76 |        2.32 |
| 2023-10-15 | Spain    | Eibar              | Huesca         |       1.64 |   4.18 |       5    |       2    |        2    |
| 2023-10-15 | Spain    | Eldense            | Elche          |       2.6  |   4.12 |       3.42 |       1.92 |        2.09 |
| 2023-10-15 | Spain    | Leganes            | Amorebieta     |       2.11 |   4.09 |       4.11 |       1.97 |        2.03 |
| 2023-10-15 | Spain    | Burgos             | Villarreal B   |       2.35 |   3.94 |       3.79 |       1.93 |        2.07 |
| 2023-10-15 | Spain    | Andorra            | Alcorcon       |       2.36 |   4.38 |       3.67 |       1.96 |        2.05 |
| 2023-10-16 | Spain    | Levante            | Ferrol         |       2.1  |   4.08 |       4.22 |       1.86 |        2.16 |

**Backtesting results**

| Training Start      | Training End        | Training Period    | Testing Start       | Testing End         | Testing Period    |   Start Value |   End Value |   Total Return [%] |   Total Bets |   Win Rate [%] |   Best Bet [%] |   Worst Bet [%] |   Avg Winning Bet [%] |   Avg Losing Bet [%] |   Profit Factor |   Sharpe Ratio |   Avg Bet Yield [%] |   Std Bet Yield [%] |
|:--------------------|:--------------------|:-------------------|:--------------------|:--------------------|:------------------|--------------:|------------:|-------------------:|-------------:|---------------:|---------------:|----------------:|----------------------:|---------------------:|----------------:|---------------:|--------------------:|--------------------:|
| 2016-01-08 00:00:00 | 2017-09-25 00:00:00 | 626 days 00:00:00  | 2017-09-25 00:00:00 | 2018-11-25 00:00:00 | 427 days 00:00:00 |          1000 |     1000    |              0     |            0 |       nan      |        nan     |         nan     |              nan      |             nan      |       nan       |      inf       |          nan        |            nan      |
| 2016-01-08 00:00:00 | 2018-11-25 00:00:00 | 1052 days 00:00:00 | 2018-11-25 00:00:00 | 2020-01-24 00:00:00 | 426 days 00:00:00 |          1000 |     1204.4  |             20.44  |          680 |        48.3824 |        674     |        -175     |               80.0253 |             -64.0311 |         1.1945  |        1.52356 |            6.13755  |             97.9237 |
| 2016-01-08 00:00:00 | 2020-01-24 00:00:00 | 1477 days 00:00:00 | 2020-01-24 00:00:00 | 2021-04-24 00:00:00 | 457 days 00:00:00 |          1000 |     1626.66 |             62.666 |         1641 |        48.9336 |        857.143 |        -181.818 |               83.8059 |             -68.3494 |         1.2479  |        2.92606 |            6.14726  |            102.423  |
| 2016-01-08 00:00:00 | 2021-04-24 00:00:00 | 1933 days 00:00:00 | 2021-04-24 00:00:00 | 2022-08-16 00:00:00 | 480 days 00:00:00 |          1000 |     1551.96 |             55.196 |         1490 |        49.7987 |        536.667 |        -185.714 |               80.3641 |             -65.3676 |         1.23447 |        2.56372 |            7.29261  |             97.4973 |
| 2016-01-08 00:00:00 | 2022-08-16 00:00:00 | 2412 days 00:00:00 | 2022-08-16 00:00:00 | 2023-10-09 00:00:00 | 420 days 00:00:00 |          1000 |     1350.43 |             35.043 |         1444 |        48.0609 |        522     |        -186.667 |               75.4121 |             -68.3366 |         1.15157 |        1.70726 |            0.845068 |             93.7018 |

## Quick start

`sports-betting` supports all common sports betting needs i.e. fetching historical and fixtures data as well as backtesting of
betting strategies.

### Parameters

Assume that we would like to fetch historical data of various leagues for specific years, including the maximum odds of the market
and dropping columns that contain more than 20% of missing values:

```python
leagues = ['England', 'Scotland', 'Germany', 'Italy', 'Spain', 'France', 'Netherlands', 'Belgium', 'Portugal', 'Turkey', 'Greece']
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
odds_type = 'market_maximum'
drop_na_thres = 0.8
```

We would like also to use a `GradientBoostingClassifier` to support our betting strategy:

```python
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sportsbet.evaluation import ClassifierBettor

classifier = make_pipeline(
  make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']), remainder='passthrough'
  ),
  SimpleImputer(),
  MultiOutputClassifier(GradientBoostingClassifier(random_state=5)),
)
```

Finally, our backtesting parameters would include a 5-fold time ordered cross-validation and initial portfolio value of 1000:

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(5)
init_cash = 1000
```

### Process

Using the above selections, the betting process is the following:

- Create a dataloader that is used to fetch the training and fixtures data.

```python
from sportsbet.datasets import SoccerDataLoader
dataloader = SoccerDataLoader({'league': leagues, 'year': years})
X_train, Y_train, O_train = dataloader.extract_train_data(drop_na_thres=drop_na_thres, odds_type=odds_type)
X_fix, _, O_fix = dataloader.extract_fixtures_data()
```

- Create a bettor that selects and configures the betting strategy.

```python
from sportsbet.evaluation import ClassifierBettor
bettor = ClassifierBettor(classifier)
```

- Backtest the bettor on the training data to evaluate the betting strategy:

```python
bettor.backtest(X_train, Y_train, O_train, tscv=tscv, init_cash=init_cash)
bettor.backtest_results_[['Sharpe Ratio', 'Total Return [%]', 'Testing Period']].mean()
```

- Predict the value bets:

```python
bettor.bet(X_fix, O_fix)
```

## Sports betting in practice

You can think of any sports betting event as a random experiment with unknown probabilities for the various outcomes. Even for the
most unlikely outcome, for example scoring more than 10 goals in a soccer match, a small probability is still assigned. The
bookmaker estimates this probability P and offers the corresponding odds O. In theory, if the bookmaker offers the so-called fair
odds O = 1 / P in the long run, neither the bettor nor the bookmaker would make any money.

The bookmaker's strategy is to adjust the odds in their favor using the over-round of probabilities. In practice, it offers odds
less than the estimated fair odds. The important point here is that the bookmaker still has to estimate the probabilities of
outcomes and provide odds that guarantee them long-term profit.

On the other hand, the bettor can also estimate the probabilities and compare them to the odds the bookmaker offers. If the
estimated probability of an outcome is higher than the implied probability from the provided odds, then the bet is called a value
bet.

The only long-term betting strategy that makes sense is to select value bets. However, you have to remember that neither the
bettor nor the bookmaker can access the actual probabilities of outcomes. Therefore, identifying a value bet from the side of the
bettor is still an estimation. The bettor or the bookmaker might be wrong, or both of them.

Another essential point is that bookmakers can access resources that the typical bettor is rare to access. For instance, they have
more data, computational power, and teams of experts working on predictive models. You may assume that trying to beat them is
pointless, but this is not necessarily correct. The bookmakers have multiple factors to consider when they offer their adjusted
odds. This is the reason there is a considerable variation among the offered odds. The bettor should aim to systematically
estimate the value bets, backtest their performance, and not create arbitrarily accurate predictive models. This is a realistic
goal, and `sports-betting` can help by providing appropriate tools.

## Installation

For user installation, `sports-betting` is currently available on the PyPi's repository, and you can install it via `pip`:

```bash
pip install sports-betting
```

Development installation requires to clone the repository and then use [PDM](https://github.com/pdm-project/pdm) to install the
project as well as the main and development dependencies:

```bash
git clone https://github.com/georgedouzas/sports-betting.git
cd sports-betting
pdm install
```

## Usage

You can use the Python API or the CLI to access the full functionality of `sports-betting`. Nevertheless, it is recommended to be
familiar with the Python API since it is still needed to write configuration files for the CLI.

### API

The `sports-betting` package makes it easy to download sports betting data:

```python
from sportsbet.datasets import SoccerDataLoader
dataloader = SoccerDataLoader(param_grid={'league': ['Italy'], 'year': [2020]})
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum', drop_na_thres=1.0)
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

`X_train` are the historical/training data and `X_fix` are the test/fixtures data. The historical data can be used to backtest the
performance of a bettor model:

```python
from sportsbet.evaluation import ClassifierBettor
from sklearn.dummy import DummyClassifier
bettor = ClassifierBettor(DummyClassifier())
bettor.backtest(X_train, Y_train, O_train)
```

We can use the trained bettor model to predict the value bets using the fixtures data:

```python
bettor.bet(X_fix, O_fix)
```

### CLI

The command `sportsbet` provides various sub-commands to download data and predict the value bets. For any sub-command you may
add the `--help` flag to get more information about its usage.

#### Configuration

In order to use the commands, a configuration file is required. You can find examples of such configuration files in
`sports-betting/configs/`. The configuration file should have a Python file extension and contain a dictionary `CONFIG`:

```python
CONFIG = {
  'data': {
    'dataloader': ...,
    'param_grid': {

    },
    'drop_na_thres': ...,
    'odds_type': ...
  },
  'betting': {
    ...: ...,
    'bettor': ...,
    'tscv': ...,
    'init_cash': ...
  }
}
```

The dictionary `CONFIG` has the following structure:

- Two mandatory keys `'data'` and `'betting'` that configure the data extraction and betting strategy, respectively and contain
  other nested dictionaries as values.
- The `'data'` key has a nested dictionary as a value with a mandatory key '`dataloader`' and the optional keys `'param_grid'`,
  `'drop_na_thres'` and `'drop_na'`. You can refer to the [API](api/datasets) for more details about their values.
- The `'betting'` key has a nested dictionary as a value with a mandatory key '`bettor`' and the optional keys `'tscv'`, and
  `'init_cash'`. You can refer to the [API](api/datasets) for more details about their values.

#### Dataloader

Show available parameters for dataloaders:

```bash
sportsbet dataloader params -c config.py
```

Show available odds types:

```bash
sportsbet dataloader odds-types -c config.py
```

Extract training data and save them as CSV files:

```bash
sportsbet dataloader training -c config.py -d /path/to/directory
```

Extract fixtures data and save them as CSV files:

```bash
sportsbet dataloader fixtures -c config.py -d /path/to/directory
```

#### Bettor

Backtest the bettor and save the results as CSV file:

```bash
sportsbet bettor backtest -c config.py -d /path/to/directory
```

Get the value bets and save them as CSV file:

```bash
sportsbet bettor bet -c config.py -d /path/to/directory
```