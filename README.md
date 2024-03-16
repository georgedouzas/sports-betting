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

| Date       | League      | Home Team           | Away Team     |   Home Win |   Draw |   Away Win |   Over 2.5 |   Under 2.5 |
|:-----------|:------------|:--------------------|:--------------|-----------:|-------:|-----------:|-----------:|------------:|
| 2024-03-17 | Italy       | Verona              | Milan         |       2.98 |   4.15 |       3.14 |       1.81 |        2.23 |
| 2024-03-17 | France      | Brest               | Lille         |       2.64 |   4.11 |       3.67 |       1.78 |        2.28 |
| 2024-03-17 | Germany     | Karlsruhe           | Magdeburg     |       2.42 |   3.52 |       3.6  |       1.66 |        2.51 |
| 2024-03-17 | Belgium     | Cercle Brugge       | RWD Molenbeek |       1.93 |   4.26 |       4.27 |       1.68 |        2.47 |
| 2024-03-17 | Belgium     | Oud-Heverlee Leuven | Mechelen      |       2.61 |   4.1  |       3.48 |       1.83 |        2.21 |
| 2024-03-17 | Belgium     | Westerlo            | Genk          |       2.86 |   4.15 |       3.44 |       1.62 |        2.6  |
| 2024-03-17 | England     | West Ham            | Aston Villa   |       2.59 |   4.22 |       3.45 |       1.84 |        2.19 |
| 2024-03-17 | England     | Leeds               | Millwall      |       1.84 |   4.3  |       4.27 |       1.82 |        2.22 |
| 2024-03-17 | England     | Salford             | Morecambe     |       2.42 |   4.09 |       3.74 |       1.79 |        2.27 |
| 2024-03-17 | England     | Oxford City         | Halifax       |       2.86 |   4.27 |       3.44 |       2.03 |        1.97 |
| 2024-03-17 | Belgium     | Gent                | Charleroi     |       2.19 |   4.23 |       3.87 |       1.79 |        2.26 |
| 2024-03-17 | Turkey      | Kasimpasa           | Galatasaray   |       2.9  |   4.74 |       3.37 |       1.56 |        2.79 |
| 2024-03-17 | France      | Clermont            | Le Havre      |       2.61 |   4.13 |       3.55 |       1.78 |        2.28 |
| 2024-03-17 | France      | Reims               | Metz          |       2.38 |   4.23 |       3.57 |       1.81 |        2.24 |
| 2024-03-17 | France      | Rennes              | Marseille     |       2.49 |   4.09 |       3.58 |       1.77 |        2.29 |
| 2024-03-17 | France      | Montpellier         | Paris SG      |       2.9  |   4.74 |       3.37 |       1.81 |        2.23 |
| 2024-03-17 | Germany     | Freiburg            | Leverkusen    |       2.73 |   4.57 |       4.15 |       1.69 |        2.44 |
| 2024-03-17 | Germany     | Dortmund            | Ein Frankfurt |       2.19 |   4.2  |       3.87 |       1.53 |        2.87 |
| 2024-03-17 | Germany     | Hamburg             | Wehen         |       2.19 |   4.2  |       3.83 |       1.68 |        2.48 |
| 2024-03-17 | Germany     | Hertha              | Schalke 04    |       2.54 |   3.97 |       3.58 |       1.66 |        2.51 |
| 2024-03-17 | France      | Monaco              | Lorient       |       1.72 |   4.26 |       4.27 |       1.68 |        2.48 |
| 2024-03-17 | Turkey      | Trabzonspor         | Fenerbahce    |       2.86 |   4.26 |       3.18 |       1.85 |        2.18 |
| 2024-03-17 | Belgium     | St Truiden          | Club Brugge   |       2.9  |   4.26 |       3.41 |       1.78 |        2.28 |
| 2024-03-17 | Italy       | Juventus            | Genoa         |       2.09 |   4.08 |       3.93 |       1.83 |        2.21 |
| 2024-03-17 | Italy       | Atalanta            | Fiorentina    |       2.4  |   4.07 |       3.8  |       1.78 |        2.29 |
| 2024-03-17 | Netherlands | Sparta Rotterdam    | Ajax          |       2.59 |   4.1  |       3.48 |       1.79 |        2.26 |
| 2024-03-17 | Spain       | Ferrol              | Valladolid    |       2.64 |   4.65 |       3.61 |       1.92 |        2.09 |
| 2024-03-17 | Turkey      | Rizespor            | Gaziantep     |       2.37 |   4.02 |       3.8  |       1.81 |        2.23 |
| 2024-03-17 | Turkey      | Hatayspor           | Samsunspor    |       2.61 |   4.14 |       3.49 |       1.81 |        2.24 |
| 2024-03-17 | Spain       | Zaragoza            | Espanol       |       2.71 |   4.11 |       3.63 |       1.89 |        2.12 |
| 2024-03-17 | Belgium     | Antwerp             | St. Gilloise  |       2.94 |   4.1  |       2.81 |       1.82 |        2.21 |
| 2024-03-17 | Italy       | Roma                | Sassuolo      |       1.85 |   4.19 |       4.27 |       1.69 |        2.45 |
| 2024-03-17 | Italy       | Inter               | Napoli        |       2.21 |   4.23 |       3.76 |       1.8  |        2.25 |
| 2024-03-17 | Italy       | Ascoli              | Lecco         |       2.41 |   3.95 |       3.8  |       1.79 |        2.27 |
| 2024-03-17 | Netherlands | Volendam            | AZ Alkmaar    |       2.45 |   3.93 |       3.03 |       1.8  |        2.25 |
| 2024-03-17 | Netherlands | Heerenveen          | Feyenoord     |       2.99 |   4.74 |       3.37 |       1.78 |        2.28 |
| 2024-03-17 | Netherlands | Utrecht             | Nijmegen      |       2.53 |   4.39 |       3.62 |       1.76 |        2.32 |
| 2024-03-17 | Spain       | Elche               | Albacete      |       2.35 |   4.11 |       3.97 |       1.88 |        2.13 |
| 2024-03-17 | Netherlands | PSV Eindhoven       | Twente        |       1.84 |   4.26 |       4.27 |       1.79 |        2.27 |
| 2024-03-17 | Portugal    | Moreirense          | Arouca        |       2.61 |   4.16 |       3.66 |       1.78 |        2.28 |
| 2024-03-17 | Portugal    | Casa Pia            | Benfica       |       2.45 |   4.77 |       3.37 |       1.83 |        2.21 |
| 2024-03-17 | Portugal    | Sp Lisbon           | Boavista      |       1.44 |   4.77 |       4.73 |       1.84 |        2.19 |
| 2024-03-17 | Scotland    | Dundee              | Rangers       |       2.9  |   4.74 |       3.37 |       1.85 |        2.18 |
| 2024-03-17 | Spain       | Sevilla             | Celta         |       2.45 |   4.09 |       3.73 |       1.82 |        2.22 |
| 2024-03-17 | Spain       | Las Palmas          | Almeria       |       2.37 |   4.09 |       3.73 |       1.82 |        2.22 |
| 2024-03-17 | Spain       | Villarreal          | Valencia      |       2.5  |   4.09 |       3.68 |       1.81 |        2.24 |
| 2024-03-17 | Spain       | Vallecano           | Betis         |       2.59 |   4.13 |       3.65 |       1.84 |        2.2  |
| 2024-03-17 | Spain       | Ath Madrid          | Barcelona     |       2.54 |   4.09 |       3.65 |       1.84 |        2.19 |
| 2024-03-17 | Spain       | Leganes             | Mirandes      |       2.36 |   4.23 |       3.97 |       1.88 |        2.13 |
| 2024-03-17 | Portugal    | Chaves              | Guimaraes     |       2.86 |   4.16 |       3.46 |       1.82 |        2.22 |
| 2024-03-18 | England     | Crawley Town        | Stockport     |       2.8  |   4.2  |       3.46 |       1.83 |        2.2  |
| 2024-03-18 | Spain       | Andorra             | Amorebieta    |       1.94 |   4.31 |       3.98 |       1.88 |        2.13 |

**Backtesting results**

| Training Start      | Training End        | Training Period    | Testing Start       | Testing End         | Testing Period    |   Start Value |   End Value |   Total Return [%] |   Total Bets |   Win Rate [%] |   Best Bet [%] |   Worst Bet [%] |   Avg Winning Bet [%] |   Avg Losing Bet [%] |   Profit Factor |   Sharpe Ratio |   Avg Bet Yield [%] |   Std Bet Yield [%] |
|:--------------------|:--------------------|:-------------------|:--------------------|:--------------------|:------------------|--------------:|------------:|-------------------:|-------------:|---------------:|---------------:|----------------:|----------------------:|---------------------:|----------------:|---------------:|--------------------:|--------------------:|
| 2016-01-08 00:00:00 | 2018-01-01 00:00:00 | 724 days 00:00:00  | 2018-01-01 00:00:00 | 2019-08-23 00:00:00 | 600 days 00:00:00 |          1000 |     1086.67 |              8.667 |           94 |        54.2553 |        298.667 |        -177.778 |               97.013  |             -75.2638 |         1.82677 |        1.84751 |            19.0062  |            109.276  |
| 2016-01-08 00:00:00 | 2019-08-23 00:00:00 | 1323 days 00:00:00 | 2019-08-23 00:00:00 | 2021-03-13 00:00:00 | 569 days 00:00:00 |          1000 |     1621.76 |             62.176 |         2046 |        47.9472 |       1280     |        -181.818 |               83.0855 |             -66.4805 |         1.19179 |        2.19658 |             5.39468 |            102.742  |
| 2016-01-08 00:00:00 | 2021-03-13 00:00:00 | 1891 days 00:00:00 | 2021-03-13 00:00:00 | 2022-10-08 00:00:00 | 575 days 00:00:00 |          1000 |     1748.45 |             74.845 |         1874 |        50.1067 |        761     |        -184.615 |               79.6518 |             -65.0628 |         1.2515  |        2.71599 |             7.51841 |             98.0629 |
| 2016-01-08 00:00:00 | 2022-10-08 00:00:00 | 2465 days 00:00:00 | 2022-10-08 00:00:00 | 2024-03-14 00:00:00 | 524 days 00:00:00 |          1000 |     1440.1  |             44.01  |         1896 |        49.3143 |        601.333 |        -185.714 |               75.438  |             -70.5617 |         1.14638 |        1.82575 |             1.66038 |             96.9537 |

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