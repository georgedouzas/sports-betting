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

| Date       | League      | Home Team     | Away Team        |   Home Win |   Draw |   Away Win |   Over 2.5 |   Under 2.5 |
|:-----------|:------------|:--------------|:-----------------|-----------:|-------:|-----------:|-----------:|------------:|
| 2023-10-01 | Italy       | Bologna       | Empoli           |       2.27 |   4.12 |       3.92 |       1.8  |        2.24 |
| 2023-10-01 | Italy       | Udinese       | Genoa            |       2.49 |   4.17 |       3.6  |       1.92 |        2.09 |
| 2023-10-01 | Turkey      | Konyaspor     | Besiktas         |       2.57 |   4.05 |       3.38 |       1.85 |        2.17 |
| 2023-10-01 | France      | Toulouse      | Metz             |       2.7  |   4.32 |       3.35 |       1.84 |        2.19 |
| 2023-10-01 | France      | Lorient       | Montpellier      |       2.57 |   4.11 |       3.38 |       1.86 |        2.16 |
| 2023-10-01 | France      | Rennes        | Nantes           |       2.05 |   4.41 |       4.81 |       1.87 |        2.15 |
| 2023-10-01 | England     | Blackburn     | Leicester        |       2.57 |   4.22 |       3.38 |       1.85 |        2.18 |
| 2023-10-01 | Germany     | Darmstadt     | Werder Bremen    |       2.57 |   4.02 |       3.28 |       1.71 |        2.42 |
| 2023-10-01 | Germany     | Freiburg      | Augsburg         |       2.06 |   4.05 |       4.5  |       1.67 |        2.5  |
| 2023-10-01 | France      | Nice          | Brest            |       2.29 |   4.14 |       3.78 |       1.8  |        2.25 |
| 2023-10-01 | France      | Reims         | Lyon             |       2.45 |   4.21 |       3.45 |       1.95 |        2.05 |
| 2023-10-01 | Greece      | Volos NFC     | Panserraikos     |       2.49 |   3.01 |       3.51 |       1.95 |        2.05 |
| 2023-10-01 | Germany     | Elversberg    | Greuther Furth   |       2.5  |   3.82 |       3.4  |       1.67 |        2.48 |
| 2023-10-01 | Greece      | Lamia         | Panetolikos      |       2.34 |   4.53 |       3.83 |       1.94 |        2.06 |
| 2023-10-01 | England     | Nott&#39;m Forest | Brentford        |       2.57 |   4.33 |       3.4  |       1.95 |        2.05 |
| 2023-10-01 | Belgium     | RWD Molenbeek | Gent             |       2.83 |   4.3  |       2.44 |       1.8  |        2.25 |
| 2023-10-01 | Belgium     | Club Brugge   | St Truiden       |       1.6  |   4.41 |       5.03 |       1.86 |        2.16 |
| 2023-10-01 | Greece      | Aris          | Kifisias         |       1.71 |   4.6  |       5.48 |       2.02 |        1.98 |
| 2023-10-01 | Greece      | Panathinaikos | PAOK             |       2.34 |   4.31 |       3.83 |       1.95 |        2.06 |
| 2023-10-01 | Greece      | Giannina      | Olympiakos       |       3.19 |   4.55 |       1.71 |       2.02 |        1.98 |
| 2023-10-01 | Germany     | Osnabruck     | Kaiserslautern   |       2.57 |   4.8  |       2.44 |       1.73 |        2.37 |
| 2023-10-01 | France      | Le Havre      | Lille            |       2.79 |   4.37 |       2.6  |       1.86 |        2.16 |
| 2023-10-01 | Portugal    | Arouca        | Chaves           |       2.34 |   4.58 |       3.67 |       1.81 |        2.23 |
| 2023-10-01 | Germany     | Nurnberg      | Magdeburg        |       2.57 |   4.05 |       3.28 |       1.67 |        2.48 |
| 2023-10-01 | Spain       | Zaragoza      | Mirandes         |       2.25 |   4.11 |       3.98 |       1.94 |        2.06 |
| 2023-10-01 | Italy       | Cremonese     | Parma            |       2.49 |   4.17 |       3.57 |       1.73 |        2.38 |
| 2023-10-01 | Spain       | Amorebieta    | Cartagena        |       2.2  |   4.31 |       3.54 |       1.95 |        2.05 |
| 2023-10-01 | Turkey      | Fenerbahce    | Rizespor         |       1.55 |   4.78 |       5.37 |       1.75 |        2.33 |
| 2023-10-01 | Turkey      | Karagumruk    | Kasimpasa        |       2.51 |   4.11 |       3.51 |       1.82 |        2.22 |
| 2023-10-01 | Turkey      | Ad. Demirspor | Alanyaspor       |       2.34 |   4.22 |       3.75 |       1.77 |        2.3  |
| 2023-10-01 | Spain       | Huesca        | Sp Gijon         |       2.51 |   4.28 |       3.92 |       1.96 |        2.04 |
| 2023-10-01 | Belgium     | Genk          | Westerlo         |       1.91 |   4.41 |       4.8  |       1.86 |        2.16 |
| 2023-10-01 | Italy       | Atalanta      | Juventus         |       2.64 |   3.98 |       3.51 |       1.84 |        2.18 |
| 2023-10-01 | Italy       | Roma          | Frosinone        |       1.56 |   4.31 |       5.48 |       1.87 |        2.15 |
| 2023-10-01 | Italy       | Bari          | Como             |       2.48 |   3.34 |       3.6  |       1.93 |        2.08 |
| 2023-10-01 | Italy       | Cittadella    | Lecco            |       2.27 |   4.06 |       3.92 |       1.87 |        2.14 |
| 2023-10-01 | Spain       | Valladolid    | Burgos           |       2.25 |   4.11 |       4.05 |       1.94 |        2.06 |
| 2023-10-01 | Italy       | Sampdoria     | Catanzaro        |       2.48 |   3.98 |       3.53 |       1.9  |        2.11 |
| 2023-10-01 | Italy       | Palermo       | Sudtirol         |       1.86 |   4    |       3.4  |       1.91 |        2.1  |
| 2023-10-01 | Netherlands | Excelsior     | Sparta Rotterdam |       2.57 |   4.21 |       3.38 |       1.81 |        2.23 |
| 2023-10-01 | Netherlands | Heracles      | Zwolle           |       2.26 |   4.18 |       3.53 |       1.78 |        2.29 |
| 2023-10-01 | Belgium     | St. Gilloise  | Charleroi        |       2.21 |   4.41 |       3.92 |       1.8  |        2.25 |
| 2023-10-01 | Netherlands | AZ Alkmaar    | For Sittard      |       1.55 |   4.78 |       5.97 |       1.84 |        2.19 |
| 2023-10-01 | Portugal    | Guimaraes     | Estoril          |       2.34 |   4.28 |       3.83 |       1.84 |        2.2  |
| 2023-10-01 | Portugal    | Rio Ave       | Moreirense       |       2.47 |   4.2  |       3.51 |       1.86 |        2.16 |
| 2023-10-01 | Spain       | Almeria       | Granada          |       2.4  |   4.18 |       3.65 |       1.83 |        2.21 |
| 2023-10-01 | Spain       | Alaves        | Osasuna          |       2.62 |   4.28 |       3.55 |       2    |        2    |
| 2023-10-01 | Spain       | Ath Madrid    | Cadiz            |       1.5  |   4.78 |       5.53 |       1.98 |        2.02 |
| 2023-10-01 | Spain       | Betis         | Valencia         |       2.47 |   4.46 |       3.54 |       1.87 |        2.14 |
| 2023-10-01 | Spain       | Eldense       | Oviedo           |       2.69 |   4.28 |       3.55 |       1.99 |        2.01 |
| 2023-10-01 | Netherlands | Nijmegen      | Vitesse          |       2.47 |   4.11 |       3.53 |       1.82 |        2.22 |
| 2023-10-02 | Italy       | Fiorentina    | Cagliari         |       1.84 |   3.87 |       5.48 |       1.87 |        2.15 |
| 2023-10-02 | Italy       | Sassuolo      | Monza            |       2.49 |   4.04 |       3.4  |       1.8  |        2.24 |
| 2023-10-02 | Italy       | Torino        | Verona           |       2.29 |   4.15 |       3.83 |       1.84 |        2.19 |
| 2023-10-02 | Spain       | Espanol       | Ferrol           |       2.25 |   4.2  |       3.95 |       1.92 |        2.09 |
| 2023-10-02 | Portugal    | Gil Vicente   | Casa Pia         |       2.47 |   4.18 |       3.51 |       1.86 |        2.16 |
| 2023-10-02 | France      | Ajaccio       | Bastia           |       2.53 |   4.28 |       3.66 |       1.92 |        2.08 |
| 2023-10-02 | Greece      | OFI Crete     | AEK              |       2.87 |   4.45 |       1.71 |       1.95 |        2.05 |
| 2023-10-02 | Spain       | Las Palmas    | Celta            |       2.54 |   4.28 |       3.62 |       1.89 |        2.12 |
| 2023-10-02 | England     | Fulham        | Chelsea          |       2.6  |   4.39 |       3.34 |       1.92 |        2.09 |
| 2023-10-02 | Greece      | Atromitos     | Asteras Tripolis |       2.48 |   4.28 |       3.65 |       2.12 |        1.9  |
| 2023-10-02 | Turkey      | Kayserispor   | Buyuksehyr       |       2.57 |   4.29 |       3.51 |       1.83 |        2.21 |

**Backtesting results**

| Training Start      | Training End        | Training Period    | Testing Start       | Testing End         | Testing Period    |   Start Value |   End Value |   Total Return [%] |   Total Bets |   Win Rate [%] |   Best Bet [%] |   Worst Bet [%] |   Avg Winning Bet [%] |   Avg Losing Bet [%] |   Profit Factor |   Sharpe Ratio |   Avg Bet Yield [%] |   Std Bet Yield [%] |
|:--------------------|:--------------------|:-------------------|:--------------------|:--------------------|:------------------|--------------:|------------:|-------------------:|-------------:|---------------:|---------------:|----------------:|----------------------:|---------------------:|----------------:|---------------:|--------------------:|--------------------:|
| 2016-01-08 00:00:00 | 2017-09-23 00:00:00 | 624 days 00:00:00  | 2017-09-23 00:00:00 | 2018-11-23 00:00:00 | 427 days 00:00:00 |          1000 |     1000    |              0     |            0 |       nan      |         nan    |         nan     |              nan      |             nan      |       nan       |      inf       |          nan        |            nan      |
| 2016-01-08 00:00:00 | 2018-11-23 00:00:00 | 1050 days 00:00:00 | 2018-11-23 00:00:00 | 2020-01-11 00:00:00 | 415 days 00:00:00 |          1000 |     1156.87 |             15.687 |          622 |        49.5177 |         674    |        -175     |               77.1742 |             -64.2006 |         1.15485 |        1.26801 |            6.11455  |             96.0011 |
| 2016-01-08 00:00:00 | 2020-01-11 00:00:00 | 1464 days 00:00:00 | 2020-01-11 00:00:00 | 2021-04-16 00:00:00 | 462 days 00:00:00 |          1000 |     1529.91 |             52.991 |         1665 |        48.2883 |        1033.33 |        -183.333 |               84.9691 |             -69.6008 |         1.20502 |        2.63005 |            5.16376  |            106.067  |
| 2016-01-08 00:00:00 | 2021-04-16 00:00:00 | 1925 days 00:00:00 | 2021-04-16 00:00:00 | 2022-07-31 00:00:00 | 472 days 00:00:00 |          1000 |     1766.97 |             76.697 |         1485 |        50.5724 |         573.25 |        -183.333 |               80.2816 |             -64.9321 |         1.33934 |        3.50553 |            8.59336  |             97.5795 |
| 2016-01-08 00:00:00 | 2022-07-31 00:00:00 | 2396 days 00:00:00 | 2022-07-31 00:00:00 | 2023-09-28 00:00:00 | 425 days 00:00:00 |          1000 |     1225.3  |             22.53  |         1430 |        47.2028 |         436.4  |        -180     |               75.0159 |             -68.3229 |         1.09758 |        1.19924 |           -0.471869 |             93.7992 |

## Quick start

`sports-betting` supports all common sports betting needs i.e. fetching historical and fixtures data as well as backtesting of
betting strategies.

### Parameters

Assume that we would like to fetch historical data of various leagues for specific years, including the maximum odds of the market
and dropping columns that contain more than 20% of missing values:

```python
leagues = ['England', 'Scotland', 'Germany', 'Italy', 'Spain', 'France', 'Netherlands', 'Belgium', 'Portugal', 'Turkey', 'Greece']
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
data_sources = ['footballdata']
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
dataloader = SoccerDataLoader({'league': leagues, 'year': years, 'data_source': data_sources})
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
  `'inti_cash'`. You can refer to the [API](api/datasets) for more details about their values.

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
sportsbet dataloader training -c dataloader_config.py -d /path/to/directory
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