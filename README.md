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
[ci badge]: <https://github.com/georgedouzas/sports-betting/actions/workflows/ci.yml/badge.svg>
[doc]: <https://github.com/georgedouzas/sports-betting/actions?query=workflow>
[doc badge]: <https://github.com/georgedouzas/sports-betting/actions/workflows/doc.yml/badge.svg?branch=master>

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

### Sports betting in practice

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

Using `sports-betting` CLI, the betting process includes the following steps:

- Create a dataloader's configuration file that selects the training and fixtures data.

- Create a bettor's configuration file that selects and configures the betting strategy.

You can find examples of configuration files in `sports-betting/configs`.
  
Extract the training data:

```bash
sportsbet dataloader training -d dataloader_config.py
```

Extract the fixtures data:

```bash
sportsbet dataloader fixtures -d dataloader_config.py
```

Apply backtesting to estimate the performance of the model on future data:

```bash
sportsbet bettor backtest -b bettor_config.py -d dataloader_config.py
```

Get the value bets:

```bash
sportsbet bettor bettor -b bettor_config.py -d dataloader_config.py
```

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

#### Dataloader

The sub-command `sportsbet dataloader` requires a dataloader configuration file. You can find examples of such configuration files
in `sports-betting/configs/dataloaders`. The following conventions apply:

- The configuration file has a Python extension.

- It should include a dictionary called `MAIN` and two key-value pairs. A `'dataloader'` key with a dataloader class as
  value and a `'path'` key with a relative to the configuration file path as value. It will be the path of the pickled dataloader.

- It may include a dictionary called `OPTIONAL` and up to three key-value pairs. The three keys are `'param_grid'`,
  `'drop_na_thres'` and `'drop_na'`. You can refer to the [API](api/datasets) for more details about their values.

Using a dataloader configuration file and the following commands you can extract training and fixtures data.

Show available parameters for dataloaders:

```bash
sportsbet dataloader params -d dataloader_config.py
```

Show available odds types:

```bash
sportsbet dataloader odds-types -d dataloader_config.py
```

Extract training data:

```bash
sportsbet dataloader training -d dataloader_config.py
```

Extract fixtures data:

```bash
sportsbet dataloader fixtures -d dataloader_config.py
```

#### Bettor

The sub-command `sportsbet bettor` requires both bettor and dataloader configuration files. The dataloader configuration files are
explained above. For the bettor configuration files, you can find examples of such configuration files in
`sports-betting/configs/bettors`, while the following conventions apply:

- The configuration file has a Python extension.

- It should include a dictionary called `MAIN` and two key-value pairs. A `'bettor'` key with a bettor class as
  value and a `'path'` key with a relative to the configuration file path as value. It will be the path of the pickled bettor.

- It may include a dictionary called `OPTIONAL` and multiple key-value pairs. Two of the optional keys are `'tscv'` and
  `'init_cash'` The rest of the keys are the initialization parameters of the selected bettor. You can refer to the
  [API](api/evaluation) for more details about their values.

Using a bettor configuration file and the following commands you can backtest the bettor and estimate the value bets.

Backtest the bettor:

```bash
sportsbet bettor backtest -b bettor_config.py -d dataloader_config.py
```

Get the value bets:

```bash
sportsbet bettor bet -b bettor_config.py -d dataloader_config.py
```
