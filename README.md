[scikit-learn]: <http://scikit-learn.org/stable/>
[imbalanced-learn]: <http://imbalanced-learn.org/stable/>
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

## Installation

`sports-betting` is currently available on the PyPi's repository, and you can install it via `pip`:

```bash
pip install sports-betting
```

## Usage

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

Show available sports names:

```bash
sportsbet dataloader names
```

Show available parameters for dataloaders:

```bash
sportsbet dataloader params --name soccer
```

Select some of the above parameters and create a dataloader:

```bash
sportsbet dataloader create --name soccer --dataloader-path /path/dataloader.pkl --param-grid "league: Italy, Spain | year: 2021, 2022" --param-grid "league: Greece"
```

Show available odds types for the above dataloader:

```bash
sportsbet dataloader odds-types --dataloader-path /path/dataloader.pkl
```

Show and export training data:

```bash
sportsbet dataloader training --odds-type bet365 --drop-na-thres 1.0 --dataloader-path /path/dataloader.pkl --data-path /another/path/directory
```

Show and export fixtures data:

```bash
sportsbet dataloader fixtures --dataloader-path /path/dataloader.pkl --data-path /another/path/directory
```
