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

`sports-betting` is a set of tools for creating, testing and using sports betting models. You can use it from an AI agent, from
Python, or from the command line.

Its two main components are dataloaders and bettors:

- A dataloader downloads the data and shapes it for modelling. It is built from a statistics source and an odds source. You choose
  both, so you always know where your data came from.
- A bettor backtests a betting strategy and predicts the value bets of upcoming events. It wraps any scikit-learn
  estimator.

## Installation

For user installation, `sports-betting` is currently available on the PyPi's repository, and you can install it via `pip`:

```bash
pip install sports_betting
```

To drive the library from an AI agent, install the MCP server:

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

## Quick start

### AI agent

The agent is a first-class way into the library. It reaches everything Python and the command line reach, and it does
more besides. It explores the data, tables the results, plots them, writes the model, and does anything else you ask. A
betting model is a scikit-learn estimator, and an agent can write one, run it, and tell you whether it was any good.

```bash
pip install 'sports_betting[mcp]'
claude mcp add sportsbet -- sportsbet-mcp
```

What follows is a chat with Claude, using the library through the MCP server, with the real output.

You: What soccer data can I work with?

> Claude: 918 league, division and season combinations across 27 leagues, from 1994 to 2026.
>
> England goes deepest, with five divisions. Germany, Italy, France, Spain, Scotland, the Netherlands, Belgium,
> Portugal, Turkey and Greece have one or two each, and there are single-division feeds for another sixteen countries,
> from Argentina to the USA.
>
> Every one comes with historical odds from several bookmakers, so a backtest runs against real prices. I have not
> downloaded anything, since reading the catalogue is free.

You: Find me a strategy that works on the top two divisions of Germany, Italy and France, 2021 to 2024.

> Claude: Downloaded 24 season files. Then I searched, rather than guessing.
>
> First, which market is even worth betting. A logistic model on each, at the best odds in the market:
>
> | Market | Yield per bet | Bets per fold |
> | --- | ---: | ---: |
> | home_win | −3.8% | 901 |
> | draw | +3.9% | 1,319 |
> | away_win | −3.6% | 1,217 |
> | over_2.5 | +0.3% | 642 |
> | under_2.5 | −2.2% | 702 |
>
> Only the draw pays. That is not a surprise: the draw is the outcome punters least like backing, so the price drifts.
>
> Then, whether the market is worth searching over. The markets to bet on are a hyperparameter like any other, so I put
> them in the grid alongside the model's regularisation and let the search pick:
>
> ```python
> BettorGridSearchCV(
>     estimator=ClassifierBettor(classifier, init_cash=10000.0, stake=50.0),
>     param_grid={
>         'classifier__multioutputclassifier__estimator__C': [0.1, 1.0, 50.0],
>         'betting_markets': [['home_win'], ['draw'], ['away_win'], ['home_win', 'draw', 'away_win']],
>     },
>     cv=TimeSeriesSplit(3),
> )
> ```
>
> Backtested over five time-ordered folds:
>
> | Testing period | Bets | Yield per bet | Final cash |
> | --- | ---: | ---: | ---: |
> | Feb 2021 to Nov 2021 | 1,407 | +1.0% | £10,715 |
> | Nov 2021 to May 2022 | 1,406 | +8.3% | £15,812 |
> | May 2022 to Feb 2023 | 1,406 | −0.5% | £9,675 |
> | Feb 2023 to Nov 2023 | 1,406 | +4.5% | £13,161 |
> | Nov 2023 to Jun 2024 | 1,407 | −0.2% | £9,858 |
>
> Mean yield +2.6%, positive in three folds of five. Better than the fixed model, which managed +1.2%.
>
> Worth being honest about: I have now tried a dozen configurations on this data. Some of that +2.6% is me searching,
> not the model knowing. Before you stake anything, hold a season back and check it there.

You: Which league is that draw edge actually in?

> Claude: Almost entirely Italy.
>
> | League | Yield per bet on the draw |
> | --- | ---: |
> | Italy | +9.0% |
> | Germany | +2.9% |
> | France | −0.6% |
>
> That is a big number and I would treat it with suspicion rather than excitement. It is one league over four seasons,
> found after I had already looked at several splits.

You: Write me the Italy draw model, then, and show me the folds.

> Claude: Here it is, a logistic regression on the two Italian divisions, betting only the draw:
>
> ```python
> from sklearn.compose import make_column_transformer
> from sklearn.impute import SimpleImputer
> from sklearn.linear_model import LogisticRegression
> from sklearn.multioutput import MultiOutputClassifier
> from sklearn.pipeline import make_pipeline
> from sklearn.preprocessing import OneHotEncoder
> from sportsbet.evaluation import ClassifierBettor
>
> classifier = make_pipeline(
>     make_column_transformer(
>         (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']), remainder='passthrough'
>     ),
>     SimpleImputer(),
>     MultiOutputClassifier(LogisticRegression(solver='liblinear', class_weight='balanced')),
> )
> bettor = ClassifierBettor(classifier, betting_markets=['draw'], init_cash=10000.0, stake=50.0)
> ```
>
> | Testing period | Bets | Yield per bet | Final cash |
> | --- | ---: | ---: | ---: |
> | Mar 2021 to Nov 2021 | 440 | +1.7% | £10,365 |
> | Nov 2021 to Aug 2022 | 461 | +12.7% | £12,934 |
> | Aug 2022 to Mar 2023 | 487 | +4.5% | £11,097 |
> | Mar 2023 to Nov 2023 | 470 | +8.9% | £12,085 |
> | Nov 2023 to Jun 2024 | 485 | +17.2% | £14,171 |
>
> Positive in all five folds, mean +9.0%. That is the kind of result I would want to distrust hardest. I found it by
> narrowing to the league that already looked good. Give me a season I have never touched and I will tell you if it
> holds.

You: Fine. Any value bets in the upcoming fixtures?

> Claude: None right now. `extract_fixtures_data` comes back empty for Germany, Italy and France: their seasons are in
> the summer break, so there is no upcoming match to price.
>
> The mechanism is worth seeing. `param_grid` chose the seasons to train on, and those are all played. The fixtures come
> from a separate download, the current season of the leagues you selected, so training and fixtures never overlap.
>
> ```python
> dataloader = DataLoader(
>     param_grid={'league': ['Germany', 'Italy', 'France'], 'division': [1, 2], 'year': [2021, 2022, 2023, 2024]},
>     stats=FootballDataStats(), odds=FootballDataOdds(),
> )
> X, Y, O = dataloader.extract_train_data(odds_type='market_maximum')   # the seasons above, to train on
> X_fix, _, O_fix = dataloader.extract_fixtures_data()                  # the current season's upcoming matches
> ```
>
> They share their columns, so when the season restarts the model trained on the history bets on the fixtures. To bet on
> another league, select it. Given what we found I would hold off staking: the edge is Italy alone, four seasons found
> after a dozen tries. Hold a season back and check it there first.

The agent names the environment variable holding your API key, never the key itself. Extracting the data downloads it,
so a paid odds feed spends only when you extract. Do it once and `save` the dataloader rather than re-extracting.

### Python API

The same functionality, in code. A dataloader is created by a statistics source and an odds source:

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

A betting model is any scikit-learn estimator wrapped in a bettor. Here we backtest a bettor that wraps a logistic regression
classifier:

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

The fixtures are downloaded separately by `extract_fixtures_data`, which returns the upcoming betting events. The
training and fixtures data share their columns, so the model trained on the history can bet on the fixtures.

### CLI

The same scenario from the command line, which mirrors the API. `dataloader` extracts the data and `evaluation` works a
model on it. `dataloader train extract` downloads the seasons once and saves the dataloader. The evaluation commands
read that file, so the data is downloaded once and the model is trained once.

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

The last command prints the value bets of the upcoming matches. `--stats` and `--odds` say where the data comes from,
and `dataloader train extract` downloads it. The ready-made models cover the common cases. A model of your own is
Python, so you write it in a file and name it, as in `--model models.py:bettor`, where `bettor` is the object below.

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
