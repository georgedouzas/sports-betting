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

```text
                                                       Number of betting days  Number of bets  Yield percentage per bet  ROI percentage  Final cash
Training start Training end Testing start Testing end
2019-08-23     2019-10-05   2019-10-06    2019-11-24                      61             243                     -20.6           -25.0      7502.5
               2019-11-24   2019-11-25    2020-01-18                      72             248                      -8.7           -10.8      8922.0
               2020-01-18   2020-01-18    2020-03-01                      72             246                       3.5             4.3     10430.5
               2020-03-01   2020-03-01    2020-07-10                      63             242                      -4.3            -5.2      9479.0
               2020-07-10   2020-07-10    2020-08-02                      49             241                       0.7             0.8     10082.0
```

The results are reported for each fold of the time ordered cross-validation, along with a breakdown per betting market
that is not shown above. A model that bets at random loses money, as it should.

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

```text
                                                       Number of betting days  Number of bets  Yield percentage per bet  ROI percentage  Final cash
Training start Training end Testing start Testing end
2020-08-21     2021-02-27   2021-02-27    2021-11-06                     625            1407                      -0.1            -0.4      9957.5
               2021-11-06   2021-11-06    2022-05-14                     645            1407                       8.7            61.4     16137.0
               2022-05-14   2022-05-15    2023-02-25                     639            1408                       1.7            11.9     11186.5
               2023-02-25   2023-02-26    2023-11-06                     637            1407                      -0.4            -3.0      9703.5
               2023-11-06   2023-11-06    2024-06-02                     659            1404                      -4.0           -28.0      7204.5
```

The model wins on some periods and loses on others, which is what a betting model does. A backtest that reports a yield
of tens of percent is telling you about a mistake in the data, not about an edge.

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

`prepare` reports what it downloaded:

```text
Downloaded 25 items.
```

The backtesting results are the same as the ones above. They are shown as a table when the terminal is wide enough, and
a fold at a time when it is not:

```text
Backtesting results
Training start                        2020-08-21
Training end                          2021-02-27
Testing start                         2021-02-27
Testing end                           2021-11-06
Number of betting days                625
Number of bets                        1407
Yield percentage per bet              -0.10
ROI percentage                        -0.40
Final cash                            9,957.50
Number of bets (home_win)             238
Number of bets (draw)                 709
Number of bets (away_win)             460
Yield percentage per bet (home_win)   -0.40
Yield percentage per bet (draw)       4.20
Yield percentage per bet (away_win)   -6.40
...
```

Add `-o ./data` to write the results as CSV files.

### AI assistant

The assistant is a first-class way into the library, alongside the Python API and the CLI. It reaches everything they
reach — every sport, every league, every data source, every model — because all three are told the same things in the
same way.

```bash
pip install 'sports_betting[mcp]'
```

Register the `sportsbet-mcp` command with your assistant. With [Claude Code](https://claude.com/claude-code):

```bash
claude mcp add sportsbet -- sportsbet-mcp
```

Or add it to Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sportsbet": {
      "command": "sportsbet-mcp"
    }
  }
}
```

Now ask for what you want.

---

**You:** *What soccer data can I work with?*

> **Claude:** There are **918 league–division–season combinations** across **27 leagues**, from 1994 to 2026.
>
> The big European leagues go deepest. England has five divisions covering 1994–2026; Germany, Italy, France, Spain,
> Scotland, Netherlands, Belgium, Portugal, Turkey and Greece have one or two divisions each. There are also
> single-division feeds for Argentina, Austria, Brazil, China, Denmark, Finland, Ireland, Japan, Mexico, Norway,
> Poland, Romania, Russia, Sweden, Switzerland and the USA.
>
> Every one of them comes with historical odds from several bookmakers, so you can backtest against real prices.
> Nothing has been downloaded yet — I only read the catalogue, which is free.

---

**You:** *Backtest a logistic model on the top two divisions of Germany, Italy and France, 2021 to 2024. Bet the match
outcome at the best odds in the market, £50 a bet from a £10,000 bank.*

> **Claude:** Downloaded 25 files, then backtested over five time-ordered folds. The model bets on every match, so
> roughly 1,400 bets a fold.
>
> | Testing period | Bets | Yield per bet | Final cash |
> | --- | ---: | ---: | ---: |
> | Feb 2021 – Nov 2021 | 1,407 | −0.1% | £9,957 |
> | Nov 2021 – May 2022 | 1,407 | **+8.7%** | £16,137 |
> | May 2022 – Feb 2023 | 1,408 | +1.7% | £11,186 |
> | Feb 2023 – Nov 2023 | 1,407 | −0.4% | £9,703 |
> | Nov 2023 – Jun 2024 | 1,404 | **−4.0%** | £7,204 |
>
> **It is not a winning strategy.** One period is strongly positive and the last is clearly negative, which is what an
> edge of roughly zero looks like when you slice it five ways. The model is betting indiscriminately — 1,400 bets a
> fold is nearly every match, in every market.
>
> Two things worth trying: raise the threshold so it only bets where it disagrees with the market, or restrict it to
> the market it is least bad at. On this run the draw was the only outcome it was ahead on.

---

**You:** *Are there any value bets in the upcoming fixtures?*

> **Claude:** One fixture is currently published, and the model does like it:
>
> | Date | Match | Bet | Odds |
> | --- | --- | --- | ---: |
> | 5 Dec, 19:00 | Bastia vs Red Star (France) | **Away win** | 2.90 |
>
> The model puts Red Star's chance above the 34.5% the price implies. The other outcomes are not value: home win at
> 2.60 and the draw at 3.13 are both priced above what the model thinks they are worth.
>
> Bear in mind the backtest above: this model's edge over five years was about zero. I would not stake real money on
> one of its picks without first finding a version that actually beats the market.

---

**You:** *What about the NBA?*

> **Claude:** NBA statistics are free, but nobody gives basketball odds away, so you would need a paid key from The
> Odds API.
>
> With one, a full 2025–26 season of pre-match head-to-head prices would cost **8,861 credits** and fetch 898 items. I
> have not started it — that is the estimate, and the estimate is free.
>
> Say the word and I will download it. Soccer, by contrast, costs nothing at all.

---

Every number above is real output from the library. The assistant never spends money without telling you the price
first: a download that costs anything is refused until the cost has been reported back to you. Your API key is not one
of the arguments it passes — it names the *environment variable* holding the key, so the key itself stays out of the
conversation.
