[scikit-learn]: <https://scikit-learn.org>
[GridSearchCV]: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>
[TimeSeriesSplit]: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
[decision tree classifier]: <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>

# Bettor

A bettor turns probabilities into bets. The library ships three.

* [`OddsComparisonBettor`][sportsbet.evaluation.OddsComparisonBettor]: bets by comparing the odds of different providers.
* [`ClassifierBettor`][sportsbet.evaluation.ClassifierBettor]: bets with a [scikit-learn] classifier.
* [`BettorGridSearchCV`][sportsbet.evaluation.BettorGridSearchCV]: tunes a bettor's parameters by cross validated search.

## Betting strategy

A betting strategy looks for value bets, the events where the bookmaker underestimates the probability of the outcome.
The true probability is unknown, so the comparison is between the bettor's estimate and the bookmaker's. A bet is a
value bet when the bettor gives the outcome a higher probability than the odds imply.

## Initialization

Every bettor takes three parameters from the base class.

* `betting_markets`: the markets to bet on, as a list of market base names such as `['home_win', 'draw', 'away_win']` or
  `['over_2.5', 'under_2.5']`. When it is `None`, the default, every market in the targets is used.
* `init_cash`: the starting bankroll for the backtest cash simulation. Defaults to `10000.0`.
* `stake`: the amount staked on each bet. Defaults to `50.0`.

```python
from sportsbet.evaluation import OddsComparisonBettor
bettor = OddsComparisonBettor(betting_markets=['home_win', 'draw', 'away_win'], init_cash=10000.0, stake=50.0)
assert bettor.betting_markets == ['home_win', 'draw', 'away_win']
```

Each bettor adds its own parameters on top of these.

* [`ClassifierBettor`][sportsbet.evaluation.ClassifierBettor] adds `classifier`, any [scikit-learn] classifier with
  `fit` and `predict_proba`.
* [`OddsComparisonBettor`][sportsbet.evaluation.OddsComparisonBettor] adds `odds_types`, the odds providers averaged into
  the consensus probability, defaulting to `'market_average'`, and `alpha`, an adjustment subtracted from that
  probability, where larger values bet less often.

The example below uses a classifier bettor built around a [decision tree classifier]. The features include categorical
columns (`league`, `home_team`, `away_team`) and columns with missing values, so the classifier goes in a pipeline that
encodes and imputes them.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sportsbet.evaluation import ClassifierBettor
classifier = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']),
        remainder='passthrough',
    ),
    SimpleImputer(),
    DecisionTreeClassifier(random_state=0),
)
bettor = ClassifierBettor(classifier=classifier, betting_markets=['home_win', 'draw', 'away_win'])
```

The data is the sample the library ships with, so everything runs offline.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats
dataloader = DataLoader(
    param_grid={'league': ['England']},
    stats=SampleSoccerStats(),
    odds=SampleSoccerOdds(),
)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')
```

It is a real Premier League season, frozen, read from the bundled files on your disk. The season is finished, so it has
no fixtures, and everything below runs on the training data. To bet on upcoming matches, point the same bettor at
`extract_fixtures_data()` of a live source such as [`FootballDataStats`][sportsbet.sources.FootballDataStats].

## Implementation

Bettors implement these public methods.

* `fit` fits the model to the input data `X`, the multi output targets `Y` and optional odds `O`.
* `predict` predicts the class labels of betting events.
* `predict_proba` predicts the class probabilities of betting events.
* `bet` returns the value bets.
* `score` returns the annual Sharpe ratio of the predicted value bets.

The `backtest` function computes backtesting statistics for a bettor over historical data. All of the above rest on two
private methods, `_fit` and `_predict_proba`, so a new betting model is easy to write. `_fit` learns from historical
data, and `_predict_proba` returns the class probabilities.

### Writing your own bettor

Subclass [`BaseBettor`][sportsbet.evaluation.BaseBettor] and implement those two. You get value bets, backtesting and
hyperparameter search for free.

```python
import numpy as np
from sportsbet.evaluation import BaseBettor


class HomeAdvantageBettor(BaseBettor):
    """A bettor that always likes the home side a little more than the market does."""

    def __init__(self, edge=0.05, betting_markets=None):
        super().__init__(betting_markets=betting_markets)
        self.edge = edge

    def _fit(self, X, Y, O):
        self.rates_ = Y.mean(axis=0).to_numpy()      # how often each market came in
        return self

    def _predict_proba(self, X):
        markets = list(self.betting_markets_)
        probabilities = np.tile(self.rates_, (len(X), 1))
        probabilities[:, markets.index('home_win')] += self.edge
        for event in self.complementary_events_:
            outcomes = [markets.index(market) for market in event]
            probabilities[:, outcomes] /= probabilities[:, outcomes].sum(axis=1, keepdims=True)
        return probabilities
```

`complementary_events_` comes from the data, so the same bettor renormalises over three outcomes in soccer and two in
basketball, and over each totals line, from the columns alone. See
[complementary events](#which-markets-are-mutually-exclusive-is-derived-from-the-data).

## Model fit

You fit the bettor to `(X_train, Y_train)` with `fit`. Fitting means the bettor takes from `(X_train, Y_train)` whatever
it uses at prediction time, whether or not that is a machine learning model.

```python
bettor.fit(X_train, Y_train)
```

The selected markets are stored as their base names.

```python
assert bettor.betting_markets_.tolist() == ['home_win', 'draw', 'away_win']
```

## Class labels prediction

Once fitted, predicting class labels is straightforward.

```python
predictions = bettor.predict(X_train)
assert predictions.shape == (380, 3)
```

There are 380 training matches and three selected markets, so the result is `(380, 3)`. The target columns of the
training data show every market that is modelled.

```python
assert Y_train.columns.tolist() == [
    'home_win__postplay__0min',
    'draw__postplay__0min',
    'away_win__postplay__0min',
    'over_2.5__postplay__0min',
    'under_2.5__postplay__0min'
]
```

Value bets come from comparing the predicted probabilities to the odds matrix `O`, so the probabilities carry the
signal, more than the labels.

## Class probabilities predictions

Predicting positive class probabilities is also simple. There is one probability per selected market, and for mutually
exclusive markets like `home_win`, `draw` and `away_win` the probabilities are normalised to sum to one.

```python
probabilities = bettor.predict_proba(X_train)
assert probabilities.shape == (380, 3)
assert abs(probabilities.sum(axis=1)[0] - 1.0) < 1e-6
```

### Which markets are mutually exclusive is derived from the data

Nothing is named in advance. [`complementary_events`][sportsbet.evaluation.complementary_events] reads the markets your
data carries and works out which of them are exhaustive.

* `over` and `under` are complementary at whatever the line is, 2.5 goals, 1.5 goals, 220.5 points. A line the library
  has never seen is grouped like any other.
* The outcome of a match is whichever of `home_win`, `draw` and `away_win` the data has.

That second rule comes from the data: `home_win` and `away_win` are complementary in a sport without a draw, and joined
by `draw` in one with it. The columns say which sport this is.

```python
from sportsbet.evaluation import complementary_events

# soccer: a draw is possible, so a home win and an away win are not exhaustive on their own
assert complementary_events(['home_win', 'draw', 'away_win', 'over_2.5', 'under_2.5']) == [
    ['home_win', 'draw', 'away_win'], ['over_2.5', 'under_2.5'],
]

# a sport without a draw: the outcome is two-way, and the line is wherever it is
assert complementary_events(['home_win', 'away_win', 'over_220.5', 'under_220.5']) == [
    ['home_win', 'away_win'], ['over_220.5', 'under_220.5'],
]
```

A bettor can override it with `COMPLEMENTARY_EVENTS` on its class.

## Value bets prediction

The fitted bettor predicts the value bets with `bet`, which returns one boolean column per selected market. You can join
these with the identity columns of `X_train`.

```python
import pandas as pd
markets = bettor.betting_markets_.tolist()
value_bets = pd.concat(
    [
        X_train.reset_index()[['date', 'home_team', 'away_team']],
        pd.DataFrame(bettor.bet(X_train, O_train), columns=markets),
    ],
    axis=1,
).set_index('date')
assert value_bets.columns.tolist() == ['home_team', 'away_team', 'home_win', 'draw', 'away_win']
assert len(value_bets) == len(X_train)
```

## Backtesting

The `backtest` function evaluates a bettor's strategy over the training data tuple `(X_train, Y_train, O_train)`.

```python
from sportsbet.evaluation import backtest
backtesting_results = backtest(bettor, X_train, Y_train, O_train)
```

It takes three further parameters.

* `cv`: a [scikit-learn] [TimeSeriesSplit] giving the successive train and test splits. `None`, the default, uses a
  default `TimeSeriesSplit`.
* `n_jobs`: the number of CPU cores for the parallel runs. `-1`, the default, uses all of them.
* `verbose`: the verbosity level.

```python
from sklearn.model_selection import TimeSeriesSplit
backtesting_results = backtest(bettor, X_train, Y_train, O_train, cv=TimeSeriesSplit(2), n_jobs=1)
```

The results are indexed by the training and testing periods and carry overall and per market metrics, the latter
labelled by the market base names.

```python
assert backtesting_results.index.names == [
    'Training start',
    'Training end',
    'Testing start',
    'Testing end'
]
assert backtesting_results.columns.tolist() == [
    'Number of betting days',
    'Number of bets',
    'Yield percentage per bet',
    'ROI percentage',
    'Final cash',
    'Number of bets (home_win)',
    'Number of bets (draw)',
    'Number of bets (away_win)',
    'Yield percentage per bet (home_win)',
    'Yield percentage per bet (draw)',
    'Yield percentage per bet (away_win)'
]
```

## Scoring

The `score` method returns the annual Sharpe ratio of the value bets predicted for the given data, which is convenient
as an optimisation objective.

```python
sharpe_ratio = bettor.score(X_train, Y_train, O_train)
```

## Odds comparison bettor

The [`OddsComparisonBettor`][sportsbet.evaluation.OddsComparisonBettor] takes its probabilities straight from the odds,
averaging the `odds_types` providers and subtracting `alpha`. So it needs the odds matrix `O` at fit time too.

```python
from sportsbet.evaluation import OddsComparisonBettor
odds_bettor = OddsComparisonBettor(odds_types=['market_average'], alpha=0.03, betting_markets=['home_win', 'draw', 'away_win'])
odds_bettor.fit(X_train, Y_train, O_train)
value_bets = odds_bettor.bet(X_train, O_train)
assert value_bets.shape == (380, 3)
```

## Hyperparameter search

[`BettorGridSearchCV`][sportsbet.evaluation.BettorGridSearchCV] tunes a bettor's parameters by cross validated grid
search, like [scikit-learn]'s [GridSearchCV]. It wraps a bettor as `estimator`, searches the values in `param_grid`,
whose keys are the bettor's parameter names, and like the underlying bettor exposes `fit`, `predict`, `predict_proba`,
`bet` and `score`. It also takes the usual scikit-learn search parameters, `scoring`, `n_jobs`, `refit`, `cv` and
`verbose`, with `cv` defaulting to a [TimeSeriesSplit]. Once wrapped it is used like any other bettor, for example
inside `backtest`.

```python
from sportsbet.evaluation import BettorGridSearchCV, OddsComparisonBettor
from sklearn.model_selection import TimeSeriesSplit
search = BettorGridSearchCV(
    estimator=OddsComparisonBettor(),
    param_grid={'alpha': [0.02, 0.05, 0.1]},
    cv=TimeSeriesSplit(2),
)
backtesting_results = backtest(search, X_train, Y_train, O_train, cv=TimeSeriesSplit(2))
assert 'Number of bets' in backtesting_results.columns
```

## Saving and loading

You can save a fitted bettor to disk and reload it later with `load_bettor`.

```python
import tempfile
from pathlib import Path
from sportsbet.evaluation import save_bettor, load_bettor
path = str(Path(tempfile.mkdtemp()) / 'bettor.pkl')
save_bettor(bettor, path)
reloaded = load_bettor(path)
assert reloaded.predict(X_train).shape == (380, 3)
```
