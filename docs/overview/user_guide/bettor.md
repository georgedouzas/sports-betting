[scikit-learn]: <https://scikit-learn.org>
[GridSearchCV]: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>
[TimeSeriesSplit]: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
[decision tree classifier]: <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>

# Bettor

This section presents the bettor object in detail. The available bettors are the following:

- [`OddsComparisonBettor`][sportsbet.evaluation.OddsComparisonBettor]: Bettor based on comparison of odds.
- [`ClassifierBettor`][sportsbet.evaluation.ClassifierBettor]: Bettor based on a [scikit-learn] classifier.
- [`BettorGridSearchCV`][sportsbet.evaluation.BettorGridSearchCV]: Tunes a bettor's parameters by cross-validated search.

## Betting strategy

The essence of any betting strategy is to identify the value bets i.e. betting events where the bookmaker underestimates the
probability of the event. Of course, the true probability of the betting event is unknown, thus the comparison is between the
estimated probability of the bettor and the bookmaker: a bet is a value bet when the bettor estimates a higher probability for the
event than the one implied by the odds.

## Initialization

Every bettor accepts three parameters from the base class:

- `betting_markets`: the markets to bet on, as a list of market **base** names such as `['home_win', 'draw', 'away_win']` or
  `['over_2.5', 'under_2.5']`. When it is `None` (the default) every market present in the targets is used.
- `init_cash`: the starting bankroll for the backtest cash simulation. Defaults to `10000.0`.
- `stake`: the amount staked on each bet. Defaults to `50.0`.

```python
from sportsbet.evaluation import OddsComparisonBettor
bettor = OddsComparisonBettor(betting_markets=['home_win', 'draw', 'away_win'], init_cash=10000.0, stake=50.0)
assert bettor.betting_markets == ['home_win', 'draw', 'away_win']
```

Each bettor adds its own parameters on top of these:

- [`ClassifierBettor`][sportsbet.evaluation.ClassifierBettor] adds `classifier`: any [scikit-learn] classifier implementing
  `fit` and `predict_proba`.
- [`OddsComparisonBettor`][sportsbet.evaluation.OddsComparisonBettor] adds `odds_types` (the odds providers averaged into the
  consensus probability, defaulting to `'market_average'`) and `alpha` (an adjustment subtracted from the consensus probability;
  larger values bet less often).

We use a classifier-based bettor built around [scikit-learn]'s [decision tree classifier]. The extracted features include
categorical columns (`league`, `home_team`, `away_team`) and columns with missing values, so we wrap the classifier in a pipeline
that encodes and imputes them:

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

We also use the offline dummy training and fixtures data, so everything runs without any network access:

```python
from sportsbet.datasets import DummySoccerDataLoader
dataloader = DummySoccerDataLoader(param_grid={'league': ['England']})
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

## Implementation

Bettors implement the following public methods:

- `fit` fits the model to the input data `X`, multi-output targets `Y` and optional odds `O`.
- `predict` predicts the class labels of betting events.
- `predict_proba` predicts the class probabilities of betting events.
- `bet` returns the value bets.
- `score` returns the annual Sharpe ratio of the predicted value bets.

The `backtest` function calculates backtesting statistics for a bettor over historical data. All of the above are based on the
private methods `_fit` and `_predict_proba`, which is what makes it easy to define new betting models: `_fit` learns from
historical data (and can be omitted, e.g. for an arbitrage bettor), while `_predict_proba` predicts the class probabilities (and
can be trivial when value bets are not derived from probabilities).

### Writing your own bettor

Subclass [`BaseBettor`][sportsbet.evaluation.BaseBettor] and implement those two. You get value bets, backtesting and
hyperparameter search for free:

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

Note what you did **not** write. `complementary_events_` is derived from the data, so the same bettor renormalizes over
three outcomes in soccer and two in basketball, and over each totals line separately, without being told which sport it
is — see [complementary events](#which-markets-are-mutually-exclusive-is-derived-from-the-data).

## Model fit

The bettor is fitted to the training data `(X_train, Y_train)` via the `fit` method. Fitting does not necessarily require a machine
learning model; more generally it means the bettor extracts from `(X_train, Y_train)` the information used when predictions are
made:

```python
bettor.fit(X_train, Y_train)
```

The selected markets are stored as their base names:

```python
assert bettor.betting_markets_.tolist() == ['home_win', 'draw', 'away_win']
```

## Class labels prediction

Once the model is fitted, predicting class labels is straightforward:

```python
predictions = bettor.predict(X_fix)
assert predictions.shape == (1, 3)
```

There is a single upcoming betting event and three selected markets, hence a `(1, 3)` result. The target columns of the training
data show every market that is modelled:

```python
assert Y_train.columns.tolist() == [
    'home_win__postplay__0min',
    'draw__postplay__0min',
    'away_win__postplay__0min',
    'over_2.5__postplay__0min',
    'under_2.5__postplay__0min'
]
```

Predicting class labels is not directly useful, since value bets should be based on the comparison of predicted probabilities to
the `O_fix` matrix, not on the labels.

## Class probabilities predictions

Predicting positive class probabilities is also simple. There is one probability per selected market and, for mutually-exclusive
markets like `home_win`/`draw`/`away_win`, the probabilities are normalized to sum to one:

```python
probabilities = bettor.predict_proba(X_fix)
assert probabilities.shape == (1, 3)
assert abs(probabilities.sum(axis=1)[0] - 1.0) < 1e-6
```

### Which markets are mutually exclusive is derived from the data

Nothing is named in advance. [`complementary_events`][sportsbet.evaluation.complementary_events] reads the markets your data
actually carries and works out which of them are exhaustive:

- `over` and `under` are complementary at **whatever the line is** — 2.5 goals, 1.5 goals, 220.5 points. A line the library has
  never seen is grouped like any other.
- The outcome of a match is **whichever of `home_win`, `draw` and `away_win` the data has**.

That second rule has to come from the data, and cannot be a list. `home_win` and `away_win` *are* complementary in a sport that
cannot be drawn, and are *not* in a sport that can — and only the data knows which sport this is:

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

A bettor can still override it by setting `COMPLEMENTARY_EVENTS` on its class, but it should not need to.

## Value bets prediction

The fitted bettor predicts the value bets with the `bet` method, which returns one boolean column per selected market. We can join
these with the identity columns of `X_fix`:

```python
import pandas as pd
markets = bettor.betting_markets_.tolist()
value_bets = pd.concat(
    [
        X_fix.reset_index()[['date', 'home_team', 'away_team']],
        pd.DataFrame(bettor.bet(X_fix, O_fix), columns=markets),
    ],
    axis=1,
).set_index('date')
assert value_bets.columns.tolist() == ['home_team', 'away_team', 'home_win', 'draw', 'away_win']
assert value_bets.reset_index()[['home_team', 'away_team']].values.tolist() == [['Arsenal', 'Chelsea']]
```

## Backtesting

The `backtest` function evaluates a bettor's strategy over the training data tuple `(X_train, Y_train, O_train)`:

```python
from sportsbet.evaluation import backtest
backtesting_results = backtest(bettor, X_train, Y_train, O_train)
```

It accepts three further parameters:

- `cv`: a [scikit-learn] [TimeSeriesSplit] providing the successive train/test splits. `None` (the default) uses a default
  `TimeSeriesSplit`.
- `n_jobs`: the number of CPU cores for the parallel runs; `-1` (the default) uses all processors.
- `verbose`: the verbosity level.

```python
from sklearn.model_selection import TimeSeriesSplit
backtesting_results = backtest(bettor, X_train, Y_train, O_train, cv=TimeSeriesSplit(2), n_jobs=1)
```

The results are indexed by the training/testing periods and carry overall and per-market metrics, the latter labelled by the
market base names:

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

The `score` method returns the annual Sharpe ratio of the value bets predicted for the given data, which is convenient as an
optimization objective:

```python
sharpe_ratio = bettor.score(X_train, Y_train, O_train)
```

## Odds comparison bettor

The [`OddsComparisonBettor`][sportsbet.evaluation.OddsComparisonBettor] derives its probabilities directly from the odds, averaging
the `odds_types` providers and subtracting `alpha`. It therefore requires the odds matrix `O` at fit time as well:

```python
from sportsbet.evaluation import OddsComparisonBettor
odds_bettor = OddsComparisonBettor(odds_types=['market_average'], alpha=0.03, betting_markets=['home_win', 'draw', 'away_win'])
odds_bettor.fit(X_train, Y_train, O_train)
value_bets = odds_bettor.bet(X_fix, O_fix)
assert value_bets.shape == (1, 3)
```

## Hyperparameter search

[`BettorGridSearchCV`][sportsbet.evaluation.BettorGridSearchCV] tunes a bettor's parameters by cross-validated grid search,
mirroring [scikit-learn]'s [GridSearchCV]. It wraps a bettor as `estimator`, searches the values in `param_grid` (keys are the
bettor's parameter names) and, like the underlying bettor, exposes `fit`, `predict`, `predict_proba`, `bet` and `score`. It also
accepts the usual scikit-learn search parameters — `scoring`, `n_jobs`, `refit`, `cv`, `verbose` — with `cv` defaulting to a
[TimeSeriesSplit]. Once wrapped, it is used like any other bettor, for example inside `backtest`:

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

A fitted bettor can be saved to disk and reloaded later with `load_bettor`:

```python
import tempfile
from pathlib import Path
from sportsbet.evaluation import save_bettor, load_bettor
path = str(Path(tempfile.mkdtemp()) / 'bettor.pkl')
save_bettor(bettor, path)
reloaded = load_bettor(path)
assert reloaded.predict(X_fix).shape == (1, 3)
```
