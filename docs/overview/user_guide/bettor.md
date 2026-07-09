[scikit-learn]: <https://scikit-learn.org>
[decision tree classifier]: <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>

# Bettor

This section presents the bettor object in details. The available bettors are the following:

- [`OddsComparisonBettor`][sportsbet.evaluation.OddsComparisonBettor]: Bettor based on comparison of odds.
- [`ClassifierBettor`][sportsbet.evaluation.ClassifierBettor]: Bettor based on a classifier.

## Initialization

Every bettor has its own initialization parameters. For example, the provided bettor
[`ClassifierBettor`][sportsbet.evaluation.ClassifierBettor] is initialized with the parameter `classifier`, that defines the
[scikit-learn] classifier to predict the probabilities of betting events.

All bettors also accept a `betting_markets` parameter that selects the markets to bet on. It is a list of market **base** names
such as `['home_win', 'draw', 'away_win']`, `['over_2.5', 'under_2.5']` etc. When it is `None`, all markets present in the targets
are used.

## Betting strategy

The essence of any betting strategy is to identify the value bets i.e. betting events where the bookmaker underestimates the
probability of the event. Of course, the true probability of the betting event is unknown, thus the comparison is between the
estimated probability of the bettor and the bookmaker: A bet is identified as 'value bet' when the bettor estimates a higher
probability for the betting event compared to the estimated probability of the bookmaker as derived from the odds.

## Implementation

The creation and evaluation of betting strategies is made via the bettor objects. Specifically, bettors implement the following public methods:

- `fit` that fits the model to any input data `X`, multi-output targets `Y` and optional odds `O`.
- `predict` that predicts the class labels of betting events.
- `predict_proba` that predicts the class probabilities of betting events.
- `bet` that returns the value bets.
- `score` that returns the annual Sharpe ratio of the predicted value bets.

The `backtest` function calculates various backtesting statistics for a bettor over historical data.

All the above methods are based on the implementation of the private methods `_fit` and `_predict_proba`. This provides a
flexibility on the type of betting models that can be defined. The `_fit` method allows learning from historical data. If the
betting model does not need it, for example an arbitrage bettor, then it can be omitted. The `_predict_proba` method predicts the
class probabilities of betting events. Again if the model identifies value bets in a way that is not based on class probabilities
then the implementation of `_predict_proba` can be trivial i.e. set the output to `1.0` for value bets and `0.0` otherwise.

For the rest of this section we use a classifier-based bettor built around [scikit-learn]'s [decision tree classifier]. The
extracted features include categorical columns (`league`, `home_team`, `away_team`) and columns with missing values, so we wrap the
classifier in a pipeline that encodes and imputes them:

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

## Model fit

The bettor is fitted to the training data `(X_train, Y_train)` via the `fit` method. This fitting procedure does not necessarily
require machine learning models but more generally means that the bettor extracts information from `(X_train, Y_train)` that will
be used when predictions are made. Fitting the model is very simple:

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

In order to understand the above predictions we extract the number of rows of the fixtures input matrix:

```python
n_rows, _ = X_fix.shape
assert n_rows == 1
```

We have a single upcoming betting event. The target columns of the training data show which markets are modelled:

```python
assert Y_train.columns.tolist() == [
    'home_win__postplay__0min',
    'draw__postplay__0min',
    'away_win__postplay__0min',
    'over_2.5__postplay__0min',
    'under_2.5__postplay__0min'
]
```

The bettor was initialized with `betting_markets=['home_win', 'draw', 'away_win']`, so the predictions have three columns, one per
selected market. Nevertheless, predicting the class labels is not useful since the value bets should be based not directly on them
but on the comparison of predicted probabilities to the `O_fix` matrix.

## Class probabilities predictions

Predicting positive class probabilities is also simple. There is one probability per selected market and, for mutually-exclusive
markets like `home_win`/`draw`/`away_win`, the probabilities are normalized to sum to one:

```python
probabilities = bettor.predict_proba(X_fix)
assert probabilities.shape == (1, 3)
assert abs(probabilities.sum(axis=1)[0] - 1.0) < 1e-6
```

## Backtesting

Backtesting the bettor's strategy requires the training data tuple `(X_train, Y_train, O_train)` to be used:

```python
from sportsbet.evaluation import backtest
backtesting_results = backtest(bettor, X_train, Y_train, O_train)
```

The backtesting results include information of the various training/testing periods and metrics. The per-market metrics are
labelled by the market base names:

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

## Value bets prediction

Similarly, the fitted bettor can be used to predict the value bets. The `bet` method returns one boolean column per selected
market, so we can combine these predictions with the identity columns of `X_fix`:

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

## Odds comparison bettor

The [`OddsComparisonBettor`][sportsbet.evaluation.OddsComparisonBettor] derives its probabilities directly from the odds, so it
requires the odds matrix `O` at fit time as well:

```python
from sportsbet.evaluation import OddsComparisonBettor
odds_bettor = OddsComparisonBettor(betting_markets=['home_win', 'draw', 'away_win'], alpha=0.03)
odds_bettor.fit(X_train, Y_train, O_train)
value_bets = odds_bettor.bet(X_fix, O_fix)
assert value_bets.shape == (1, 3)
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
