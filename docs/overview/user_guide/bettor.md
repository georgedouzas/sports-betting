[scikit-learn]: <https://scikit-learn.org>
[dummy classifier]: <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html>

# Bettor

This section presents the bettor object in details. The available bettors are the following:

- [`OddsComparisonBettor`][sportsbet.evaluation.ClassifierBettor]: Bettor based on comparison of average and maximum market odds.
- [`ClassifierBettor`][sportsbet.evaluation.ClassifierBettor]: Bettor based on a classifier.

## Initialization

Every bettor has its own initialization parameters. For example, the provided bettor
[`ClassifierBettor`][sportsbet.evaluation.ClassifierBettor] is initialized with the parameter `classifier`, that defines the
[scikit-learn] classifier to predict the probabilities of betting events.

## Betting strategy

The essence of any betting strategy is to identify the value bets i.e. betting events where the bookmaker underestimates the
probability of the event. Of course, the true probability of the betting event is unknown, thus the comparison is between the
estimated probability of the bettor and the bookmaker: A bet is identified as 'value bet' when the bettor estimates a higher
probability for the betting event compared to the estimated probability of the bookmaker as derived from the odds.

## Implementation

The creation and evaluation of betting strategies is made via the bettor objects. Specifically, bettors implement the following public methods:

- `fit` that fits the model to any input data `X` and multi-ouput targets `Y`. 
- `predict` that predicts the class labels of betting events.
- `predict_proba` that predicts the class probabilities of betting events.
- `backtest` that calculates various backtesting statistics.
- `bet` that returns the value bets.

All the above methods are based on the implementation of the private methods `_fit` and `_predict_proba`. This provides a
flexibility on the type of betting models that can be defined. The `_fit` method allows learning from historical data. If the
betting model does not need it, for example an arbitrage bettor, then it can be omitted. The `_predict_proba` method predicts the
class probabilities of betting events. Again if the model identifies value bets in a way that is not based on class probabilities
then the implementation of `_predict_proba` can be trivial i.e. set the output to `1.0` for value bets and `0.0` otherwise.

For the rest of the [Bettors] section we use a classifier-based bettor that selects [scikit-learn]'s [dummy classifier]:

```python
from sklearn.dummy import DummyClassifier
from sportsbet.evaluation import ClassifierBettor
bettor = ClassifierBettor(classifier=DummyClassifier())
```

We also use dummy training and fixtures data:

```python
from sportsbet.datasets import DummySoccerDataLoader
dataloader = DummySoccerDataLoader()
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='interwetten')
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

## Model fit

The bettor is fitted to the training data `(X_train, Y_train)` via the `fit` method. This fitting procedure does not necessarily
require machine learning models but more generally means that the bettor extracts information from `(X_train, Y_train)` that will
be used when predictions are made. Fitting the model is very simple:

```python
bettor.fit(X_train, Y_train)
```

## Class labels prediction

Once the model is fitted, predicting class labels is straightforward:

```python
assert bettor.predict(X_fix).tolist()  == [
    [False, False, False], 
    [False, False, False]
]
```

In order to understand the above predictions we extract the number of rows of the fixtures input matrix:

```python
n_rows, _ = X_fix.shape
assert n_rows == 2
```

We have two betting events. Similarly, we extract the output columns of the training data:

```python
assert Y_train.columns.tolist() == [
    'output__home_win__full_time_goals',
    'output__draw__full_time_goals',
    'output__away_win__full_time_goals'
]
```

Each betting event includes three betting markets:

- `home_win`
- `draw`
- `away win`

Therefore, the predictions have the correct shape. Nevertheless, predicting the class labels is not useful since the value bets
should be based not directly on them but on the comparison of predicted probabilities to the `O_fix` matrix.

## Class probabilities predictions

Predicting positive class probabilities is also simple:

```python
assert bettor.predict_proba(X_fix).tolist() == [
    [0.375, 0.25, 0.375],
    [0.375, 0.25, 0.375]
]    
```

Their interpretation is similar to the one in the previous section.

## Backtesting

Backtesting the bettor's strategy requires the training data tuple `(X_train, Y_train, O_train)` to be used:

```python
from sportsbet.evaluation import backtest
backtesting_results = backtest(bettor, X_train, Y_train, O_train)
```

The backtesting results include information of the various training/testing periods and metrics:

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
    'Number of bets (home_win__full_time_goals)',
    'Number of bets (draw__full_time_goals)',
    'Number of bets (away_win__full_time_goals)',
    'Yield percentage per bet (home_win__full_time_goals)',
    'Yield percentage per bet (draw__full_time_goals)',
    'Yield percentage per bet (away_win__full_time_goals)'
]
```

## Value bets prediction

Similarly, the fitted bettor can be used to predict the value bets. We can combine these predictions with `X_fix`:

```python
import pandas as pd
value_bets = pd.concat(
    [
        X_fix.reset_index()[['date', 'home_team', 'away_team']],
        pd.DataFrame(bettor.bet(X_fix, O_fix), columns=[col.split('__')[2] for col in O_fix.columns])
    ], axis=1
).set_index('date')
assert value_bets.columns.tolist() == ['home_team', 'away_team', 'home_win', 'draw', 'away_win']
assert value_bets.values.tolist() == [
    ['Barcelona', 'Real Madrid', True, False, False],
    ['Monaco', 'PSG', False, False, False]
]
```
