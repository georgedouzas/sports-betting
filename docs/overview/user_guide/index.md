[pandas]: <https://pandas.pydata.org>
[scikit-learn]: <https://scikit-learn.org>

# User guide

`sports-betting` extracts sports betting data and trains predictive models on it. You need to know two objects. A
dataloader gets the data. A bettor bets on it.

## Dataloader

Betting data rarely comes in a shape you can model directly. The dataloader extracts it in a consistent format. Here is
a dataloader for the Italian and Spanish leagues, seasons 2023 and 2024.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import FootballDataOdds, FootballDataStats
dataloader = DataLoader(
    param_grid={'league': ['Italy', 'Spain'], 'year': [2023, 2024]},
    stats=FootballDataStats(),
    odds=FootballDataOdds(),
)
```

Extract the training data with the market maximum odds.

```python
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum')
```

Then extract the fixtures data.

```python
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

## Bettor

You now have the training and fixtures data. A bettor evaluates a model and predicts the value bets of the upcoming
matches. Here is a [`ClassifierBettor`][sportsbet.evaluation.ClassifierBettor] around a [scikit-learn]
`KNeighborsClassifier`.

```python
from sportsbet.evaluation import ClassifierBettor, backtest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
bettor = ClassifierBettor(classifier=make_pipeline(SimpleImputer(), KNeighborsClassifier()))
```

Backtest it on the historical data, using the numerical features.

```python
num_cols = X_train.columns[['float' in col_type.name for col_type in X_train.dtypes]]
backtest(bettor, X_train[num_cols], Y_train, O_train)
```

Fit it and predict the value bets of the fixtures.

```python
bettor.fit(X_train[num_cols], Y_train)
value_bets = bettor.bet(X_fix[num_cols], O_fix)
```
