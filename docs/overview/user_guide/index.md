[pandas]: <https://pandas.pydata.org>
[scikit-learn]: <https://scikit-learn.org>

# User guide

The goal of the `sports-betting` package is to provide various tools to extract sports betting data and create predictive models.
It integrates with other well-known Python libraries like [pandas] and [scikit-learn]. The basic objects that
`sports-betting` provides are dataloaders and bettors.

## Dataloader

Sports betting datasets usually come in a format not suitable for modelling. The dataloader object provides methods to extract the
data in a consistent format that makes it easy to create predictive models. We initialize the
[`SoccerDataLoader`][sportsbet.datasets.SoccerDataLoader] for the Italian, Spanish leagues and years 2023, 2024:

```python
from sportsbet.datasets import SoccerDataLoader
dataloader = SoccerDataLoader(param_grid={'league': ['Italy', 'Spain'], 'year': [2023, 2024]})
```

The next step is to extract the training data, including the market maximum odds:

```python
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum')
```

Finally, the corresponding fixtures data are easily extracted:

```python
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

## Bettor

Once the training and fixtures data are available, bettor objects provide an easy way to evaluate a model and get predictions on
upcoming betting events:

We initialize a [`ClassifierBettor`][sportsbet.evaluation.ClassifierBettor] using a [scikit-learn]'s `KNeighborsClassifier`:

```python
from sportsbet.evaluation import ClassifierBettor, backtest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
bettor = ClassifierBettor(classifier=make_pipeline(SimpleImputer(), KNeighborsClassifier()))
```

For the backtesting of the bettor's performance we use only the historical data and numerical features:

```python
num_cols = X_train.columns[['float' in col_type.name for col_type in X_train.dtypes]]
backtest(bettor, X_train[num_cols], Y_train, O_train)
```

Similarly to get the value bets for upcoming betting events, we use the fixtures data:

```python
bettor.fit(X_train[num_cols], Y_train)
value_bets = bettor.bet(X_fix[num_cols], O_fix)
```
