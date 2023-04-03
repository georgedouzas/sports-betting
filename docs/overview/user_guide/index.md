[pandas]: <https://pandas.pydata.org>
[scikit-learn]: <https://scikit-learn.org>
[vectorbt]: <https://vectorbt.pro>
[Football-Data.co.uk]: <https://www.football-data.co.uk/data.php>
[FiveThirtyEight]: <https://github.com/fivethirtyeight/data/tree/master/soccer-spi>

# User guide

The goal of the `sports-betting` package is to provide various tools to extract sports betting data and create predictive models.
It integrates with other well-known Python libraries like [pandas], [scikit-learn] and [vectorbt]. The basic objects that
`sports-betting` provides are dataloaders and bettors.

## Dataloader

Sports betting datasets usually come in a format not suitable for modelling. The dataloader object provides methods to extract the
data in a consistent format that makes it easy to create predictive models. We initialize the
[`SoccerDataLoader`][sportsbet.datasets.SoccerDataLoader] for the Italian, Spanish leagues and years 2019, 2020:

```python
from sportsbet.datasets import SoccerDataLoader
dataloader = SoccerDataLoader(param_grid={'league': ['Italy', 'Spain'], 'year': [2019, 2020]})
```

This dataloader includes soccer data, extracted from [Football-Data.co.uk] and [FiveThirtyEight].

The next step is to extract the training data, including the `'interwetten'` bookmaker:

```python
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='interwetten')
```

Finally, the corresponding fixtures data are easily extracted:

```python
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

## Bettor

Once the training and fixtures data are available, bettor objects provide an easy way to evaluate a model and get predictions on
upcoming betting events:

We initialize a [`ClassifierBettor`][sportsbet.evaluation.ClassifierBettor] using a [scikit-learn]'s `NearestNeighbors` classfier:

```python
from sportsbet.evaluation import ClassifierBettor
from sklearn.neighbors import NearestNeighbors
bettor = ClassifierBettor(NearestNeighbors())
```

For the backtesting of the bettor's performance we use only the historical data:

```python
bettor.backtest(X_train, Y_train, O_train)
```

Similarly to get the value bets for upcoming betting events, we use the fixtures data:

```python
value_bets = bettor.bet(X_fix, O_fix)
```
