[ParameterGrid]: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html>
[pandas DataFrame]: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>
[pandas DateTimeIndex]: <https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html>

# Dataloader

This section presents the dataloader object in details. The available dataloaders are the following:

- [`DummySoccerDataLoader`][sportsbet.datasets.DummySoccerDataLoader]: Soccer data loader with dummy data, just for testing.
- [`SoccerDataLoader`][sportsbet.datasets.SoccerDataLoader]: Soccer data loader.

We aim to include in the future more dataloaders for various sports and betting markets:

- Basketball
- NFL
- Hockey

## Initialization

A dataloader is initialized with the parameter `param_grid` that selects the training data to extract. Indirectly this parameter
also affects the extracted fixtures data since dataloaders ensure that these two are in correspondence i.e. input and odds
matrices of training and fixtures data have the same columns.

### Available parameters

The available parameters and their values are provided from the class method `get_all_params`:

```python
from sportsbet.datasets import DummySoccerDataLoader
assert DummySoccerDataLoader.get_all_params() == [
    {'division': 1, 'year': 1998},
    {'division': 1, 'league': 'France', 'year': 2000},
    {'division': 1, 'league': 'France', 'year': 2001},
    {'division': 1, 'league': 'Greece', 'year': 2017},
    {'division': 1, 'league': 'Greece', 'year': 2019},
    {'division': 1, 'league': 'Spain', 'year': 1997},
    {'division': 2, 'league': 'England', 'year': 1997},
    {'division': 2, 'league': 'Spain', 'year': 1999},
    {'division': 3, 'league': 'England', 'year': 1998}
]
```

Similarly, for [`SoccerDataLoader`][sportsbet.datasets.SoccerDataLoader]:

```python
from sportsbet.datasets import SoccerDataLoader
assert SoccerDataLoader.get_all_params() == [
    {'division': 1, 'league': 'Argentina', 'year': 2018},
    {'division': 1, 'league': 'Argentina', 'year': 2019},
    {'division': 1, 'league': 'Argentina', 'year': 2020},
    ...,
    {'division': 5,  'league': 'England', 'year': 2022},
    {'division': 5, 'league': 'England', 'year': 2023},
    {'division': 5, 'league': 'England', 'year': 2024}
]
```

### Selection of parameters

The parameter `param_grid` has the same usage as the initialization parameter of scikit-learn's [ParameterGrid]. The default value
of `param_grid` is `None` and corresponds to the selection of all training data i.e. all leagues, years and divisions. If
`param_grid` is provided, then it should be a list of dictionaries with the above keys and values as lists.

For example, if we use the [`SoccerDataLoader`][sportsbet.datasets.SoccerDataLoader] and include data for the Spanish and Italian
leagues of first division and 2018-2020 years, as well as data for the French league of all divisions for 2020-2021 years, we will
use the following `param_grid` and dataloader:

```python
from sportsbet.datasets import SoccerDataLoader
param_grid = [
    {'division': [1], 'league': ['Spain', 'Italy'], 'year': [2018, 2019, 2020]},
    {'league': ['France'], 'year': [2020, 2021]} 
]
dataloader = SoccerDataLoader(param_grid=param_grid)
```

Once the dataloader is initialized, the training and fixtures data can be extracted.

## Training data

The training data is a tuple of the input matrix `X_train`, the multi-output targets `Y_train` and the odds' matrix `O_train`. You
can extract the training data using the method `extract_train_data` that accepts the parameters `drop_na_thres` and `odds_type`.
Both parameters are important, therefore we discuss briefly their usage. We will use the following dataloader as an example:

```python
from sportsbet.datasets import SoccerDataLoader
param_grid = [
    {'division': [1], 'league': ['Greece'], 'year': [2019, 2020]}
]
dataloader = SoccerDataLoader(param_grid=param_grid)
```

- Parameter `drop_na_thres` adjusts the threshold of a column with missing values to be removed from the input matrix `X_train`. It
takes values in the range [0.0, 1.0]. This parameter is included for convenience since historical data often come with columns
that have many missing values, therefore their presence does not enhance the predictive power of models.

    If we set `drop_na_thres=0` then all columns are kept:

    ```python
    X_train, *_ = dataloader.extract_train_data()
    assert len(X_train.columns) == 39
    ```

    Similarly, if we set `drop_na_thres=1.0` then only columns with non-missing values are kept:

    ```python
    X_train, *_ = dataloader.extract_train_data(drop_na_thres=1.0)
    assert len(X_train.columns) == 5
    ```

- Parameter `odds_type` selects the type of odds that will be used for the odds' matrix `O_train`. It also affects the columns of
the multi-output targets `Y_train` since there is a match between `Y_train` and `O_train` columns as explained below. You can get
the available odds types from the method `get_odds_types`:

    ```python
    assert dataloader.get_odds_types() == ['market_average', 'market_maximum']
    ```

    When `odds_type` is not provided, its default value is `None` and `O_train` is None:

    ```python
    *_, O_train, = dataloader.extract_train_data(drop_na_thres=0.5)
    assert O_train is None
    ```

    Selecting one of the above odds types, returns the corresponding data:

    ```python
    X_train, _, O_train, = dataloader.extract_train_data(drop_na_thres=0.5, odds_type='market_average')
    assert O_train.columns.tolist() == [
        'odds__market_average__home_win__full_time_goals',
        'odds__market_average__draw__full_time_goals',
        'odds__market_average__away_win__full_time_goals',
        'odds__market_average__over_2.5__full_time_goals',
        'odds__market_average__under_2.5__full_time_goals'
    ]
    ```

    Notice that `odds_type` parameter affects only the odds matrix. The input matrix `X_train` still contains information from all
    available odds types:

    ```python
    assert [col for col in X_train.columns if col.startswith('odds')] == [
        'odds__market_maximum__home_win__full_time_goals',
        'odds__market_maximum__draw__full_time_goals',
        'odds__market_maximum__away_win__full_time_goals',
        'odds__market_maximum__over_2.5__full_time_goals',
        'odds__market_maximum__under_2.5__full_time_goals',
        'odds__market_average__home_win__full_time_goals',
        'odds__market_average__draw__full_time_goals',
        'odds__market_average__away_win__full_time_goals',
        'odds__market_average__over_2.5__full_time_goals',
        'odds__market_average__under_2.5__full_time_goals'
    ]
    ```

    This is because we use `O_train` for backtesting when bets are placed against a specific bookmaker, but the information from
    other bookmakers may still be useful for the predictive model, thus they are included in `X_train`.

## Fixtures data

Once the training data are extracted, it is straightforward to extract the corresponding fixtures data using the method
`extract_fixtures_data`:

```python
from sportsbet.datasets import SoccerDataLoader
param_grid = [
    {'division': [1], 'league': ['Spain', 'Italy'], 'year': [2018, 2019, 2020]},
]
dataloader = SoccerDataLoader(param_grid=param_grid)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

The method accepts no parameters and the extracted fixtures input and odds matrices have the same columns as the latest extracted
input and odds matrices for the training data:

```python
assert X_train.columns.tolist() == X_fix.columns.tolist()
assert O_train.columns.tolist() == O_fix.columns.tolist()
```

Since we are extracting the fixtures data, there is no target matrix:

```python
assert Y_fix is None
```

## Description of data

As we have seen above, the extracted data are the following:

- Training: `(X_train, Y_train, O_train)`
- Fixtures: `(X_fix, None, O_fix)`

As an example we use the following data:

```python
from sportsbet.datasets import SoccerDataLoader
param_grid = {'league': ['England'], 'year': [2021]}
dataloader = SoccerDataLoader(param_grid=param_grid)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum')
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

A detailed description of the above tuples of data is provided below.

### X_train

`X_train` is the first component of the training data tuple. `X_train` is a [pandas DataFrame] that contains information known
before the start of the betting event like the date, the names of the opponents, features related to the past performance of the
opponents and any other information useful for predictive modelling:

```python
assert X_train.columns.tolist() == [
    'league',
    'division',
    'year',
    'home_team',
    'away_team',
    'odds__market_maximum__home_win__full_time_goals',
    'odds__market_maximum__draw__full_time_goals',
    'odds__market_maximum__away_win__full_time_goals',
    'odds__market_maximum__over_2.5__full_time_goals',
    'odds__market_maximum__under_2.5__full_time_goals',
    'odds__market_average__home_win__full_time_goals',
    'odds__market_average__draw__full_time_goals',
    'odds__market_average__away_win__full_time_goals',
    'odds__market_average__over_2.5__full_time_goals',
    'odds__market_average__under_2.5__full_time_goals',
    'home__points__avg',
    'home__adj_points__avg',
    'home__goals_for__avg',
    'home__goals_against__avg',
    'home__adj_goals_for__avg',
    'home__adj_goals_against__avg',
    'home__points__latest_avg',
    'home__adj_points__latest_avg',
    'home__goals_for__latest_avg',
    'home__goals_against__latest_avg',
    'home__adj_goals_for__latest_avg',
    'home__adj_goals_against__latest_avg',
    'away__points__avg',
    'away__adj_points__avg',
    'away__goals_for__avg',
    'away__goals_against__avg',
    'away__adj_goals_for__avg',
    'away__adj_goals_against__avg',
    'away__points__latest_avg',
    'away__adj_points__latest_avg',
    'away__goals_for__latest_avg',
    'away__goals_against__latest_avg',
    'away__adj_goals_for__latest_avg',
    'away__adj_goals_against__latest_avg'
]
```

It may also include odds data as shown above. The index of `X_train` is a [pandas DatetimeIndex] and the data are always sorted by
date:

```python
assert X_train.index.tolist() == [
    Timestamp('2020-09-11 00:00:00'),
    Timestamp('2020-09-12 00:00:00'),
    ...,
    Timestamp('2021-05-23 00:00:00'),
    Timestamp('2021-05-23 00:00:00')
]
```

### Y_train

`Y_train` is the second component of the training data tuple:

```python
assert Y_train.columns.tolist() == [
    'output__home_win__full_time_goals',
    'output__draw__full_time_goals',
    'output__away_win__full_time_goals',
    'output__over_2.5__full_time_goals',
    'output__under_2.5__full_time_goals'
]
```

`Y_train` is a [pandas DataFrame] that contains information known after the end of the betting event like goals or points scored,
fouls committed etc. Column names follow a naming convention of the form `f'output__{betting_market}__{target}'`:

- `betting_market`: Any supported betting market like home win, over 2.5, draw, home points etc.
- `target`: The outcome that was used to extract the targets like `'full_time_goals'`, `'half_time_goals'`, `'full_time_points'`
  etc.

The entries of `Y_train` show whether an outcome of a betting event is `True` or `False`. In order to make
the data suitable for modelling, `Y_train` does not contain any missing values i.e. rows of raw data that contain any missing
values are removed. This last step also includes `X_train` and `O_train`: Their corresponding rows are removed to match `Y_train`.

### O_train

`O_train` is the last component of the training data tuple:

```python
assert O_train.columns.tolist() == [
    'odds__market_maximum__home_win__full_time_goals',
    'odds__market_maximum__draw__full_time_goals',
    'odds__market_maximum__away_win__full_time_goals',
    'odds__market_maximum__over_2.5__full_time_goals',
    'odds__market_maximum__under_2.5__full_time_goals'
]
```

`O_train` is a [pandas DataFrame] that contains information related to the odds for various betting markets. Column names follow a
naming convention of the form `f'odds__{bookmaker}__{betting_market}__{target}'`:

- `bookmaker`: Any supported bookmaker or aggregation of bookmakers return by the method `get_odds_types`.
- `betting_market`: Similar to `Y_train`.
- `target`: Similar to `Y_train`.

The entries of `O_train` are the odd values of betting events and, depending on the data source, it may contain missing values.
`Y_train` and `O_train` columns match, i.e. `Y_train` and `O_train` have the same shape and
`f'output__{betting_market}__{target}'` column of `Y_train` is at the same position as the
`f'odds__{bookmaker}__{betting_market}__{target}'` column of `O_train`. The correspondence is clear in the examples above.

### X_fix

`X_fix` is the first component of the fixtures data tuple. It is a [pandas DataFrame] that contains information known before the
start of the betting event. The features of `X_fix` are identical to the features of `X_train`:

```python
assert X_train.columns.tolist() == X_fix.columns.tolist()
```

`X_fix` is not affected by the initialization parameter `param_grid` of the dataloader i.e. it contains the latest fixtures for
every league, division or any other parameter, even if they are not included in the training data.

### Y_fix

`Y_fix` is always equal to `None` since the output of betting events for fixtures data is not known:

```python
assert Y_fix is None
```

### O_fix

`O_fix` is the last component of the fixtures data tuple. It is a [pandas DataFrame] that contains information related to the odds for various betting markets. The features of `O_fix`
are identical to the features of `O_train`:

```python
assert O_train.columns.tolist() == O_fix.columns.tolist()
```
