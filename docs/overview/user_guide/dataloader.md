[ParameterGrid]: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html>
[pandas DataFrame]: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>
[pandas DateTimeIndex]: <https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html>
[pandas Timedelta]: <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>
[pandera]: <https://pandera.readthedocs.io>

# Dataloader

This section presents the dataloader object in details. The available dataloaders are the following:

- [`DummySoccerDataLoader`][sportsbet.datasets.DummySoccerDataLoader]: Soccer data loader with bundled sample data, for offline testing and the examples below.
- [`SoccerDataLoader`][sportsbet.datasets.SoccerDataLoader]: Soccer data loader that downloads historical and fixtures data.

We aim to include in the future more dataloaders for various sports and betting markets:

- Basketball
- NFL
- Hockey

## The event-snapshot data model

Internally the data are stored in long format as event *snapshots*. Each row is a single match observed at a single moment,
identified by two columns:

- `event_status`: the phase of the match, one of `'preplay'` (before kick-off), `'inplay'` (while the match is running) or
  `'postplay'` (final result).
- `event_time`: a [pandas Timedelta] measuring the elapsed time from kick-off, e.g. `pd.Timedelta('30min')`. It is `0min` for
  `preplay` and `postplay` snapshots.

The snapshots are validated with [pandera] schemas that describe the statistics and the odds columns. This moment-aware model is
what enables both pre-match and in-play predictions from a single interface: you pick a *target moment* and the dataloader turns
every earlier snapshot into features and the target-moment outcome into labels.

All the examples in this section use the offline [`DummySoccerDataLoader`][sportsbet.datasets.DummySoccerDataLoader], so they run
without any network access.

## Initialization

A dataloader is initialized with the parameter `param_grid` that selects the training data to extract. Indirectly this parameter
also affects the extracted fixtures data since dataloaders ensure that these two are in correspondence i.e. input and odds
matrices of training and fixtures data have the same columns.

### Available parameters

The available parameters and their values are provided from the class method `get_all_params`:

```python
from sportsbet.datasets import DummySoccerDataLoader
assert DummySoccerDataLoader.get_all_params() == [
    {'division': 1, 'league': 'England', 'year': 2025},
    {'division': 1, 'league': 'Spain', 'year': 2025}
]
```

Similarly, for [`SoccerDataLoader`][sportsbet.datasets.SoccerDataLoader]:

```python
from sportsbet.datasets import SoccerDataLoader
assert SoccerDataLoader.get_all_params()[:3] == [
    {'division': 1, 'league': 'England', 'year': 2018},
    {'division': 1, 'league': 'England', 'year': 2019},
    {'division': 1, 'league': 'England', 'year': 2020}
]
```

### Selection of parameters

The parameter `param_grid` has the same usage as the initialization parameter of scikit-learn's [ParameterGrid]. The default value
of `param_grid` is `None` and corresponds to the selection of all training data i.e. all leagues, years and divisions. If
`param_grid` is provided, then it should be a dictionary or a list of dictionaries with the above keys and values as lists.

For example, using the [`DummySoccerDataLoader`][sportsbet.datasets.DummySoccerDataLoader] and selecting only the English league:

```python
from sportsbet.datasets import DummySoccerDataLoader
dataloader = DummySoccerDataLoader(param_grid={'league': ['England']})
```

Once the dataloader is initialized, the training and fixtures data can be extracted.

## Column-naming grammar

The extracted `X`, `Y` and `O` matrices are wide tables whose columns encode the moment they refer to. There are four kinds of
columns, all using a fixed double-underscore (`__`) delimiter, with event times rendered as whole minutes (`{n}min`):

- **Fixed (time-invariant) features and identity**: a bare name, e.g. `league`, `home_team`, `home_points_avg`.
- **Time-varying features**: `{col}__{event_status}__{event_time}`, e.g. `home_goals__inplay__30min`.
- **Odds**: `{provider}__{market}__{event_status}__{event_time}`, e.g. `market_average__home_win__preplay__0min`.
- **Targets (`Y`)**: `{market}__{target_event_status}__{target_event_time}`, e.g. `home_win__postplay__0min`.

The supported betting markets are `home_win`, `draw`, `away_win`, `over_2.5` and `under_2.5`.

## Training data

The training data is a tuple of the input matrix `X_train`, the multi-output targets `Y_train` and the odds' matrix `O_train`. You
extract the training data using the method `extract_train_data`. All of its parameters are keyword-only:

- `drop_na_thres`: threshold in the range `[0.0, 1.0]` controlling how aggressively feature columns with missing values are dropped.
- `odds_type`: the provider used for the odds' matrix `O_train`.
- `target_event_status` and `target_event_time`: the target moment to predict.

We use the following dataloader as an example:

```python
from sportsbet.datasets import DummySoccerDataLoader
dataloader = DummySoccerDataLoader(param_grid={'league': ['England']})
```

### The `drop_na_thres` parameter

Parameter `drop_na_thres` adjusts the threshold of a column with missing values to be removed from the input matrix `X_train`. It
takes values in the range `[0.0, 1.0]`. This parameter is included for convenience since historical data often come with columns
that have many missing values, therefore their presence does not enhance the predictive power of models.

If we set `drop_na_thres=0.0` then all columns are kept:

```python
X_train, *_ = dataloader.extract_train_data(drop_na_thres=0.0, odds_type='market_average')
assert len(X_train.columns) == 28
```

The sample data have no missing feature values, so raising the threshold to `1.0` keeps the same columns here:

```python
X_train, *_ = dataloader.extract_train_data(drop_na_thres=1.0, odds_type='market_average')
assert len(X_train.columns) == 28
```

### The `odds_type` parameter

Parameter `odds_type` selects the provider that will be used for the odds' matrix `O_train`. You can get the available odds types
from the method `get_odds_types`:

```python
assert dataloader.get_odds_types() == ['market_average', 'market_maximum']
```

When `odds_type` is not provided, its default value is `None` and `O_train` has no columns:

```python
*_, O_train = dataloader.extract_train_data(drop_na_thres=0.0)
assert O_train.columns.tolist() == []
```

Selecting one of the above odds types returns the corresponding per-provider odds columns:

```python
X_train, _, O_train = dataloader.extract_train_data(drop_na_thres=0.0, odds_type='market_average')
assert all(col.startswith('market_average__') for col in O_train.columns)
assert 'market_average__home_win__preplay__0min' in O_train.columns.tolist()
```

### The target moment

By default `extract_train_data` predicts the final (`postplay`) outcome, so every earlier `preplay` and `inplay` snapshot becomes
a feature:

```python
import pandas as pd
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')
assert Y_train.columns.tolist() == [
    'home_win__postplay__0min',
    'draw__postplay__0min',
    'away_win__postplay__0min',
    'over_2.5__postplay__0min',
    'under_2.5__postplay__0min'
]
```

To predict an in-play moment, set `target_event_status='inplay'` and `target_event_time` to a [pandas Timedelta]. Only snapshots
strictly before the target moment are used as features, so the target moment cannot leak into `X_train`:

```python
X_inplay, Y_inplay, O_inplay = dataloader.extract_train_data(
    odds_type='market_average',
    target_event_status='inplay',
    target_event_time=pd.Timedelta('60min'),
)
assert Y_inplay.columns.tolist() == [
    'home_win__inplay__60min',
    'draw__inplay__60min',
    'away_win__inplay__60min',
    'over_2.5__inplay__60min',
    'under_2.5__inplay__60min'
]
# Only the 30-minute in-play snapshot is available as a feature, not 60/90.
assert 'home_goals__inplay__30min' in X_inplay.columns.tolist()
assert 'home_goals__inplay__60min' not in X_inplay.columns.tolist()
```

### Unsupervised extraction

Passing `learning_type='unsupervised'` returns only features and odds; the targets `Y_train` are `None`:

```python
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average', learning_type='unsupervised')
assert Y_train is None
```

## Fixtures data

Once the training data are extracted, it is straightforward to extract the corresponding fixtures data using the method
`extract_fixtures_data`:

```python
from sportsbet.datasets import DummySoccerDataLoader
dataloader = DummySoccerDataLoader(param_grid={'league': ['England']})
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

!!! warning

    The `extract_train_data` method should be called before `extract_fixtures_data`, in order to fix the columns of the input and
    odds data.

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
from sportsbet.datasets import DummySoccerDataLoader
dataloader = DummySoccerDataLoader(param_grid={'league': ['England']})
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

A detailed description of the above tuples of data is provided below.

### X_train

`X_train` is the first component of the training data tuple. `X_train` is a [pandas DataFrame] that contains information known
before the target moment: the identity of the match (fixed columns) and any moment-aware feature snapshots that precede the
target. For the default `postplay` target, this includes the pre-match points averages and every in-play snapshot:

```python
assert X_train.columns.tolist() == [
    'league',
    'division',
    'year',
    'home_team',
    'away_team',
    'away_goals__inplay__30min',
    'away_goals__inplay__60min',
    'away_goals__inplay__90min',
    'away_points_avg',
    'away_win__inplay__30min',
    'away_win__inplay__60min',
    'away_win__inplay__90min',
    'draw__inplay__30min',
    'draw__inplay__60min',
    'draw__inplay__90min',
    'home_goals__inplay__30min',
    'home_goals__inplay__60min',
    'home_goals__inplay__90min',
    'home_points_avg',
    'home_win__inplay__30min',
    'home_win__inplay__60min',
    'home_win__inplay__90min',
    'over_2.5__inplay__30min',
    'over_2.5__inplay__60min',
    'over_2.5__inplay__90min',
    'under_2.5__inplay__30min',
    'under_2.5__inplay__60min',
    'under_2.5__inplay__90min'
]
```

The index of `X_train` is a [pandas DateTimeIndex] named `date` and the data are always sorted by date:

```python
import pandas as pd
assert isinstance(X_train.index, pd.DatetimeIndex)
assert X_train.index.name == 'date'
assert X_train.index.is_monotonic_increasing
```

### Y_train

`Y_train` is the second component of the training data tuple:

```python
assert Y_train.columns.tolist() == [
    'home_win__postplay__0min',
    'draw__postplay__0min',
    'away_win__postplay__0min',
    'over_2.5__postplay__0min',
    'under_2.5__postplay__0min'
]
```

`Y_train` is a [pandas DataFrame] that contains the outcomes evaluated at the target moment. Column names follow the target
grammar `f'{market}__{target_event_status}__{target_event_time}'`:

- `market`: any supported betting market like `home_win`, `over_2.5` or `draw`.
- `target_event_status`: `'postplay'` for the final result or `'inplay'` for an in-play moment.
- `target_event_time`: the target time rendered as whole minutes, e.g. `0min` for `postplay` or `60min` for a 60-minute in-play
  target.

The entries of `Y_train` show whether an outcome of a betting event is `True` or `False`. The three components `X_train`,
`Y_train` and `O_train` share the same `date` index and the same rows.

### O_train

`O_train` is the last component of the training data tuple:

```python
assert O_train.columns.tolist() == [
    'market_average__away_win__inplay__30min',
    'market_average__away_win__inplay__60min',
    'market_average__away_win__inplay__90min',
    'market_average__away_win__preplay__0min',
    'market_average__draw__inplay__30min',
    'market_average__draw__inplay__60min',
    'market_average__draw__inplay__90min',
    'market_average__draw__preplay__0min',
    'market_average__home_win__inplay__30min',
    'market_average__home_win__inplay__60min',
    'market_average__home_win__inplay__90min',
    'market_average__home_win__preplay__0min',
    'market_average__over_2.5__inplay__30min',
    'market_average__over_2.5__inplay__60min',
    'market_average__over_2.5__inplay__90min',
    'market_average__over_2.5__preplay__0min',
    'market_average__under_2.5__inplay__30min',
    'market_average__under_2.5__inplay__60min',
    'market_average__under_2.5__inplay__90min',
    'market_average__under_2.5__preplay__0min'
]
```

`O_train` is a [pandas DataFrame] that contains the odds for various betting markets and moments. Column names follow the odds
grammar `f'{provider}__{market}__{event_status}__{event_time}'`:

- `provider`: the odds type selected through `odds_type`, one of the values returned by `get_odds_types`.
- `market`: any supported betting market.
- `event_status` and `event_time`: the snapshot the odds refer to.

The entries of `O_train` are the odd values of betting events and, depending on the data source, it may contain missing values.
`Y_train` and `O_train` share the same `date` index and rows as `X_train`. The bettor objects select, for each target market, the
odds column of the latest available snapshot, so `Y_train` and `O_train` stay aligned.

### X_fix

`X_fix` is the first component of the fixtures data tuple. It is a [pandas DataFrame] that contains information known before the
target moment. The features of `X_fix` are identical to the features of `X_train`:

```python
assert X_train.columns.tolist() == X_fix.columns.tolist()
```

`X_fix` contains the latest fixtures i.e. matches whose target-moment outcome is not yet known.

### Y_fix

`Y_fix` is always equal to `None` since the output of betting events for fixtures data is not known:

```python
assert Y_fix is None
```

### O_fix

`O_fix` is the last component of the fixtures data tuple. It is a [pandas DataFrame] that contains the odds for various betting
markets. The features of `O_fix` are identical to the features of `O_train`:

```python
assert O_train.columns.tolist() == O_fix.columns.tolist()
```

## Saving and loading

A dataloader can be saved to disk and reloaded later with `load_dataloader`, preserving the selected parameters and any extracted
column layout:

```python
import tempfile
from pathlib import Path
from sportsbet.datasets import DummySoccerDataLoader, load_dataloader
dataloader = DummySoccerDataLoader(param_grid={'league': ['England']})
dataloader.extract_train_data(odds_type='market_average')
path = str(Path(tempfile.mkdtemp()) / 'dataloader.pkl')
dataloader.save(path)
reloaded = load_dataloader(path)
assert reloaded.get_all_params() == dataloader.get_all_params()
```
