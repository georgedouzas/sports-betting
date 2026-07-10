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

The snapshots are stored as two long tables — a `stats` table with the match statistics and an `odds` table with one row per
odds `provider` — sharing the same identity and event columns. This moment-aware model is what enables both pre-match and in-play
predictions from a single interface: you pick a *target moment* and the dataloader turns every earlier snapshot into features and
the target-moment outcome into labels.

### Everything is derived from the data

Nothing about the feed is hardcoded. When a dataloader reads its snapshots it *derives* the whole layout from the data itself:

- the available odds **providers** (from the `provider` column of the `odds` table),
- the betting **markets** (the value columns of the `odds` table, e.g. `home_win`, `over_2.5`),
- the **features** (the remaining value columns of the `stats` table), and
- for every value column, whether it is **fixed** (constant within a match, so it keeps a bare name) or **time-varying** (so it
  is expanded per moment), and at which `event_status` it actually carries values.

Those derived roles are captured in [pandera] schemas that are *built from the data* and used to validate it. A practical
consequence is that you can feed the dataloader *any* set of columns that follows this long format and it will adapt — see
[Consuming your own data](#consuming-your-own-data).

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
params = SoccerDataLoader.get_all_params()
# The available combinations are discovered from the feed, so only the
# league/division/year that actually exist are ever offered.
assert {'division': 1, 'league': 'England', 'year': 2024} in params
assert all({'league', 'division', 'year'} == set(combination) for combination in params)
```

### Selection of parameters

The parameter `param_grid` has the same usage as the initialization parameter of scikit-learn's [ParameterGrid]. It accepts:

- `None` (the default) — selects **all** available data, i.e. every combination returned by `get_all_params`.
- a **dictionary** whose keys are a subset of `'league'`, `'division'`, `'year'` and whose values are lists.
- a **list of dictionaries**, to select several disjoint groups at once.

Only combinations that actually exist in the feed are selected, and any dimension you omit defaults to all of its available values
— so an invalid combination (for example a division a league does not have) is never requested.

Selecting a single league, letting `division` and `year` default to all their available values:

```python
from sportsbet.datasets import DummySoccerDataLoader
dataloader = DummySoccerDataLoader(param_grid={'league': ['England']})
```

Selecting explicit combinations with a dictionary of several keys:

```python
dataloader = DummySoccerDataLoader(param_grid={'league': ['England', 'Spain'], 'division': [1], 'year': [2025]})
```

Selecting two disjoint groups with a list of dictionaries:

```python
dataloader = DummySoccerDataLoader(param_grid=[{'league': ['England']}, {'league': ['Spain']}])
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
- `input_event_status` and `input_event_time`: the latest snapshot to keep as a feature (the *input horizon*).
- `learning_type`: `'supervised'` (the default) or `'unsupervised'`, in which case `Y_train` is `None`.

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

### The input horizon

By default every snapshot before the target moment becomes a feature. Often you do not want all of them: to bet *before kick-off*
you can only rely on pre-match information, so half-time or other in-play snapshots must not enter the model. The
`input_event_status` and `input_event_time` parameters cap the features at a chosen moment — the *input horizon* — keeping only
snapshots up to and including it. For a pre-match model, set the horizon to `preplay`:

```python
X_pre, Y_pre, O_pre = dataloader.extract_train_data(
    odds_type='market_average',
    input_event_status='preplay',
    input_event_time=pd.Timedelta('0min'),
)
assert not [col for col in X_pre.columns if '__inplay__' in col]
```

To use information up to half-time only, set the horizon to `inplay` at 45 minutes; snapshots after it are dropped:

```python
X_ht, *_ = dataloader.extract_train_data(
    odds_type='market_average',
    input_event_status='inplay',
    input_event_time=pd.Timedelta('45min'),
)
assert all('__inplay__60min' not in col and '__inplay__90min' not in col for col in X_ht.columns)
```

The same horizon is applied to the fixtures data, so training and prediction always use the same feature set. This is central to
using the dataloader in practice — see [Sports betting in practice](in_practice.md).

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

## Consuming your own data

The extraction, grammar and moment-aware model described above are not tied to the bundled feed: three factory functions build a
dataloader straight from data you provide — [`from_snapshots`][sportsbet.datasets.from_snapshots],
[`from_dataframe`][sportsbet.datasets.from_dataframe] and [`from_csv`][sportsbet.datasets.from_csv]. Because the layout is
[derived from the data](#everything-is-derived-from-the-data), your columns only need to follow the long format — the providers,
markets, features and their fixed/time-varying roles are inferred for you.

### From long snapshots

If your data already follows the long format, pass the `stats` and `odds` tables to `from_snapshots`. Each row is a match at one
moment, tagged with `event_status` and `event_time` (whole minutes); a match with no resolvable result is treated as a fixture. The
`odds` table adds a `provider` column, and the markets it carries become the prediction targets. Here we build two finished matches
(with a half-time, `inplay`/`45min`, snapshot) and one upcoming fixture, deriving the market outcomes from the goals with the
[`market_outcomes`][sportsbet.datasets.market_outcomes] helper:

```python
import pandas as pd
from sportsbet.datasets import from_snapshots, market_outcomes

def snapshot(event_status, event_time, date, home, away, home_goals, away_goals, home_avg, away_avg):
    return dict(
        event_status=event_status, event_time=event_time, date=date, league='England', division=1, year=2025,
        home_team=home, away_team=away, home_goals=home_goals, away_goals=away_goals,
        home_points_avg=home_avg, away_points_avg=away_avg,
    )

stats = pd.DataFrame([
    snapshot('preplay', 0, '2024-08-16', 'Arsenal', 'Chelsea', None, None, 2.1, 1.5),
    snapshot('inplay', 45, '2024-08-16', 'Arsenal', 'Chelsea', 1, 0, None, None),
    snapshot('postplay', 0, '2024-08-16', 'Arsenal', 'Chelsea', 2, 0, None, None),
    snapshot('preplay', 0, '2024-08-23', 'Everton', 'Spurs', None, None, 1.2, 1.9),
    snapshot('inplay', 45, '2024-08-23', 'Everton', 'Spurs', 0, 1, None, None),
    snapshot('postplay', 0, '2024-08-23', 'Everton', 'Spurs', 1, 2, None, None),
    snapshot('preplay', 0, '2025-09-01', 'Liverpool', 'Wolves', None, None, 2.4, 1.0),  # upcoming fixture
])
markets = ['home_win', 'draw', 'away_win']
played = stats['home_goals'].notna()
stats.loc[played, markets] = market_outcomes(
    stats.loc[played, 'home_goals'], stats.loc[played, 'away_goals'], markets,
).to_numpy()

def quote(date, home, away, home_win, draw, away_win):
    return dict(
        event_status='preplay', event_time=0, date=date, league='England', division=1, year=2025,
        home_team=home, away_team=away, provider='market_average',
        home_win=home_win, draw=draw, away_win=away_win,
    )

odds = pd.DataFrame([
    quote('2024-08-16', 'Arsenal', 'Chelsea', 1.7, 3.6, 4.8),
    quote('2024-08-23', 'Everton', 'Spurs', 2.6, 3.3, 2.5),
    quote('2025-09-01', 'Liverpool', 'Wolves', 1.4, 4.5, 7.0),
])

dataloader = from_snapshots(stats, odds)
assert dataloader.get_odds_types() == ['market_average']
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')
assert Y_train.columns.tolist() == ['home_win__postplay__0min', 'draw__postplay__0min', 'away_win__postplay__0min']
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
assert list(zip(X_fix['home_team'], X_fix['away_team'])) == [('Liverpool', 'Wolves')]
```

### From a single-moment table

If instead you have one wide row per match, all observed at the *same* moment, use `from_dataframe` (or `from_csv` for a file) and
declare exactly what that moment is with `event_status` and `event_time` — nothing is assumed. Odds columns follow the
`{provider}__{market}` naming and are split out into the `odds` table automatically:

```python
import pandas as pd
from sportsbet.datasets import from_dataframe

upcoming = pd.DataFrame([{
    'date': '2025-09-01', 'league': 'England', 'division': 1, 'year': 2025,
    'home_team': 'Liverpool', 'away_team': 'Wolves', 'home_points_avg': 2.4, 'away_points_avg': 1.0,
    'market_average__home_win': 1.4, 'market_average__draw': 4.5, 'market_average__away_win': 7.0,
}])
dataloader = from_dataframe(upcoming, event_status='preplay', event_time=pd.Timedelta('0min'))
assert dataloader.get_odds_types() == ['market_average']
```

Since the whole frame is a single moment, this is a building block for one snapshot at a time: to build a full training set,
provide long snapshots through `from_snapshots` instead, or combine several single-moment frames.

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
