[ParameterGrid]: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html>
[pandas DataFrame]: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>
[pandas DateTimeIndex]: <https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html>
[pandas Timedelta]: <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>
[pandera]: <https://pandera.readthedocs.io>

# Dataloader

A [`DataLoader`][sportsbet.dataloaders.DataLoader] downloads historical and fixtures data from the sources you give it
and shapes it for modelling. This page covers how to select, extract and save that data.

Most examples use [`SampleSoccerStats`][sportsbet.sources.SampleSoccerStats] and
[`SampleSoccerOdds`][sportsbet.sources.SampleSoccerOdds], a real season frozen and shipped with the library, so they run
offline. A few use a live feed or a paid odds source, marked where they appear.

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

One `DataLoader` serves every sport. The sport comes from the statistics source, so a soccer source makes a soccer
dataloader and a basketball source a basketball one. To add a sport, write a source. See
[other sports and paid odds](#other-sports-and-paid-odds).

## Code, not data

The library ships the code that fetches the data and runs it on your machine. Bookmakers own their odds, so `DataLoader`
downloads [football-data.co.uk](https://www.football-data.co.uk) to your machine and transforms it there.

This has two consequences.

* Extracting the data downloads it. See [downloading the data](#downloading-the-data).
* You choose the sources. Statistics and odds are separate parameters, so free statistics pair with paid odds.

## The event-snapshot data model

The library stores the data in long format as event snapshots. Each row is one match at one moment, marked by two
columns.

* `event_status`: the phase of the match, `'preplay'`, `'inplay'` or `'postplay'`.
* `event_time`: a [pandas Timedelta] since kick off, for example `pd.Timedelta('30min')`. It is `0min` at `preplay` and
  `postplay`.

### Time is always UTC, and `date` is the kick-off

`date` is the kick-off instant in UTC. Every source converts its own local time at its boundary, so you always get UTC.
football-data.co.uk, for example, publishes every league in UK time, and the library resolves that before you see it.

This gives the data model its central rule.

```text
date + event_time  =  the wall-clock instant of the snapshot
```

So a moment has an address. "The odds at minute 45 of Arsenal vs Chelsea" is a real timestamp you can ask a provider
for. Older seasons carry no kick-off time. They fall back to midnight UTC and pair with the free feeds.

The snapshots live in two long tables, `stats` and `odds`. The two tables share their identity and event columns, and
`odds` has one row per `provider`. This moment-aware model produces both pre-match and in-play predictions from one
interface. You pick a target moment. Every earlier snapshot becomes a feature, and the target outcome becomes the
labels.

### Everything is derived from the data

A dataloader reads its snapshots and works out the layout from them.

* the odds providers, from the `provider` column,
* the markets, from the odds value columns such as `home_win` and `over_2.5`,
* the features, from the stats value columns,
* and for each value column, whether it is fixed within a match, so it keeps a bare name, or time varying, so it is
  expanded per moment, and where it carries values.

The library builds [pandera] schemas from the data and validates the data against them. So a source may publish any
columns in this long format, and the dataloader adapts. See [data of your own](#data-of-your-own).

## Initialization

You initialise a dataloader with `param_grid`. This selects the training data. It fixes the fixtures data too, since the
two keep the same columns.

### Available parameters

Ask the source what exists before you write a `param_grid`. The source holds the catalogue and decides what is
available.

```python
from sportsbet.sources import FootballDataStats

params = FootballDataStats().list_available_params()
# Only the league/division/year combinations the feed actually publishes are
# ever offered, so an invalid one can never be requested.
assert {'division': 1, 'league': 'England', 'year': 2024} in params
assert all({'league', 'division', 'year'} == set(combination) for combination in params)
```

`list_available_params` is an instance method, because what a source publishes depends on its configuration. A
credential may cover part of it. The source reads the catalogue fresh on each call, so a new season appears as soon as
the feed publishes it.

The dataloader offers the seasons that both your statistics and odds sources publish. It takes their intersection, so
every selected season has both.

### Selection of parameters

`param_grid` works like the argument of scikit-learn's [ParameterGrid]. It accepts:

* `None`, the default, every combination both sources publish.
* a dictionary whose keys are a subset of `'league'`, `'division'`, `'year'` and whose values are lists.
* a list of dictionaries, to select several groups at once.

The dataloader selects the combinations the feed publishes, and a dimension you omit takes all its values.

Select a single league and let `division` and `year` default to all their values:

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats
dataloader = DataLoader(
    param_grid={'league': ['England']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds()
)
```

Select explicit combinations with a dictionary of several keys:

```python
dataloader = DataLoader(
    param_grid={'league': ['England', 'Spain'], 'division': [1], 'year': [2024]},
    stats=SampleSoccerStats(),
    odds=SampleSoccerOdds(),
)
```

Select two separate groups with a list of dictionaries:

```python
dataloader = DataLoader(
    param_grid=[{'league': ['England']}, {'league': ['Spain']}],
    stats=SampleSoccerStats(),
    odds=SampleSoccerOdds(),
)
```

Once initialised, extract the data, which downloads it.

## Sources

A source is where the data comes from. `DataLoader` takes two sources. Each source carries its own settings, so a
source's configuration stays with the source.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import FootballDataOdds, FootballDataStats

dataloader = DataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2025]},
    stats=FootballDataStats(),
    odds=FootballDataOdds(),
)
```

You choose both sources, so you always know what you are modelling. Pass `stats` to say where the statistics come from.
Pass `odds` for markets to bet on. Omit `odds` and use `extract_exploration_data` for the features alone.

Statistics and odds are separate on purpose. The free feed carries pre-match closing odds, which are enough to backtest
a pre-match bet. A source with time-stamped prices backtests an in-play bet too. So free statistics with your own paid
odds is the realistic setup.

### Bringing your own odds

[`OddsApi`][sportsbet.sources.OddsApi] buys time-stamped prices from [The Odds API](https://the-odds-api.com) with your
key. This makes an in-play bet backtestable, because the odds are the ones on offer at the minute the bet would have
been placed.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import FootballDataStats, OddsApi

dataloader = DataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2025]},
    stats=FootballDataStats(),                                    # free
    odds=OddsApi(key_env='ODDS_API_KEY', markets=['h2h', 'totals'], regions=['eu']),   # yours
)
```

This feed is metered, so extract deliberately. Extracting downloads the seasons and buys their odds. Extract once and
[keep the result](#keeping-the-data) with `save`. The free statistics say when each match kicks off, so the feed buys
one snapshot per kick-off and moment, and it buys matches that kick off together once. What a request costs is between
you and the vendor.

Your key joins a request at the moment the request is made, so it stays on your machine and out of the data you save.

Two limits are worth knowing. The vendor's history begins in June 2020, and historical prices are a paid tier. The
dataloader offers the seasons both sources publish, so an `OddsApi` selection stays within what it can price.

### When two sources name a club differently

This is the most dangerous thing in the library, so read it carefully.

Two sources rarely spell a club the same way. The free feed says `Man United`, `Nott'm Forest`, `Wolves`. The odds
vendor says `Manchester United`, `Nottingham Forest`, `Wolverhampton Wanderers`. When a name fails to match, that game
has no odds. A missing odd does not look like an error. It looks like a slightly smaller dataset. The result is a
backtest that is clean, plausible and wrong.

So the dataloader reconciles statistics and odds from different sources. It compares the spellings and pairs the two
feeds. A name it cannot place is dropped, not guessed at.

Most of the time you do nothing. The dataloader pairs names within a league and season, where both sources hold the same
twenty clubs. Every name that could be confused is present on both sides and matches itself first. `Manchester City`
pairs with `Man City` because `Manchester United` and `Man United` are there too, and they claim each other.

Clubs are abbreviated by shortening their words, so the dataloader compares names by the prefixes their words share.
`Wolves` matches `Wolverhampton Wanderers`. `Everton` stays apart from `Liverpool`. The dataloader pairs a name when it
is clearly the best on the roster and clearly better than the next best. It leaves anything ambiguous out.

The dataloader drops a name it cannot place, and it does so quietly. A small alias fixes this. When you know the club a
leftover name means, you pass it, and the match keeps its odds.

```python
DataLoader(..., aliases={'Athletic Bilbao': 'Ath Bilbao'})
```

The reconciled odds take the statistics' identity, their kick-off and spelling, so the two tables line up. The free path
skips all of this, since statistics and odds come from the same row.

## Downloading the data

Extracting the data downloads it, in one step.

```python
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum')
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

`extract_train_data` downloads the selected seasons. `extract_fixtures_data` downloads the current data the upcoming
matches need. Each call downloads afresh, so the object carries the latest data, held as `stats_` and `odds_`.

The dataloader reads the catalogue scoped to the selection. So three selected leagues read three indexes, and a paid
feed prices only the matches you selected and those still to be played.

## Keeping the data

The dataloader is the store. After an extraction it holds the snapshots. So you keep the snapshots by keeping the
object.

```python
dataloader.save('england.pkl')

from sportsbet.dataloaders import load_dataloader
dataloader = load_dataloader('england.pkl')      # the data comes back with it
```

Extract once, save, and load to reuse the data. For a paid feed, this reuses what you paid for. You own the file. Where
it lives and how long it lasts is up to you. For fresh data, extract again.

## Column-naming grammar

The extracted `X`, `Y` and `O` matrices are wide tables whose columns encode the moment they refer to. There are four
kinds of column, all using a double underscore (`__`) delimiter, with event times written as whole minutes (`{n}min`).

* Fixed features and identity: a bare name, such as `league`, `home_team`, `home_points_avg`.
* Time varying features: `{col}__{event_status}__{event_time}`, such as `home_goals__inplay__30min`.
* Odds: `{provider}__{market}__{event_status}__{event_time}`, such as `market_average__home_win__preplay__0min`.
* Targets in `Y`: `{market}__{target_event_status}__{target_event_time}`, such as `home_win__postplay__0min`.

The supported betting markets are `home_win`, `draw`, `away_win`, `over_2.5` and `under_2.5`.

## Training data

The training data is `(X_train, Y_train, O_train)`: the input matrix, the multi output targets and the odds. Extract it
with `extract_train_data`, whose parameters are keyword only.

* `drop_na_thres`: how aggressively to drop feature columns with missing values, in `[0.0, 1.0]`.
* `odds_type`: the provider used for `O_train`.
* `target_event_status` and `target_event_time`: the target moment.
* `input_event_status` and `input_event_time`: the input horizon, the latest snapshot kept as a feature.

For the features alone, use [exploration data](#exploration-data).

The examples below use this dataloader.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats
dataloader = DataLoader(
    param_grid={'league': ['England']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds()
)
```

### A target that does not exist

A feed loses a result now and then. scikit-learn needs a value in `y`, so the dataloader drops a match whose outcome the
feed never recorded. It drops the match's `X`, `Y` and `O` rows together, so the three stay aligned. This concerns the
targets. `drop_na_thres` handles missing features.

### The `drop_na_thres` parameter

`drop_na_thres` sets how empty a feature column may be before it is dropped from `X_train`, in `[0.0, 1.0]`, where a
larger value drops more. Historical data often has sparse columns that add little to a model.

At `0.0` every column is kept.

```python
X_train, *_ = dataloader.extract_train_data(drop_na_thres=0.0, odds_type='market_average')
assert len(X_train.columns) == 7
```

At `1.0` the dataloader drops any column with a missing value. The two points averages are empty for a team's first
match of the season, so the dataloader drops them, leaving 5.

```python
X_train, *_ = dataloader.extract_train_data(drop_na_thres=1.0, odds_type='market_average')
assert len(X_train.columns) == 5
```

### The `odds_type` parameter

`odds_type` selects the provider used for `O_train`. Get the available odds types from `get_odds_types`.

```python
assert dataloader.get_odds_types() == ['market_average', 'market_maximum']
```

Its default is `None`, which gives `O_train` no columns.

```python
*_, O_train = dataloader.extract_train_data(drop_na_thres=0.0)
assert O_train.columns.tolist() == []
```

A named odds type gives the matching per provider odds columns.

```python
X_train, _, O_train = dataloader.extract_train_data(drop_na_thres=0.0, odds_type='market_average')
assert all(col.startswith('market_average__') for col in O_train.columns)
assert 'market_average__home_win__preplay__0min' in O_train.columns.tolist()
```

### The target moment

By default `extract_train_data` predicts the final `postplay` outcome, so every earlier snapshot becomes a feature.

```python
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')
assert Y_train.columns.tolist() == [
    'home_win__postplay__0min',
    'draw__postplay__0min',
    'away_win__postplay__0min',
    'over_2.5__postplay__0min',
    'under_2.5__postplay__0min'
]
```

For an in play target, set `target_event_status='inplay'` and `target_event_time` to a [pandas Timedelta]. The features
are the snapshots before that moment, so the target stays out of `X_train`. In play targets need time stamped odds such
as [`OddsApi`][sportsbet.sources.OddsApi].

```python
X_inplay, Y_inplay, O_inplay = dataloader.extract_train_data(
    odds_type='pinnacle',
    target_event_status='inplay',
    target_event_time=pd.Timedelta('60min'),
)
```

Here `Y_inplay` holds the outcome at 60 minutes, `home_win__inplay__60min` and the rest. `X_inplay` holds the snapshots
before it.

### The input horizon

Every snapshot before the target becomes a feature by default. The input horizon caps them at a chosen moment, keeping
the snapshots up to and including it. To train a pre match model, set the horizon to `preplay`.

```python
import pandas as pd
X_pre, Y_pre, O_pre = dataloader.extract_train_data(
    odds_type='market_average',
    input_event_status='preplay',
    input_event_time=pd.Timedelta('0min'),
)
assert not [col for col in X_pre.columns if '__inplay__' in col]
```

Set it to `inplay` at 45 minutes to use information up to half time. An in play horizon needs odds that reach that
moment, so the horizon and the odds agree on when the bet is placed.

The same horizon applies to the fixtures, so training and prediction share the feature set. See
[The moment you bet](../../practice/find_betting_moment.md).

## Exploration data

`extract_exploration_data` returns the features alone, as a single frame `X`, with no targets and no odds. Use it to
look at a sport before choosing a `param_grid` or a model, or when the source carries no odds. It takes the same target
moment and input horizon parameters as `extract_train_data`.

```python
X = dataloader.extract_exploration_data()
assert 'home_points_avg' in X.columns
```

It keeps every match. With no odds to cap the horizon, it carries every snapshot as a feature, including the in-play
ones.

## Fixtures data

A fixture is a match still to be played. After extracting the training data, which fixes the columns, extract the
fixtures with `extract_fixtures_data`.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import FootballDataOdds, FootballDataStats
dataloader = DataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2022, 2023]},
    stats=FootballDataStats(),
    odds=FootballDataOdds(),
)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum')
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

### The fixtures share the training columns, not the training data

`param_grid` chose the seasons to train on, England 2022 and 2023, and those are all played. A fixture is still to be
played, so the two never overlap. `extract_fixtures_data` downloads the current season of each selected league and
returns whatever is still to be played.

```text
X       760 matches   England, 2022-2023   <- the seasons you selected, to train on
X_fix     3 matches   England, upcoming    <- the current season, still to be played
```

The two frames share their columns. That is the contract: the model trained on the history bets on the fixtures.

```python
assert X_train.columns.tolist() == X_fix.columns.tolist()
assert O_train.columns.tolist() == O_fix.columns.tolist()
```

The fixtures follow the leagues you selected, so to bet on Italy, select Italy. A fixture is described by its two teams'
current form, so `extract_fixtures_data` downloads the season they are currently playing.

A finished season yields no fixtures, so the frozen [`SampleSoccerStats`][sportsbet.sources.SampleSoccerStats] has none.
A past match still awaiting a result is a gap in the feed, an abandoned or unrecorded game, and the dataloader leaves it
out.

A fixture has no target matrix.

```python
assert Y_fix is None
```

## Other sports and paid odds

The dataloader is the same whatever the sport. What changes is the sources you give it. These examples use their own
feeds, and `OddsApi` needs a key, so they run against the network.

### Basketball

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import EuroLeagueStats, OddsApi

dataloader = DataLoader(
    param_grid={'league': ['Euroleague'], 'division': [1], 'year': [2025]},
    stats=EuroLeagueStats(),                       # free, no key
    odds=OddsApi(key_env='ODDS_API_KEY', markets=['h2h']),      # yours
)
X, Y, O = dataloader.extract_train_data(odds_type='pinnacle')
```

[`EuroLeagueStats`][sportsbet.sources.EuroLeagueStats] reads the competition's public API, free and no key, and returns
a whole season per request.

Three things differ from soccer, each read from the data.

* The outcome is two way, `home_win` and `away_win`, since a tie goes to overtime. The bettor derives the mutually
  exclusive markets from the columns, so soccer keeps three outcomes and basketball two.
* Totals move from game to game, since the bookmaker sets a line each night, so basketball offers the two way market.
* Basketball odds are yours to buy, so `odds` carries a value here. Soccer is the exception, with football-data giving
  both statistics and odds free.

The dataloader offers the seasons both sources publish.

### The NBA

A league is a source. The NBA is the EuroLeague's sport, so it is the same dataloader with a different statistics
source.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import NBAStats, OddsApi

dataloader = DataLoader(
    param_grid={'league': ['NBA'], 'year': [2026]},
    stats=NBAStats(),                              # free, no key
    odds=OddsApi(key_env='ODDS_API_KEY', markets=['h2h']),      # yours
)
```

[`NBAStats`][sportsbet.sources.NBAStats] is free and needs no key. A season is named by the year it ends, so `2026` is
2025 to 2026, covering the regular season, the play in and the play offs.

Its scores are live: this week's games carry this week's scores, which makes the current season bettable. A full season
is a lot of time stamped odds, so extract once and [keep the result](#keeping-the-data).

## Data of your own

The extraction, the grammar and the moment-aware model apply to any data in the long format, not only the shipped feeds.
The dataloader [derives the layout from the data](#everything-is-derived-from-the-data). So your columns follow the long
format, and the dataloader works out the providers, markets, features and their roles for you.

Bring data in by writing a source: four small methods, covered in
[the sources guide](sources.md#writing-your-own-source). Then give it to `DataLoader` beside any odds source.

```python
from sportsbet.dataloaders import DataLoader

dataloader = DataLoader(stats=MyStats(), odds=MyOdds())
X, Y, O = dataloader.extract_train_data(odds_type='acme')
```

A source whose data is already on disk is still a source. [`SampleSoccerStats`][sportsbet.sources.SampleSoccerStats] is
one. Its items are files, and it reads them straight off the disk.

For a table already in memory, implement [`BaseDataLoader`][sportsbet.dataloaders.BaseDataLoader] directly. It has one
abstract method, the seam every dataloader sits on.

```python
import pandas as pd
from sportsbet.dataloaders import BaseDataLoader


class MyDataLoader(BaseDataLoader):
    """A dataloader of snapshots I already hold."""

    def _load_snapshots(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return stats, odds


X, Y, O = MyDataLoader().extract_train_data(odds_type='acme')
```

## Description of data

The extracted data is a tuple, `(X_train, Y_train, O_train)` for training and `(X_fix, None, O_fix)` for fixtures. The
examples below use this data.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats
dataloader = DataLoader(
    param_grid={'league': ['England']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds()
)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')
X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
```

### X_train

`X_train` is a [pandas DataFrame] of what is known before the target moment: the match identity and the feature
snapshots that precede the target. For the sample's pre match odds and default target, that is the identity columns and
the two points averages.

```python
assert X_train.columns.tolist() == [
    'league',
    'division',
    'year',
    'home_team',
    'away_team',
    'away_points_avg',
    'home_points_avg'
]
```

Its index is a [pandas DateTimeIndex] named `date`, sorted ascending.

```python
import pandas as pd
assert isinstance(X_train.index, pd.DatetimeIndex)
assert X_train.index.name == 'date'
assert X_train.index.is_monotonic_increasing
```

### Y_train

`Y_train` holds the target outcomes as booleans. Column names follow
`f'{market}__{target_event_status}__{target_event_time}'`.

```python
assert Y_train.columns.tolist() == [
    'home_win__postplay__0min',
    'draw__postplay__0min',
    'away_win__postplay__0min',
    'over_2.5__postplay__0min',
    'under_2.5__postplay__0min'
]
```

* `market`: a betting market such as `home_win`, `over_2.5` or `draw`.
* `target_event_status`: `'postplay'` or `'inplay'`.
* `target_event_time`: whole minutes, `0min` at `postplay`.

`X_train`, `Y_train` and `O_train` share the `date` index and the same rows.

### O_train

`O_train` holds the odds. Column names follow `f'{provider}__{market}__{event_status}__{event_time}'`.

```python
assert O_train.columns.tolist() == [
    'market_average__away_win__preplay__0min',
    'market_average__draw__preplay__0min',
    'market_average__home_win__preplay__0min',
    'market_average__over_2.5__preplay__0min',
    'market_average__under_2.5__preplay__0min'
]
```

* `provider`: the odds type you chose through `odds_type`.
* `market`: a betting market.
* `event_status` and `event_time`: the snapshot the odds refer to.

Odds may contain missing values. The bettors take, per market, the odds of the latest available snapshot, so `Y_train`
and `O_train` stay aligned with `X_train`.

### X_fix

`X_fix` holds the fixtures, matches whose target outcome is still open. Its features match `X_train`.

```python
assert X_train.columns.tolist() == X_fix.columns.tolist()
```

### Y_fix

A fixture has no known outcome, so `Y_fix` is `None`.

```python
assert Y_fix is None
```

### O_fix

`O_fix` holds the fixtures' odds, with the columns of `O_train`.

```python
assert O_train.columns.tolist() == O_fix.columns.tolist()
```

## Saving and loading

Save a dataloader with `save` and reload it with `load_dataloader`. It comes back with the snapshots it downloaded. So
`extract_train_data` gives you the training data again without touching the network. The dataloader downloads the data
once and keeps it. It does not download the data again.

```python
import tempfile
from pathlib import Path
from sportsbet.dataloaders import DataLoader, load_dataloader
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats
dataloader = DataLoader(
    param_grid={'league': ['England']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds()
)
X, Y, O = dataloader.extract_train_data(odds_type='market_average')
path = str(Path(tempfile.mkdtemp()) / 'dataloader.pkl')
dataloader.save(path)
reloaded = load_dataloader(path)
X_again, Y_again, O_again = reloaded.extract_train_data()
assert X_again.equals(X)
```
