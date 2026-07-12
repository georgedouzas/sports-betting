[ParameterGrid]: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html>
[pandas DataFrame]: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>
[pandas DateTimeIndex]: <https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html>
[pandas Timedelta]: <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>
[pandera]: <https://pandera.readthedocs.io>

# Dataloader

This section presents the dataloader object in details. The available dataloaders are the following:

- [`DummySoccerDataLoader`][sportsbet.datasets.DummySoccerDataLoader]: Soccer data loader with bundled sample data, for offline testing and the examples below.
- [`SoccerDataLoader`][sportsbet.datasets.SoccerDataLoader]: Soccer data loader that downloads historical and fixtures data from the sources you give it.
- [`BasketballDataLoader`][sportsbet.datasets.BasketballDataLoader]: Basketball data loader, covering the EuroLeague.

We aim to include in the future more sports and betting markets, such as the NFL and hockey. **A sport is a data source,
not an engine** — the two dataloaders above differ only in which sources they default to, and nothing in the extraction
knows which sport it is looking at.

## Basketball

```python
from sportsbet.datasets import BasketballDataLoader, EuroLeagueStats, OddsApi

dataloader = BasketballDataLoader(
    param_grid={'league': ['Euroleague'], 'division': [1], 'year': [2025]},
    stats=EuroLeagueStats(),                       # free, no key
    odds=OddsApi(key='...', markets=['h2h']),      # yours
)
dataloader.prepare()
X, Y, O = dataloader.extract_train_data(odds_type='pinnacle')
```

[`EuroLeagueStats`][sportsbet.datasets.EuroLeagueStats] reads the competition's own public API. It is free, needs no key,
and a whole season arrives in a single request.

Three things differ from soccer, and **none of them is configured** — each is read from the data:

- **There is no draw.** A tie goes to overtime, so the outcome is two-way: `home_win` and `away_win`. The bettor derives
  which markets are mutually exclusive from the columns the data carries, so a two-way outcome simply falls out — and
  soccer keeps its three-way one.
- **There is no totals market.** Basketball has no *line*: the total points of a game run from about 125 to 229 and a
  bookmaker sets a different one every night. This library expresses a market as a **column**, and `over_167.5` would be
  true of one game and meaningless for the next. A market whose line moves needs a change to the data model, so it is not
  here yet.
- **You must bring odds.** There is **no free basketball odds source anywhere**, so `odds` has no default. A dataloader
  without one carries no betting markets and therefore has nothing to predict — and it says exactly that rather than
  failing somewhere deeper. Soccer is the exception, not the rule: football-data is the only feed in the library that
  gives statistics *and* odds for nothing.

Only the seasons **both** sources publish are offered, so you cannot select a season your odds vendor cannot price.

## Code, not data

This library ships the code that fetches the data, never the data itself. The odds published by bookmakers belong to whoever
published them, and redistributing them is not ours to do — so `SoccerDataLoader` downloads
[football-data.co.uk](https://www.football-data.co.uk) on **your** machine and runs the transform locally. Nothing is mirrored.

Two consequences follow, and they shape the whole interface:

- Data has to be **downloaded before it can be used**. That happens in an explicit [`prepare`](#preparing-the-data) step, never
  as a side effect of asking for training data.
- Where the data comes from is a **choice you make**, by injecting [sources](#sources). Free statistics and paid odds are
  complementary rather than alternatives, so they are separate parameters.

## The event-snapshot data model

Internally the data are stored in long format as event *snapshots*. Each row is a single match observed at a single moment,
identified by two columns:

- `event_status`: the phase of the match, one of `'preplay'` (before kick-off), `'inplay'` (while the match is running) or
  `'postplay'` (final result).
- `event_time`: a [pandas Timedelta] measuring the elapsed time from kick-off, e.g. `pd.Timedelta('30min')`. It is `0min` for
  `preplay` and `postplay` snapshots.

### Time is always UTC, and `date` is the kick-off

`date` is the **kick-off instant** of the match, in UTC — not the calendar day. Every source converts its own local
representation into UTC at its own boundary, so no source ever hands you a naive or a local time. The football-data.co.uk feed,
for instance, publishes *every* league's kick-off in **UK time** regardless of the country the match is played in, and that is
resolved before you ever see it.

This gives the model its central invariant:

```text
date + event_time  =  the wall-clock instant of the snapshot
```

which is what makes a moment of a match *addressable*. "The odds at minute 45 of Arsenal–Chelsea" is a real timestamp you can ask
an odds provider for. Without a kick-off time it would not be.

Some older seasons of a feed carry no kick-off time. Their matches fall back to midnight UTC, and because they predate every
time-stamped odds source, they can never be paired with one anyway.

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

Except where a real source is being shown, the examples in this section use the offline
[`DummySoccerDataLoader`][sportsbet.datasets.DummySoccerDataLoader], so they run without any network access and without a
preparation step.

## Initialization

A dataloader is initialized with the parameter `param_grid` that selects the training data to extract. Indirectly this parameter
also affects the extracted fixtures data since dataloaders ensure that these two are in correspondence i.e. input and odds
matrices of training and fixtures data have the same columns.

### Available parameters

**Ask the source.** You cannot write a `param_grid` before you know what exists, so discovery does not live on the
dataloader — it lives on the data source, which is what actually determines availability:

```python
from sportsbet.datasets import FootballDataStats

params = FootballDataStats().available_params()
# Only the league/division/year combinations the feed actually publishes are
# ever offered, so an invalid one can never be requested.
assert {'division': 1, 'league': 'England', 'year': 2024} in params
assert all({'league', 'division', 'year'} == set(combination) for combination in params)
```

`available_params` is an instance method rather than a class method, because what a source publishes depends on how it is
configured — a credential may only cover part of what the source offers.

The catalogue is re-read on every call rather than cached, so a new season shows up as soon as the feed publishes it.
It is small, and reading it takes a couple of seconds.

Only the parameters that **both** your statistics source and your odds source publish can be modelled, so the dataloader
selects their intersection. A season whose statistics exist but whose odds do not is never chosen — otherwise the missing
odds would show up as a quietly smaller dataset and a confidently wrong backtest.

### Selection of parameters

The parameter `param_grid` has the same usage as the initialization parameter of scikit-learn's [ParameterGrid]. It accepts:

- `None` (the default) — selects **all** available data, i.e. every combination both sources publish.
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

Once the dataloader is initialized, the data has to be prepared before it can be extracted.

## Sources

A source is where the data comes from. `SoccerDataLoader` takes two of them, and each carries its own settings — so adding a
source never widens the dataloader's signature:

```python
from sportsbet.datasets import SoccerDataLoader, FootballDataStats, FootballDataOdds

dataloader = SoccerDataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2025]},
    stats=FootballDataStats(),
    odds=FootballDataOdds(),
)
```

Both default to the free [football-data.co.uk](https://www.football-data.co.uk) feed, so the above is the same as
`SoccerDataLoader(param_grid=...)`.

The statistics and the odds are **separate parameters on purpose**. They are not two ways of getting the same thing: the free
feed carries pre-match closing odds, which is enough to backtest a pre-match bet but not an in-play one, since it never tells you
the price that was available at minute 45. A source with time-stamped prices does. Combining free statistics with your own paid
odds is therefore the realistic configuration, and a single "free or paid" switch could not express it.

### Bringing your own odds

[`OddsApi`][sportsbet.datasets.OddsApi] buys time-stamped prices from [The Odds API](https://the-odds-api.com) with **your** key.
That is what makes an in-play bet backtestable: the odds it returns are the ones that were actually on offer at the minute the bet
would have been placed.

```python
from sportsbet.datasets import SoccerDataLoader, FootballDataStats, OddsApi

dataloader = SoccerDataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2025]},
    stats=FootballDataStats(),                                    # free
    odds=OddsApi(key='...', markets=['h2h', 'totals'], regions=['eu']),   # yours
)
```

**It costs money, so it is priced before it is spent.** A historical snapshot costs ten times a live one, multiplied by every
market and every region you ask for. Ask what it would cost first:

```python
report = dataloader.prepare(dry_run=True)
report.estimated_cost     # {'odds_api': 8642}
```

That number is *exact*, and it costs *nothing* to obtain. The statistics are free and they say when every match kicks off, so the
snapshots to buy — one per kick-off, per moment — can be counted without asking the vendor for a single one of them. Matches that
kick off together share a snapshot and are paid for once.

Your key is added to a request at the moment it is made. It is never part of a stored item and never written to the store.

Two limits are worth knowing. The vendor's history begins in **June 2020**, and historical prices are a **paid tier**. Since only
the seasons both sources publish can be modelled, older seasons simply are not offered when `OddsApi` is the odds source — you
cannot accidentally select a season it cannot price.

### When two sources name a club differently

This is the most dangerous thing in the library, so it is worth being blunt about.

Two sources do not name the same club the same way. The free feed says `Man United`, `Nott'm Forest`, `Wolves`; the odds vendor
says `Manchester United`, `Nottingham Forest`, `Wolverhampton Wanderers`. If a name fails to match, that match simply has no odds
— and **a missing odd does not look like an error**. It looks like a slightly smaller dataset, which produces a backtest that is
clean, plausible, and wrong.

So when your statistics and your odds come from different sources, they are reconciled, and the result is a hard gate:

```python
dataloader.prepare()
X, Y, O = dataloader.extract_train_data(odds_type='pinnacle')
dataloader.reconciliation_          # matched, unmatched_rate, unmatched_stats, unmatched_odds
```

By default **not one match may go without odds** (`max_unmatched_rate=0.0`). Cross that and it raises
[`UnmatchedError`][sportsbet.datasets.UnmatchedError] rather than handing you a holed dataset.

**Most of the time you will not have to do anything.** The names are paired **within a league and a season**, where the two
sources hold the same twenty clubs. That is what makes it safe rather than a guess: every name that could be confused with another
is *present on both sides*, so it matches itself before anything is inferred. `Manchester City` pairs with `Man City` because
`Manchester United` and `Man United` are both there too, claiming each other.

Clubs are abbreviated by **shortening their words**, not by changing their letters — `Man United`, `Wolves`, `Nott'm Forest` — so
names are compared by the prefixes their words share. Comparing them as strings would be actively dangerous: it rates `Everton`
against `Liverpool` as *more* alike than `Wolves` against `Wolverhampton Wanderers`, which is exactly the mistake that attaches one
club's odds to another match.

A name is paired only when it is clearly the best of the roster *and* clearly better than the next best. Anything ambiguous is left
alone. The last name standing is never paired just for being last — a vendor carrying a club your statistics do not have would
otherwise have its odds hung on the wrong match.

When the pairing cannot place a name, it says so and tells you exactly what to do:

```text
Matched 2 of 3 matches (33.3% unmatched). These team names were not found: ['Athletic Bilbao'].
Check them and pass them as `aliases`:
aliases={
    'Athletic Bilbao': 'Ath Bilbao',
}
```

**Check it** — a suggestion is a resemblance, not a fact, and the library will never apply one on its own. Then pass it back:

```python
SoccerDataLoader(..., aliases={'Athletic Bilbao': 'Ath Bilbao'})
```

The reconciled odds then carry the **statistics'** identity: their kick-off and their spelling, so the two tables line up exactly
rather than nearly.

None of this runs on the free path, where the statistics and the odds come from the same upstream row and their identities are
equal by construction.

A source declares *what* it needs and *how* to transform it. It never fetches — that is the store's job. This is what makes the
guarantees below possible rather than merely intended.

## Preparing the data

Downloading happens in `prepare`, and nowhere else:

```python
dataloader.prepare()
X_train, Y_train, O_train = dataloader.extract_train_data()
```

Extracting data from a dataloader that was not prepared raises
[`NotPreparedError`][sportsbet.datasets.NotPreparedError] rather than quietly downloading:

```python
from sportsbet.datasets import NotPreparedError

try:
    SoccerDataLoader(param_grid={'league': ['England']}).extract_train_data()
except NotPreparedError as error:
    print(error)  # says what is missing and what obtaining it would cost
```

That is deliberate. A metered odds source turns an accidental call into money, and a large `param_grid` turns it into a long
wait. Neither should be able to happen because you asked for a dataframe.

`prepare` is **incremental**: it fetches only what the store does not already hold, so re-running it on a complete store fetches
nothing. It is **resumable**: an interrupted run continues rather than starting over. And it can tell you what it *would* do,
before it does anything:

```python
report = dataloader.prepare(dry_run=True)
report.to_fetch        # the items that would be downloaded
report.held            # the items already in the store
report.estimated_cost  # what a metered source would charge, e.g. {'odds_api': 1240}
report.unavailable     # requested parameters the source does not publish
```

A dry run downloads no data and spends nothing.

### Where the data is kept

The downloaded data lives in a [`LocalStore`][sportsbet.datasets.LocalStore], by default under `~/.sportsbet`:

```python
from sportsbet.datasets import SoccerDataLoader, LocalStore

dataloader = SoccerDataLoader(param_grid={'league': ['England']}, store=LocalStore('/data/sportsbet'))
```

The store keeps the raw downloads **forever**, and everything derived from them is rebuilt from those raw payloads at no cost.
This matters if you ever pay for odds: changing the feature engineering, or upgrading the library, never re-downloads and never
re-charges you. The derived data is keyed by the library version as well as by the raw content, so an upgrade that changes the
transform rebuilds rather than serving you what the old one produced.

### What happens when the upstream data changes

The store treats two kinds of data differently:

- **Data that is expected to change** — the fixtures, the season in progress, and the catalogue of what is available — is
  re-read on every `prepare()`. You never have to invalidate anything by hand, and there is no expiry to tune.
- **Data that is finished** — a completed season — is kept and not downloaded again. This is what makes `prepare()`
  incremental, and what stops a metered source from re-charging you for history you already bought.

The second rule has a limit worth knowing: **nothing upstream is truly immutable.** football-data.co.uk does occasionally
correct a finished season, and a correction like that will not be picked up on its own, because the store has no reason to
look. When you want it to look, ask:

```python
dataloader.prepare(refresh=True)   # re-read everything, including what is already held
```

That is a deliberate choice rather than a schedule, because on a metered source a refresh is charged for again. If the
re-fetched content turns out to be identical, nothing downstream is rebuilt — the derived data is keyed by content, so an
unchanged file costs nothing beyond the download.

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

The extraction, grammar and moment-aware model described above are not tied to the bundled feed: two factory functions build a
dataloader straight from data you provide — [`from_snapshots`][sportsbet.datasets.from_snapshots] and
[`from_dataframe`][sportsbet.datasets.from_dataframe]. Because the layout is
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

If instead you have one wide row per match, all observed at the *same* moment, use `from_dataframe` and declare exactly what that
moment is with `event_status` and `event_time` — nothing is assumed. Odds columns follow the `{provider}__{market}` naming and are
split out into the `odds` table automatically:

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
assert reloaded.param_grid_ == dataloader.param_grid_
```
