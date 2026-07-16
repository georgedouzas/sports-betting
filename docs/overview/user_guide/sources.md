[pandera]: <https://pandera.readthedocs.io>

# Data sources

A dataloader gets its data from a source. That is why adding a sport is adding a source, why odds and statistics are
bought separately, and why you can plug in a feed of your own without changing a line of the library.

The sources that ship with the library:

| Source | Sport | Cost | Carries |
| --- | --- | --- | --- |
| [`FootballDataStats`][sportsbet.sources.FootballDataStats] | soccer | free | statistics, 1993 onward |
| [`FootballDataOdds`][sportsbet.sources.FootballDataOdds] | soccer | free | pre match closing odds |
| [`EuroLeagueStats`][sportsbet.sources.EuroLeagueStats] | basketball | free | statistics, from the competition's own API |
| [`NBAStats`][sportsbet.sources.NBAStats] | basketball | free | statistics, the NBA, live through a season |
| [`OddsApi`][sportsbet.sources.OddsApi] | any | your key | time stamped odds, live and historical |
| [`SampleSoccerStats`][sportsbet.sources.SampleSoccerStats] | soccer | free | one frozen season, shipped for offline examples |
| [`SampleSoccerOdds`][sportsbet.sources.SampleSoccerOdds] | soccer | free | the odds for that frozen season |

Mix them however you like. Free statistics with paid odds is the realistic setup, and the only way to backtest an in
play bet.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import FootballDataStats, OddsApi

dataloader = DataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2025]},
    stats=FootballDataStats(),                                   # free
    odds=OddsApi(key='...', markets=['h2h', 'totals']),          # yours
)
```

## The contract

A source answers four questions by declaring what to read. The dataloader reads the items it declares into memory and
hands the payloads back, so a source stays a plain description of a feed, easy to write and easy to test.

```python
class BaseSource:

    name: ClassVar[str]              # who declared an item
    kind: ClassVar[str]              # 'stats' or 'odds'
    sport: ClassVar[str | None]      # what the data is about, or None if the source serves any sport

    def index_items(self, selection=None) -> list[RawItem]:
        """What do I need to read to know what I publish?"""

    def catalogue(self, payloads) -> list[dict]:
        """Given those, which league/division/year combinations do I publish?"""

    def required_items(self, params, schedule=None) -> list[RawItem]:
        """Given a selection, what do I need read?"""

    def fixtures_items(self, params, schedule=None) -> list[RawItem]:
        """And what do I need read for the upcoming matches? (defaults to required_items)"""

    def to_snapshots(self, payloads) -> pd.DataFrame:
        """Given those, what are the long snapshots?"""
```

The `sport` lives on the source, so the dataloader takes its sport from the statistics you pass: soccer from
`FootballDataStats`. A source that serves any sport, as `OddsApi` does, leaves it `None` and takes the sport of the
statistics it is paired with.

Two optional hooks:

* `request_url(item)` adds a credential at the moment of the request, so it never reaches a `RawItem` and is never part
  of the data you save.
* `needs_schedule()` returns `True` if you address your data by instant rather than by season. `OddsApi` does. "The
  price at minute 45" is a timestamp, and it can only build it once it knows the kick off.

### `RawItem` and `RawPayload`

A [`RawItem`][sportsbet.sources.RawItem] is one thing to read, a URL, or a `file://` path for a feed that ships with the
library.

```python
RawItem(
    source='my_stats',                    # who declared it
    key='England_1_2025',                 # its identity within the source
    url='https://example.com/2025.csv',
)
```

An item carries no price. A vendor sets its own, changes them, and prices its endpoints differently, so a library that
quoted you a cost would be quoting a number it had made up. What the library reports is the number of requests, which is
a fact. What they are worth is between you and whoever you buy them from.

Two sources declaring the same `source` and `key` declare the same item, so it is fetched once. That is how
`FootballDataStats` and `FootballDataOdds`, which read the same upstream CSV, avoid downloading it twice.

A [`RawPayload`][sportsbet.sources.RawPayload] is what came back, kept verbatim. It is what your `catalogue` and
`to_snapshots` are handed.

```python
from sportsbet.sources import RawPayload

payload = RawPayload(item=item, content=b'date,home_team,away_team\n...')
payload.item.key      # 'England_1_2025'
payload.content       # exactly what the feed returned, unparsed
```

## Writing your own source

Two feeds, a statistics one and an odds one, for a league the library has never heard of.

```python
import io
import json

import pandas as pd
from sportsbet.sources import BaseOddsSource, BaseStatsSource, RawItem, market_outcomes

MARKETS = ['home_win', 'draw', 'away_win']
IDENTITY = ['date', 'league', 'division', 'year', 'home_team', 'away_team']


class MyStats(BaseStatsSource):
    """Statistics from a feed of your own."""

    name = 'my_stats'

    def index_items(self):
        return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json')]

    def catalogue(self, payloads):
        seasons = json.loads(payloads[0].content)
        return [{'league': 'Ruritania', 'division': 1, 'year': year} for year in seasons]

    def required_items(self, params, schedule=None):
        return [
            RawItem(
                source=self.name,
                key=f'Ruritania_1_{param["year"]}',
                url=f'https://example.com/{param["year"]}.csv',
            )
            for param in params
        ]

    def to_snapshots(self, payloads):
        frames = []
        for payload in payloads:
            games = pd.read_csv(io.BytesIO(payload.content))
            games['date'] = pd.to_datetime(games['date'], utc=True)     # the kick-off, in UTC

            preplay = games[IDENTITY].assign(
                event_status='preplay', event_time=0,
                home_form=games['home_form'], away_form=games['away_form'],
            )
            played = games['home_goals'].ge(0)
            postplay = games.loc[played, IDENTITY].assign(event_status='postplay', event_time=0)
            outcomes = market_outcomes(games.loc[played, 'home_goals'], games.loc[played, 'away_goals'], MARKETS)
            postplay = pd.concat([postplay, outcomes], axis=1)

            frames.append(pd.concat([preplay, postplay], ignore_index=True))
        return pd.concat(frames, ignore_index=True)


class MyOdds(BaseOddsSource):
    """Odds from a feed of your own."""

    name = 'my_odds'

    def index_items(self):
        return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json')]

    def catalogue(self, payloads):
        return [{'league': 'Ruritania', 'division': 1, 'year': y} for y in json.loads(payloads[0].content)]

    def required_items(self, params, schedule=None):
        return [
            RawItem(source=self.name, key=f'odds_{param["year"]}', url=f'https://example.com/odds/{param["year"]}.csv')
            for param in params
        ]

    def to_snapshots(self, payloads):
        odds = pd.concat([pd.read_csv(io.BytesIO(payload.content)) for payload in payloads], ignore_index=True)
        odds['date'] = pd.to_datetime(odds['date'], utc=True)
        return odds.assign(event_status='preplay', event_time=0)
```

Then hand them to any dataloader.

```python
from sportsbet.dataloaders import DataLoader

dataloader = DataLoader(stats=MyStats(), odds=MyOdds())
X, Y, O = dataloader.extract_train_data(odds_type='acme')
```

```text
X   ['league', 'division', 'year', 'home_team', 'away_team', 'home_form', 'away_form']
Y   ['home_win__postplay__0min', 'draw__postplay__0min', 'away_win__postplay__0min']
O   ['acme__home_win__preplay__0min', 'acme__draw__preplay__0min', 'acme__away_win__preplay__0min']
```

Nothing was configured. The markets came from the odds columns, the providers from the odds `provider` column, the
features from the statistics columns, and the moments from `event_status` and `event_time`. Drop `draw` from `MARKETS`
and you have a sport that cannot be drawn, and the bettor works out the two way market on its own.

### Four rules to follow

1. Keep the four methods pure. `index_items`, `catalogue`, `required_items` and `to_snapshots` declare and transform,
   and the dataloader does the reading, so a source stays testable offline.
2. `date` is the kick off instant, in UTC. Resolve your feed's time zone at your boundary, so `date + event_time` is the
   wall clock instant of a snapshot, the address an odds vendor is asked for. Both feeds the library ships hide this:
   football-data publishes every league in UK time, and the EuroLeague every game in Central European time. Assume
   nothing.
3. The upcoming matches come from `fixtures_items`. The default reads the same items as training, which suits a feed
   whose season file already lists the matches still to be played. Override it when they live elsewhere.
4. Credentials go in `request_url`. The `RawItem` is what the transform sees and what you might save, so a key stays out
   of it.

## Keeping the data

The dataloader is the store. Extracting downloads the data into it, and `save` writes the object out.

```python
dataloader.save('italy.pkl')

from sportsbet.dataloaders import load_dataloader
dataloader = load_dataloader('italy.pkl')      # the data comes back with it
```

Extract once, save, and load to reuse the data, and for a paid odds feed to reuse what you paid for. You own the file,
so where it lives and how long it lasts is your call.

## When two sources disagree about a name

Mixing sources means one calls a club `Man United` and the other calls it `Manchester United`. If a name fails to match,
that game has no odds, and a missing odd does not look like an error. It looks like a slightly smaller dataset, and a
backtest that is clean, plausible and wrong.

So they are reconciled, and the result is a hard gate.

```python
X, Y, O = dataloader.extract_train_data(odds_type='pinnacle')
dataloader.reconciliation_        # a ReconciliationReport
```

A [`ReconciliationReport`][sportsbet.sources.ReconciliationReport] carries `matched`, `unmatched_rate`,
`unmatched_stats`, `unmatched_odds` and `suggestions`. Cross `max_unmatched_rate`, which defaults to zero, and you get
[`UnmatchedError`][sportsbet.sources.UnmatchedError] rather than a holed dataset.

```python
from sportsbet.sources import UnmatchedError

try:
    dataloader.extract_train_data(odds_type='pinnacle')
except UnmatchedError as error:
    print(error.report.aliases())
```

```text
{
    'Olimpia Milano': 'EA7 Emporio Armani Milan',
}
```

Check it, because a suggestion is a resemblance and not a fact, then pass it back as `aliases={...}`. See
[the dataloader guide](dataloader.md#when-two-sources-name-a-club-differently) for why the library never applies one on
its own.

You can reconcile two tables yourself with [`resolve`][sportsbet.sources.resolve].

```python
from sportsbet.sources import resolve

odds, report = resolve(stats, odds, aliases={'Olimpia Milano': 'EA7 Emporio Armani Milan'})
```

## Describing your own columns

Snapshots are validated against [pandera] schemas that the library builds from the data. You rarely need to write one.
When you do, to require a column or to say at which moments it carries values, subclass
[`BaseStatsSchema`][sportsbet.sources.BaseStatsSchema] or [`BaseOddsSchema`][sportsbet.sources.BaseOddsSchema] and
describe the columns with [`required_col`][sportsbet.sources.required_col] and
[`optional_col`][sportsbet.sources.optional_col].

```python
from typing import Annotated
import pandas as pd
from sportsbet.sources import BaseStatsSchema, optional_col, required_col


class MySchema(BaseStatsSchema):
    """The statistics of my feed."""

    date: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = required_col()
    league: str = required_col()
    home_team: str = required_col()
    away_team: str = required_col()
    home_goals: int = optional_col(['inplay', 'postplay'], fixed=False)   # varies by moment
    home_form: float = optional_col(['preplay'], fixed=True)              # one value per match
```

An odds schema is the same, with a `provider`.

```python
from sportsbet.sources import BaseOddsSchema


class MyOddsSchema(BaseOddsSchema):
    """The odds of my feed."""

    date: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = required_col()
    league: str = required_col()
    home_team: str = required_col()
    away_team: str = required_col()
    provider: str = optional_col(['preplay'], fixed=True)
    home_win: float = optional_col(['preplay', 'inplay'], fixed=False)
```

`fixed` is the difference between a column that keeps a bare name, like `home_form`, and one that is expanded per
moment, like `home_goals__inplay__45min`.

## The abstract classes

[`BaseDataLoader`][sportsbet.dataloaders.BaseDataLoader] is the extraction engine. Its one abstract method is
`_snapshots()`, which returns the long `stats` and `odds` tables. Everything else, the column grammar, the input
horizon, the moment aware pivot, is done for you.

```python
from sportsbet.dataloaders import BaseDataLoader


class MyDataLoader(BaseDataLoader):
    """A dataloader whose data comes from wherever you like."""

    def _snapshots(self):
        return my_stats_table, my_odds_table
```

That is the seam. There is one `DataLoader` behind it, whatever the sport, because the sport belongs to the source. So
adding a sport, a league or a feed of your own is adding a source.

[`BaseBettor`][sportsbet.evaluation.BaseBettor] is the betting strategy. Implement `_fit` and `_predict_proba` and you
get value bets, backtesting and hyperparameter search. See [the bettor guide](bettor.md#implementation).
