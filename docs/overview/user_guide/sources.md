[pandera]: <https://pandera.readthedocs.io>

# Data sources

A dataloader gets its data from a source. This has three consequences. You add a sport by adding a source. You buy odds
and statistics separately, from separate sources. You can plug in a feed of your own without changing the library.

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

Mix them however you like. Free statistics with paid odds is the realistic setup. It is the only way to backtest an
in-play bet.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import FootballDataStats, OddsApi

dataloader = DataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2025]},
    stats=FootballDataStats(),                                   # free
    odds=OddsApi(key_env='ODDS_API_KEY', markets=['h2h', 'totals']),          # yours
)
```

## The contract

A source declares what to read. It answers four questions. The dataloader reads the items the source declares into
memory and hands the payloads back. So a source stays a plain description of a feed. It is easy to write and easy to
test.

```python
class BaseSource:

    name: ClassVar[str]              # who declared an item
    kind: ClassVar[str]              # 'stats' or 'odds'
    sport: ClassVar[str | None]      # what the data is about, or None if the source serves any sport

    def list_index_items(self, selection=None) -> list[RawItem]:
        """What do I need to read to know what I publish?"""

    def read_catalogue(self, payloads) -> list[dict]:
        """Given those, which league/division/year combinations do I publish?"""

    def list_required_items(self, params, schedule=None) -> list[RawItem]:
        """Given a selection, what do I need read?"""

    def list_fixtures_items(self, params, schedule=None) -> list[RawItem]:
        """And what do I need read for the upcoming matches? (defaults to required_items)"""

    def to_snapshots(self, payloads) -> pd.DataFrame:
        """Given those, what are the long snapshots?"""
```

The `sport` lives on the source. The dataloader takes its sport from the statistics source you pass, for example soccer
from `FootballDataStats`. A source that serves any sport, as `OddsApi` does, leaves `sport` as `None`. It takes the
sport of the statistics source it is paired with.

Two optional hooks:

* `request_url(item)` adds a credential at the moment of the request. The credential never reaches a `RawItem` and never
  enters the data you save.
* `needs_schedule()` returns `True` when you address your data by instant rather than by season. `OddsApi` does. "The
  price at minute 45" is a timestamp. The source can build it only once it knows the kick off.

### `RawItem` and `RawPayload`

A [`RawItem`][sportsbet.sources.RawItem] is one thing to read. It is a URL, or a `file://` path for a feed that ships
with the library.

```python
RawItem(
    source='my_stats',                    # who declared it
    key='England_1_2025',                 # its identity within the source
    url='https://example.com/2025.csv',
)
```

An item carries no price. Each vendor sets its own prices, changes them, and prices its endpoints differently. The
library reports the number of requests, which is a fact. What they cost is between you and the vendor you buy them from.

Two sources that declare the same `source` and `key` declare the same item, so the dataloader fetches it once.
`FootballDataStats` and `FootballDataOdds` read the same upstream CSV, so they avoid downloading it twice.

A [`RawPayload`][sportsbet.sources.RawPayload] is what came back, kept verbatim. The dataloader hands it to your
`read_catalogue` and `to_snapshots`.

```python
from sportsbet.sources import RawPayload

payload = RawPayload(item=item, content=b'date,home_team,away_team\n...')
payload.item.key      # 'England_1_2025'
payload.content       # exactly what the feed returned, unparsed
```

## Writing your own source

Here are two feeds for a league the library does not ship, one for statistics and one for odds.

```python
import io
import json

import pandas as pd
from sportsbet.sources import BaseOddsSource, BaseStatsSource, RawItem, derive_market_outcomes

MARKETS = ['home_win', 'draw', 'away_win']
IDENTITY = ['date', 'league', 'division', 'year', 'home_team', 'away_team']


class MyStats(BaseStatsSource):
    """Statistics from a feed of your own."""

    name = 'my_stats'

    def list_index_items(self):
        return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json')]

    def read_catalogue(self, payloads):
        seasons = json.loads(payloads[0].content)
        return [{'league': 'Ruritania', 'division': 1, 'year': year} for year in seasons]

    def list_required_items(self, params, schedule=None):
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
            outcomes = derive_market_outcomes(games.loc[played, 'home_goals'], games.loc[played, 'away_goals'], MARKETS)
            postplay = pd.concat([postplay, outcomes], axis=1)

            frames.append(pd.concat([preplay, postplay], ignore_index=True))
        return pd.concat(frames, ignore_index=True)


class MyOdds(BaseOddsSource):
    """Odds from a feed of your own."""

    name = 'my_odds'

    def list_index_items(self):
        return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json')]

    def read_catalogue(self, payloads):
        return [{'league': 'Ruritania', 'division': 1, 'year': y} for y in json.loads(payloads[0].content)]

    def list_required_items(self, params, schedule=None):
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

You configured nothing. The markets came from the odds columns. The providers came from the odds `provider` column. The
features came from the statistics columns. The moments came from `event_status` and `event_time`. Drop `draw` from
`MARKETS` and you have a sport that cannot draw, and the bettor works out the two-way market on its own.

### Four rules to follow

1. Keep the four methods pure. `list_index_items`, `read_catalogue`, `list_required_items` and `to_snapshots` declare
   and transform. The dataloader does the reading, so a source stays testable offline.
2. `date` is the kick-off instant, in UTC. Resolve your feed's time zone at your boundary, so `date + event_time` is the
   wall-clock instant of a snapshot. That instant is the address an odds vendor is asked for. Both feeds the library
   ships hide their local time. football-data publishes every league in UK time. The EuroLeague publishes every game in
   Central European time. Check your feed's time zone before you trust it.
3. The upcoming matches come from `list_fixtures_items`. The default reads the same items as training. That suits a feed
   whose season file already lists the matches still to be played. Override it when they live elsewhere.
4. Credentials go in `request_url`. The `RawItem` is what the transform sees and what you might save, so a key stays out
   of it.

## Keeping the data

The dataloader is the store. Extracting downloads the data into it. `save` writes the object out.

```python
dataloader.save('italy.pkl')

from sportsbet.dataloaders import load_dataloader
dataloader = load_dataloader('italy.pkl')      # the data comes back with it
```

Extract once, save, and load to reuse the data. For a paid odds feed, this reuses what you paid for. You own the file.
Where it lives and how long it lasts is up to you.

## When two sources name a club differently

This is the most dangerous thing in the library, so read it carefully.

Two sources rarely spell a club the same way. One calls a club `Man United` and the other calls it `Manchester United`.
When a name fails to match, that game has no odds. A missing odd does not look like an error. It looks like a slightly
smaller dataset. The result is a backtest that is clean, plausible and wrong.

So the dataloader reconciles the two sources. It compares the spellings and pairs the two feeds for you. A name it
cannot place is dropped, not guessed at.

Sometimes the pairing leaves a club unmatched. You name that club yourself by passing `aliases`. A resemblance is not a
fact, and attaching one club's odds to another is worse than a missing row.

See [the dataloader guide](dataloader.md#when-two-sources-name-a-club-differently) for how a dataloader takes aliases.

You can reconcile two tables yourself with [`resolve_odds`][sportsbet.sources.resolve_odds]. It returns the odds
carrying the identity of the statistics, so the two tables line up.

```python
from sportsbet.sources import resolve_odds

odds = resolve_odds(stats, odds, aliases={'Olimpia Milano': 'EA7 Emporio Armani Milan'})
```

## Describing your own columns

The library validates snapshots against [pandera] schemas that it builds from the data. You rarely need to write one.
Write one when you want to require a column or to say at which moments it carries values. Subclass
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

`fixed` sets whether a column keeps a bare name or is expanded per moment. A fixed column like `home_form` keeps its
bare name. A column that is not fixed like `home_goals` becomes `home_goals__inplay__45min`.

## The abstract classes

[`BaseDataLoader`][sportsbet.dataloaders.BaseDataLoader] is the extraction engine. It has one abstract method,
`_load_snapshots()`, which returns the long `stats` and `odds` tables. The library does everything else for you: the
column grammar, the input horizon, and the moment-aware pivot.

```python
from sportsbet.dataloaders import BaseDataLoader


class MyDataLoader(BaseDataLoader):
    """A dataloader whose data comes from wherever you like."""

    def _load_snapshots(self):
        return my_stats_table, my_odds_table
```

That method is the seam. There is one `DataLoader` behind it, whatever the sport, because the sport belongs to the
source. So you add a sport, a league or a feed of your own by adding a source.

[`BaseBettor`][sportsbet.evaluation.BaseBettor] is the betting strategy. Implement `_fit` and `_predict_proba` and you
get value bets, backtesting and hyperparameter search. See [the bettor guide](bettor.md#implementation).
