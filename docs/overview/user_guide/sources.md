[pandera]: <https://pandera.readthedocs.io>

# Data sources

Sources define where the data comes from. There are two types, statistics sources and odds sources. A statistics source
is always about one sport. An odds source can cover multiple sports.

Every source knows its `name`, `kind` and `sport` as class attributes. `list_available_params` lists the leagues,
divisions and seasons it publishes. It is a method on the source object, not the class, because a source can need
configuration, such as a key, before it knows what it publishes.

The code blocks below run when the documentation is built, so their output is the real thing.

## Statistics sources

A statistics source provides the match statistics for one sport. Three ship with the library.

### FootballDataStats

Soccer statistics from [football-data.co.uk](https://www.football-data.co.uk), free, with seasons from 1993 onward.

```python exec="1" source="block" result="text"
from sportsbet.sources import FootballDataStats

print((FootballDataStats.name, FootballDataStats.kind, FootballDataStats.sport))
params = FootballDataStats().list_available_params()
print([p for p in params if p['league'] == 'England' and p['division'] == 1 and p['year'] >= 2024])
```

### EuroLeagueStats

Basketball statistics from the EuroLeague's own API, free.

```python exec="1" source="block" result="text"
from sportsbet.sources import EuroLeagueStats

print((EuroLeagueStats.name, EuroLeagueStats.kind, EuroLeagueStats.sport))
print([p for p in EuroLeagueStats().list_available_params() if p['year'] >= 2025])
```

### NBAStats

Basketball statistics for the NBA, free, updated live through a season.

```python exec="1" source="block" result="text"
from sportsbet.sources import NBAStats

print((NBAStats.name, NBAStats.kind, NBAStats.sport))
print([p for p in NBAStats().list_available_params() if p['year'] >= 2025])
```

## Odds sources

An odds source provides the betting odds. Two ship with the library.

### FootballDataOdds

Soccer odds from football-data.co.uk, free, the pre-match closing prices.

```python exec="1" source="block" result="text"
from sportsbet.sources import FootballDataOdds

print((FootballDataOdds.name, FootballDataOdds.kind, FootballDataOdds.sport))
params = FootballDataOdds().list_available_params()
print([p for p in params if p['league'] == 'England' and p['division'] == 1 and p['year'] >= 2024])
```

### OddsApi

Time-stamped odds for any sport, live and historical, from [The Odds API](https://the-odds-api.com) with your key. It
has no sport of its own until you pair it with a statistics source, so its `sport` is `None`. It takes its settings at
construction and carries them as attributes: the markets and regions to price, the moments to fetch, and the environment
variable that holds the key. `list_available_params` then reports only what your key and settings cover.

```python exec="1" source="block" result="text"
from sportsbet.sources import OddsApi

odds = OddsApi(key_env='ODDS_API_KEY', markets=['h2h'], regions=['eu'])
print((odds.name, odds.kind, odds.sport))
print((odds.key_env, odds.markets, odds.regions, odds.moments))
```

`key_env` names the environment variable, so the key stays out of your code.

## Sample sources for offline use

[`SampleSoccerStats`][sportsbet.sources.SampleSoccerStats] and [`SampleSoccerOdds`][sportsbet.sources.SampleSoccerOdds]
carry one frozen soccer season, shipped with the library so the examples run offline. They read from a bundled file,
with no network, and have the same interface as the other sources.

```python exec="1" source="block" result="text"
from sportsbet.sources import SampleSoccerStats

print(SampleSoccerStats().list_available_params())
```

A source turns its feed into snapshots, one row per match and moment, in the long format the library models. Here are
the first odds snapshots of the sample season.

```python exec="1" source="block" result="text"
from sportsbet.sources import SampleSoccerOdds, fetch_payloads

source = SampleSoccerOdds()
payloads = fetch_payloads(source.list_required_items(source.list_available_params()), source.request_url)
snapshots = source.to_snapshots(payloads)
print(snapshots[['home_team', 'away_team', 'provider', 'home_win', 'draw', 'away_win']].head(3))
```

## Writing your own source

A source declares what to read, and the library reads the items it declares and hands the payloads back. A source is
therefore a plain description of a feed, which makes it easy to write and to test. Sources do not talk to each other.
The library collects the items each source declares, reads them, and hands the payloads back. When a statistics and an
odds source share a name, like the free football-data pair, the odds already carry the statistics' identity.

### The methods to implement

A source subclasses [`BaseStatsSource`][sportsbet.sources.BaseStatsSource] or
[`BaseOddsSource`][sportsbet.sources.BaseOddsSource] and implements four methods.

- `list_index_items(selection=None)` returns the items to read to discover what the source publishes.
- `read_catalogue(payloads)` turns those index payloads into the available `league`, `division` and `year` combinations.
- `list_required_items(params, schedule=None)` returns the items to read for a selected set of parameters.
- `to_snapshots(payloads)` turns those payloads into the long snapshots table.

The rest have defaults you rarely change. `list_fixtures_items` reads the same items as training, `needs_schedule` says
the source carries its own schedule, and `request_url` fetches an item from its URL. Override them for a source whose
upcoming matches live elsewhere, whose odds are addressed by timestamp, or whose requests carry a credential.

### The items a source declares

A [`RawItem`][sportsbet.sources.RawItem] is one thing to read: a URL, or a `file://` path for a feed that ships with the
library. Two items with the same `source` and `key` are equal, so the library can tell when two declarations point at
the same thing.

```python exec="1" source="block" result="text"
from sportsbet.sources import RawItem

item = RawItem(source='my_stats', key='England_1_2025', url='https://example.com/2025.csv')
print(item.key)
```

A [`RawPayload`][sportsbet.sources.RawPayload] is what came back, kept verbatim. The library hands it to your
`read_catalogue` and `to_snapshots`.

```python exec="1" source="block" result="text"
from sportsbet.sources import RawItem, RawPayload

item = RawItem(source='my_stats', key='England_1_2025', url='https://example.com/2025.csv')
payload = RawPayload(item=item, content=b'date,home_team,away_team\n2025-08-16,Arsenal,Chelsea\n')
print((payload.item.key, payload.content))
```

### A statistics and an odds feed

Here are two feeds for a league the library does not ship, one for statistics and one for odds. Each declares its items
and turns the payloads into snapshots.

```python exec="1" source="block"
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

That is the whole source. You declare no grammar: the library derives it from the snapshots `to_snapshots` returns. The
markets come from the odds columns, the providers from the odds `provider` column, the features from the statistics
columns, and the moments from `event_status` and `event_time`. Drop `draw` from `MARKETS` and you have a sport with no
draw, and the bettor works out the two-way market on its own.

Follow four rules.

1. Keep the four methods pure. `list_index_items`, `read_catalogue`, `list_required_items` and `to_snapshots` declare
   and transform. The library does the reading, so a source stays testable offline.
2. `date` is the kick-off instant, in UTC. Resolve your feed's time zone at your boundary, so `date + event_time` is the
   wall-clock instant of a snapshot. That instant is the address an odds vendor is asked for. The feeds the library
   ships show how this varies: football-data publishes every league in UK time, which the library converts to UTC, while
   the EuroLeague API already reports UTC. Check your feed's time zone before you trust it.
3. The upcoming matches come from `list_fixtures_items`. The default reads the same items as training. That suits a feed
   whose season file already lists the matches still to be played. Override it when they live elsewhere.
4. Credentials go in `request_url`. The `RawItem` is what the transform sees and what you might save, so a key stays out
   of it.

### Describing your own columns

The library validates snapshots against [pandera] schemas that it builds from the data. You rarely need to write one.
Write one when you want to require a column or to say at which moments it carries values. Subclass
[`BaseStatsSchema`][sportsbet.sources.BaseStatsSchema] or [`BaseOddsSchema`][sportsbet.sources.BaseOddsSchema] and
describe the columns with [`required_col`][sportsbet.sources.required_col] and
[`optional_col`][sportsbet.sources.optional_col].

```python exec="1" source="block"
from typing import Annotated

import pandas as pd
from sportsbet.sources import BaseStatsSchema, optional_col, required_col


class MySchema(BaseStatsSchema):
    """The statistics of my feed."""

    date: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = required_col()
    league: str = required_col()
    home_team: str = required_col()
    away_team: str = required_col()
    home_goals: float = optional_col(['inplay', 'postplay'], fixed=False)   # varies by moment
    home_form: float = optional_col(['preplay'], fixed=True)              # one value per match
```

An odds schema is the same, with a `provider`.

```python exec="1" source="block"
import pandas as pd
from typing import Annotated
from sportsbet.sources import BaseOddsSchema, optional_col, required_col


class MyOddsSchema(BaseOddsSchema):
    """The odds of my feed."""

    date: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = required_col()
    league: str = required_col()
    home_team: str = required_col()
    away_team: str = required_col()
    provider: str = required_col()
    home_win: float = optional_col(['preplay', 'inplay'], fixed=False)
```

`fixed` sets whether a column keeps a bare name or is expanded per moment. A fixed column like `home_form` keeps its
bare name. A column that is not fixed like `home_goals` becomes `home_goals__inplay__45min`.

### Name resolution

Two sources rarely spell a club the same way. One calls a club `Man United` and the other `Manchester United`, and a
name that fails to match silently loses its odds. The library reconciles them: it compares the spellings, pairs the two
feeds, and drops a name it cannot place rather than guess. When the pairing leaves a club unmatched, you pass an alias.
You can reconcile two tables yourself with [`resolve_odds`][sportsbet.sources.resolve_odds], which returns the odds
carrying the identity of the statistics.

```python exec="1" source="block" result="text"
from sportsbet.sources import SampleSoccerStats, SampleSoccerOdds, fetch_payloads, resolve_odds


def snapshots(source):
    items = source.list_required_items(source.list_available_params())
    return source.to_snapshots(fetch_payloads(items, source.request_url))


stats, odds = snapshots(SampleSoccerStats()), snapshots(SampleSoccerOdds())
paired = resolve_odds(stats, odds)
print(paired[['home_team', 'away_team', 'provider', 'home_win']].head(3))
```
