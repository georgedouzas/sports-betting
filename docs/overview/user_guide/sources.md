[pandas DataFrame]: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>
[pandera]: <https://pandera.readthedocs.io>

# Data sources

A dataloader does not know where its data comes from. It asks a **source**. That is why adding a sport is adding a
source, why you can pay for odds without paying for statistics, and why you can plug in a feed the library has never
heard of without changing a line of it.

The sources that ship with the library:

| Source | Sport | Cost | Carries |
| --- | --- | --- | --- |
| [`FootballDataStats`][sportsbet.datasets.FootballDataStats] | soccer | free | statistics, 1993 onward |
| [`FootballDataOdds`][sportsbet.datasets.FootballDataOdds] | soccer | free | pre-match closing odds |
| [`EuroLeagueStats`][sportsbet.datasets.EuroLeagueStats] | basketball | free | statistics, from the competition's own API |
| [`OddsApi`][sportsbet.datasets.OddsApi] | any | **your key** | time-stamped odds, live and historical |

Mix them however you like. Free statistics with paid odds is the realistic configuration, and the only way to backtest an
in-play bet:

```python
from sportsbet.datasets import SoccerDataLoader, FootballDataStats, OddsApi

dataloader = SoccerDataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2025]},
    stats=FootballDataStats(),                                   # free
    odds=OddsApi(key='...', markets=['h2h', 'totals']),          # yours
)
```

## The contract

A source answers four questions, and **never fetches**. The store does the fetching, which is what makes
[`prepare(dry_run=True)`](dataloader.md#preparing-the-data) free and makes an extraction structurally incapable of
downloading anything.

```python
class BaseSource:

    name: ClassVar[str]      # who declared an item
    kind: ClassVar[str]      # 'stats' or 'odds'

    def index_items(self) -> list[RawItem]:
        """What do I need to read to know what I publish? Always free."""

    def catalogue(self, payloads) -> list[dict]:
        """Given those, which league/division/year combinations do I publish?"""

    def required_items(self, params, schedule=None) -> list[RawItem]:
        """Given a selection, what do I need fetched?"""

    def to_snapshots(self, payloads) -> pd.DataFrame:
        """Given those, what are the long snapshots?"""
```

Three optional hooks:

- `estimate(items)` — what this would cost. The default sums the items' `cost`, which is `0` for a free source.
- `request_url(item)` — add a credential *at the moment of the request*, so it never reaches a stored item.
- `needs_schedule()` — return `True` if you address your data by *instant* rather than by season. `OddsApi` does: "the
  price at minute 45" is a timestamp, and it can only build it once it knows the kick-off.

### `RawItem` and `RawPayload`

A [`RawItem`][sportsbet.datasets.RawItem] is one thing to fetch. It is the unit of caching, of resuming, and of **cost**:

```python
RawItem(
    source='my_stats',                    # who declared it
    key='England_1_2025',                 # its identity, and its file name in the store
    url='https://example.com/2025.csv',
    volatile=False,                       # True if it can still change upstream
    cost=0,                               # what the source charges to fetch it
)
```

Two sources declaring the **same** `source` and `key` declare the *same* item, so it is fetched **once**. That is how
`FootballDataStats` and `FootballDataOdds` — which read the same upstream CSV — avoid downloading it twice.

A [`RawPayload`][sportsbet.datasets.RawPayload] is what came back, kept verbatim and kept forever. It is what your
`catalogue` and `to_snapshots` are handed:

```python
from sportsbet.datasets import RawPayload

payload = RawPayload(item=item, content=b'date,home_team,away_team\n...')
payload.item.key      # 'England_1_2025'
payload.content       # exactly what the feed returned, unparsed
```

## Writing your own source

Everything below runs. Two feeds, a statistics one and an odds one, for a league the library has never heard of.

```python
import io
import json

import pandas as pd
from sportsbet.datasets import BaseOddsSource, BaseStatsSource, RawItem, market_outcomes

MARKETS = ['home_win', 'draw', 'away_win']
IDENTITY = ['date', 'league', 'division', 'year', 'home_team', 'away_team']


class MyStats(BaseStatsSource):
    """Statistics from a feed of your own."""

    name = 'my_stats'

    def index_items(self):
        return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json', volatile=True)]

    def catalogue(self, payloads):
        seasons = json.loads(payloads[0].content)
        return [{'league': 'Ruritania', 'division': 1, 'year': year} for year in seasons]

    def required_items(self, params, schedule=None):
        return [
            RawItem(
                source=self.name,
                key=f'Ruritania_1_{param["year"]}',
                url=f'https://example.com/{param["year"]}.csv',
                volatile=param['year'] >= 2026,       # a finished season never changes
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
        return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json', volatile=True)]

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

Then hand them to any dataloader:

```python
from sportsbet.datasets import SoccerDataLoader

dataloader = SoccerDataLoader(stats=MyStats(), odds=MyOdds())
dataloader.prepare()
X, Y, O = dataloader.extract_train_data(odds_type='acme')
```

```text
X   ['league', 'division', 'year', 'home_team', 'away_team', 'home_form', 'away_form']
Y   ['home_win__postplay__0min', 'draw__postplay__0min', 'away_win__postplay__0min']
O   ['acme__home_win__preplay__0min', 'acme__draw__preplay__0min', 'acme__away_win__preplay__0min']
```

Nothing was configured. The **markets** came from the odds columns, the **providers** from the odds `provider` column,
the **features** from the statistics columns, and the **moments** from `event_status`/`event_time`. Drop `draw` from
`MARKETS` and you have a sport that cannot be drawn, and the bettor works out the two-way market on its own.

### Four rules that are not style

1. **Never fetch.** `index_items`, `catalogue`, `required_items` and `to_snapshots` must be pure. If a source could
   fetch, a dry run could not be free and an extraction could download by accident.
2. **`date` is the kick-off instant, in UTC.** Resolve your feed's time zone *at your boundary*. This is what makes
   `date + event_time` the wall-clock instant of a snapshot — the address an odds vendor is asked for. Both feeds the
   library ships got this wrong in an undocumented way: football-data publishes **every** league in UK time, and the
   EuroLeague publishes **every** game in Central European time. Neither says so. Assume nothing.
3. **A finished season is not `volatile`; a fixture is.** That is what makes `prepare()` incremental.
4. **Credentials go in `request_url`**, never in a `RawItem`. An item is written to the store; a key must not be.

## Where the data is kept

The [store][sportsbet.datasets.LocalStore] is the only thing that fetches.

```python
from sportsbet.datasets import LocalStore

dataloader = SoccerDataLoader(store=LocalStore('/data/sportsbet'))   # default: ~/.sportsbet
```

Raw payloads are kept **forever** — metered data cannot be re-obtained for free — and everything derived from them is
rebuilt at no cost. The derived data is keyed by the raw content **and by the code that transformed it**, so upgrading
the library rebuilds rather than serving you what the old transform produced.

To write your own store — a shared cache, an object store, a database — implement
[`BaseStore`][sportsbet.datasets.BaseStore]:

```python
from sportsbet.datasets import BaseStore, RawItem, RawPayload


class MyStore(BaseStore):
    """A store of your own."""

    def held(self, items: list[RawItem]) -> list[RawItem]:
        """Which of these do I already have, and cannot have changed upstream?"""

    def fetch(self, items: list[RawItem], authorize=None) -> list[RawPayload]:
        """Download these and keep them. The only place anything is fetched."""

    def read(self, items: list[RawItem]) -> list[RawPayload]:
        """Give me back what I have. Raise if I do not have it."""
```

`authorize` is the source's `request_url`, so a credential reaches the request and never the store.

### What a preparation tells you

`prepare` returns a [`PreparationReport`][sportsbet.datasets.PreparationReport]:

```python
from sportsbet.datasets import PreparationReport

report: PreparationReport = dataloader.prepare(dry_run=True)

report.to_fetch          # the items that would be downloaded
report.held              # the items already in the store
report.estimated_cost    # {'odds_api': 8642} -- exact, and it cost nothing to learn
report.unavailable       # requested parameters no source publishes
```

An extraction against a store that was never prepared raises
[`NotPreparedError`][sportsbet.datasets.NotPreparedError], carrying that report — so it tells you what is missing *and*
what obtaining it would cost. It never downloads.

## When two sources disagree about a name

Mixing sources means one calls a club `Man United` and the other calls it `Manchester United`. If a name fails to match,
that game has no odds — and **a missing odd does not look like an error**. It looks like a slightly smaller dataset, and
a backtest that is clean, plausible and wrong.

So they are reconciled, and the result is a hard gate:

```python
X, Y, O = dataloader.extract_train_data(odds_type='pinnacle')
dataloader.reconciliation_        # a ReconciliationReport
```

A [`ReconciliationReport`][sportsbet.datasets.ReconciliationReport] carries `matched`, `unmatched_rate`,
`unmatched_stats`, `unmatched_odds`, and `suggestions`. Cross `max_unmatched_rate` (default **zero**) and you get
[`UnmatchedError`][sportsbet.datasets.UnmatchedError] rather than a holed dataset:

```python
from sportsbet.datasets import UnmatchedError

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

Check it — a suggestion is a resemblance, not a fact — and pass it back as `aliases={...}`. See
[the dataloader guide](dataloader.md#when-two-sources-name-a-club-differently) for why the library will never apply one
on its own.

You can reconcile two tables yourself with [`resolve`][sportsbet.datasets.resolve]:

```python
from sportsbet.datasets import resolve

odds, report = resolve(stats, odds, aliases={'Olimpia Milano': 'EA7 Emporio Armani Milan'})
```

## Describing your own columns

Snapshots are validated against [pandera] schemas that the library **builds from the data**. You rarely need to write
one. When you do — to require a column, or to say at which moments it carries values — subclass
[`BaseStatsSchema`][sportsbet.datasets.BaseStatsSchema] or [`BaseOddsSchema`][sportsbet.datasets.BaseOddsSchema] and
describe the columns with [`required_col`][sportsbet.datasets.required_col] and
[`optional_col`][sportsbet.datasets.optional_col]:

```python
from typing import Annotated
import pandas as pd
from sportsbet.datasets import BaseStatsSchema, optional_col, required_col


class MySchema(BaseStatsSchema):
    """The statistics of my feed."""

    date: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = required_col()
    league: str = required_col()
    home_team: str = required_col()
    away_team: str = required_col()
    home_goals: int = optional_col(['inplay', 'postplay'], fixed=False)   # varies by moment
    home_form: float = optional_col(['preplay'], fixed=True)              # one value per match
```

An odds schema is the same, with a `provider`:

```python
from sportsbet.datasets import BaseOddsSchema


class MyOddsSchema(BaseOddsSchema):
    """The odds of my feed."""

    date: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = required_col()
    league: str = required_col()
    home_team: str = required_col()
    away_team: str = required_col()
    provider: str = optional_col(['preplay'], fixed=True)
    home_win: float = optional_col(['preplay', 'inplay'], fixed=False)
```

`fixed` is the difference between a column that keeps a bare name (`home_form`) and one that is expanded per moment
(`home_goals__inplay__45min`).

## The abstract classes

[`BaseDataLoader`][sportsbet.datasets.BaseDataLoader] is the extraction engine. Its **one** abstract method is
`_snapshots()`, returning the long `stats` and `odds` tables. Everything else — the column grammar, the input horizon,
the moment-aware pivot — is implemented for you:

```python
from sportsbet.datasets import BaseDataLoader


class MyDataLoader(BaseDataLoader):
    """A dataloader whose data comes from wherever you like."""

    def _snapshots(self):
        return my_stats_table, my_odds_table
```

That is the seam. `SoccerDataLoader` and `BasketballDataLoader` sit behind it and differ only in which sources they
default to — which is why adding a sport is adding a source, and
[`from_snapshots`](dataloader.md#from-long-snapshots) can hand it a table you built yourself.

[`BaseBettor`][sportsbet.evaluation.BaseBettor] is the betting strategy. Implement `_fit` and `_predict_proba` and you
get value bets, backtesting and hyperparameter search — see [the bettor guide](bettor.md#implementation).
