"""Implements the base classes of the data sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import hashlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import pandas as pd

if TYPE_CHECKING:
    from .. import ParamGrid
    from ._store import BaseStore


@dataclass(frozen=True)
class RawItem:
    """A unit of fetching, caching, resuming and cost.

    Two sources declaring the same `source` and `key` declare the same item, so data shared by a statistics and an odds
    source is fetched once rather than twice.

    Attributes:
        source:
            The name of the source that declared it.

        key:
            The identity of the item within the source.

        url:
            Where to fetch it from.

        volatile:
            Whether the content can still change upstream.

    Examples:
        >>> from sportsbet.sources import RawItem
        >>> item = RawItem(
        ...     source='my_stats',
        ...     key='England_1_2025',
        ...     url='https://example.com/2025.csv',
        ...     volatile=False,
        ... )
        >>> item.key
        'England_1_2025'
        >>> # A finished season cannot change, so it is fetched once and kept.
        >>> item.volatile
        False
        >>> # The same source and key is the same item, so it is never fetched twice.
        >>> item == RawItem(source='my_stats', key='England_1_2025', url='https://example.com/2025.csv')
        True
    """

    source: str
    key: str
    url: str
    volatile: bool = False


@dataclass(frozen=True)
class RawPayload:
    r"""What a source returned, kept verbatim.

    Attributes:
        item:
            What was asked for.

        content:
            Exactly what came back.

    Examples:
        >>> from sportsbet.sources import RawItem, RawPayload
        >>> item = RawItem(source='my_stats', key='England_1_2025', url='https://example.com/2025.csv')
        >>> payload = RawPayload(item=item, content=b'date,home_team,away_team\n2025-08-16,A,B\n')
        >>> payload.item.key
        'England_1_2025'
        >>> # The bytes are exactly what the feed returned, so a transform can be changed and replayed for free.
        >>> payload.content.splitlines()[0]
        b'date,home_team,away_team'
    """

    item: RawItem
    content: bytes


class BaseSource(ABC):
    """The abstract base class for data sources.

    A source knows which sport it carries, since a feed of soccer matches is a feed of soccer matches whatever is done
    with it. `sport` is `None` for a source that carries several, as a vendor of odds does, and such a source takes the
    sport of the one it is paired with.

    A source declares the raw items a selection of parameters needs and turns the returned payloads into long snapshots.
    Its planning and transform methods never fetch, so what a download would take can always be counted without making
    it, and an extraction can never download by accident.

    It also answers what it publishes, through `available_params`. That question has to be answerable before a
    `param_grid` is written, so it belongs here and not on a dataloader that is configured with one.

    Examples:
        >>> from sportsbet.sources import FootballDataStats, OddsApi
        >>> stats = FootballDataStats()
        >>> stats.name, stats.kind, stats.sport
        ('football_data', 'stats', 'soccer')
        >>> # A vendor selling every sport carries none of its own, and takes the sport it is paired with.
        >>> OddsApi(key='...', markets=['h2h']).sport is None
        True
        >>> # Asking what a source publishes declares items rather than fetching them.
        >>> items = stats.index_items()
        >>> items[0].source, items[0].volatile
        ('football_data', True)
    """

    name: ClassVar[str]
    kind: ClassVar[str]
    sport: ClassVar[str | None] = None

    @abstractmethod
    def index_items(self: Self, selection: ParamGrid | None = None) -> list[RawItem]:
        """Return the items needed to discover what the source publishes.

        A feed that lists its seasons on an index page is cheap to ask. One that publishes a league as a single file of
        every season it ever played has no index, so the file is the catalogue, and reading it means downloading it.
        Asking such a feed what it publishes therefore costs as much as the data.

        So a source is told what is being looked for, and answers with what it needs in order to place it. A selection
        of one league has no business downloading another.

        Args:
            selection:
                What is being looked for. `None` asks for everything the source publishes, which is what discovery
                needs, since nothing can be selected before it is known what exists.

        Returns:
            items:
                The items of the catalogue. Always free, so a preparation can be priced without spending anything.
        """

    @abstractmethod
    def catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the parameter combinations the index payloads describe.

        Args:
            payloads:
                The payloads of the index items.

        Returns:
            params:
                The available `league`, `division` and `year` combinations.
        """

    def available_params(self: Self, store: BaseStore | None = None, refresh: bool = False) -> list[dict]:
        """Return the parameter combinations the source publishes.

        This is where you start: a `param_grid` cannot be written before it is known what exists, so it needs no
        dataloader. The catalogue is free and it is re-read, so a new season shows up as soon as it is published.

        What is published depends on how the source is configured, since a credential may only cover part of what the
        source offers.

        Args:
            store:
                Where the catalogue is kept. The default `None` keeps it in `~/.sportsbet`.

            refresh:
                If `True`, everything is read again, including what the store already holds.

        Returns:
            params:
                The available `league`, `division` and `year` combinations.
        """
        from ._store import LocalStore  # noqa: PLC0415

        store = store if store is not None else LocalStore()
        items = self.index_items()
        held = [] if refresh else store.held(items)
        store.fetch([item for item in items if item not in held], self.request_url)
        return self.catalogue(store.read(items))

    @abstractmethod
    def required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return the raw items the selected parameters need.

        Args:
            params:
                The selected parameter combinations.

            schedule:
                The matches of the selected parameters, with their kick-off instants. An odds source that addresses its
                prices by timestamp needs it, since a season alone does not say when its matches are played. It is
                `None` for a source that carries its own schedule.

        Returns:
            items:
                The items to fetch. Deterministic for the same parameters.
        """

    def needs_schedule(self: Self) -> bool:
        """Return whether the source has to be told when the matches are.

        A source that carries its own schedule, because its events and its odds arrive in the same file, does not.

        Returns:
            needed:
                Whether `required_items` needs a schedule.
        """
        return False

    def request_url(self: Self, item: RawItem) -> str:
        """Return the URL to fetch an item from.

        The credential is added here, at the moment of the request, so it never reaches a `RawItem` and is never
        written to the store.

        Args:
            item:
                The item to fetch.

        Returns:
            url:
                Where to fetch it from.
        """
        return item.url

    @abstractmethod
    def to_snapshots(self: Self, payloads: list[RawPayload]) -> pd.DataFrame:
        """Transform the raw payloads into a long snapshot table.

        Args:
            payloads:
                The payloads of the required items.

        Returns:
            snapshots:
                The long snapshots.
        """

    def transform_digest(self: Self) -> str:
        """Return the identity of the code that turns the payloads into snapshots.

        The snapshots are derived by code as well as from data, so what is cached against them has to know when the
        code changes. A release version cannot: an editable install keeps the version it was installed with, so an
        edited transform would keep serving what the previous one produced.

        Returns:
            digest:
                The identity of the transform.
        """
        try:
            module = Path(inspect.getfile(type(self)))
            return hashlib.sha256(module.read_bytes()).hexdigest()[:16]
        except (OSError, TypeError):
            return type(self).__name__


class BaseStatsSource(BaseSource):
    r"""The abstract base class for statistics sources.

    Examples:
        >>> import io
        >>> import pandas as pd
        >>> from sportsbet.sources import BaseStatsSource, RawItem, RawPayload, market_outcomes
        >>> IDENTITY = ['date', 'league', 'division', 'year', 'home_team', 'away_team']
        >>>
        >>> class MyStats(BaseStatsSource):
        ...     '''Statistics from a feed of your own.'''
        ...
        ...     name = 'my_stats'
        ...     sport = 'soccer'
        ...
        ...     def index_items(self, selection=None):
        ...         return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json',
        ...                         volatile=True)]
        ...
        ...     def catalogue(self, payloads):
        ...         return [{'league': 'Ruritania', 'division': 1, 'year': 2025}]
        ...
        ...     def required_items(self, params, schedule=None):
        ...         return [RawItem(source=self.name, key=f'Ruritania_1_{param["year"]}',
        ...                         url=f'https://example.com/{param["year"]}.csv')
        ...                 for param in params]
        ...
        ...     def to_snapshots(self, payloads):
        ...         games = pd.read_csv(io.BytesIO(payloads[0].content))
        ...         games['date'] = pd.to_datetime(games['date'], utc=True)
        ...         preplay = games[IDENTITY].assign(event_status='preplay', event_time=0,
        ...                                          home_form=games['home_form'])
        ...         postplay = games[IDENTITY].assign(event_status='postplay', event_time=0)
        ...         outcomes = market_outcomes(games['home_goals'], games['away_goals'], ['home_win', 'draw',
        ...                                                                               'away_win'])
        ...         postplay = pd.concat([postplay, outcomes], axis=1)
        ...         return pd.concat([preplay, postplay], ignore_index=True)
        >>>
        >>> source = MyStats()
        >>> # It never fetches. It says what it needs, and the store gets it.
        >>> source.required_items([{'year': 2025}])[0].url
        'https://example.com/2025.csv'
        >>> csv = b'date,league,division,year,home_team,away_team,home_form,home_goals,away_goals\n'
        >>> csv += b'2025-08-16,Ruritania,1,2025,A,B,0.5,2,1\n'
        >>> snapshots = source.to_snapshots([RawPayload(item=source.required_items([{'year': 2025}])[0], content=csv)])
        >>> sorted(snapshots['event_status'].unique())
        ['postplay', 'preplay']
        >>> snapshots.loc[snapshots['event_status'].eq('postplay'), 'home_win'].item()
        1.0
    """

    kind: ClassVar[str] = 'stats'


class BaseOddsSource(BaseSource):
    r"""The abstract base class for odds sources.

    Examples:
        >>> import io
        >>> import pandas as pd
        >>> from sportsbet.sources import BaseOddsSource, RawItem, RawPayload
        >>>
        >>> class MyOdds(BaseOddsSource):
        ...     '''Odds from a feed of your own.'''
        ...
        ...     name = 'my_odds'
        ...     sport = 'soccer'
        ...
        ...     def index_items(self, selection=None):
        ...         return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json',
        ...                         volatile=True)]
        ...
        ...     def catalogue(self, payloads):
        ...         return [{'league': 'Ruritania', 'division': 1, 'year': 2025}]
        ...
        ...     def required_items(self, params, schedule=None):
        ...         return [RawItem(source=self.name, key=f'odds_{param["year"]}',
        ...                         url=f'https://example.com/odds/{param["year"]}.csv')
        ...                 for param in params]
        ...
        ...     def to_snapshots(self, payloads):
        ...         odds = pd.read_csv(io.BytesIO(payloads[0].content))
        ...         odds['date'] = pd.to_datetime(odds['date'], utc=True)
        ...         return odds.assign(event_status='preplay', event_time=0)
        >>>
        >>> source = MyOdds()
        >>> source.kind
        'odds'
        >>> csv = b'date,league,division,year,home_team,away_team,provider,home_win,draw,away_win\n'
        >>> csv += b'2025-08-16,Ruritania,1,2025,A,B,acme,1.8,3.4,4.2\n'
        >>> item = source.required_items([{'year': 2025}])[0]
        >>> snapshots = source.to_snapshots([RawPayload(item=item, content=csv)])
        >>> # The markets are the columns, and the provider is a column too, so nothing has to be registered.
        >>> snapshots[['provider', 'home_win', 'event_status']].to_dict('records')
        [{'provider': 'acme', 'home_win': 1.8, 'event_status': 'preplay'}]
    """

    kind: ClassVar[str] = 'odds'
