"""Define the base a data source implements and read its raw content."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import asyncio
import io
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self
from urllib.parse import urlparse
from urllib.request import url2pathname

import aiohttp
import pandas as pd

from ..core import ParamGrid

CONNECTIONS_LIMIT = 20
ENCODING = 'ISO-8859-1'
FILE_SCHEME = 'file://'


@dataclass(frozen=True)
class RawItem:
    """A raw item to fetch, identified within its source by a key, at a URL or `file://` path.

    Two items with the same source and key are equal.

    Args:
        source:
            The source that declared it.
        key:
            The item's identity within the source.
        url:
            The URL, or a `file://` path for a bundled file.

    Examples:
        >>> from sportsbet.sources import RawItem
        >>> item = RawItem(source='my_stats', key='England_1_2025', url='https://example.com/2025.csv')
        >>> item.key
        'England_1_2025'
        >>> # The same source and key make the same item.
        >>> item == RawItem(source='my_stats', key='England_1_2025', url='https://example.com/2025.csv')
        True
    """

    source: str
    key: str
    url: str


@dataclass(frozen=True)
class RawPayload:
    r"""A payload a source returned, pairing the fetched item with its raw bytes.

    Args:
        item:
            The item that was fetched.
        content:
            The bytes of the response.

    Examples:
        >>> from sportsbet.sources import RawItem, RawPayload
        >>> item = RawItem(source='my_stats', key='England_1_2025', url='https://example.com/2025.csv')
        >>> payload = RawPayload(item=item, content=b'date,home_team,away_team\n2025-08-16,A,B\n')
        >>> payload.item.key
        'England_1_2025'
        >>> # The bytes are exactly what the feed returned.
        >>> payload.content.splitlines()[0]
        b'date,home_team,away_team'
    """

    item: RawItem
    content: bytes


async def _fetch_url(
    client: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str] | None = None,
) -> str:
    """Return the text of a URL, read over the network."""

    async with client.get(url, headers=headers or {}) as response:
        return await response.text(encoding=ENCODING)


async def _fetch_urls(requests: list[tuple[str, dict[str, str]]]) -> list[str]:
    """Return the text of several URLs, read over the network at once."""

    async with aiohttp.ClientSession(
        raise_for_status=True,
        connector=aiohttp.TCPConnector(limit=CONNECTIONS_LIMIT),
    ) as client:
        return await asyncio.gather(*[_fetch_url(client, url, headers) for url, headers in requests])


def _read_local_file(url: str) -> bytes:
    """Return the bytes of a `file://` URL, read from disk."""
    return Path(url2pathname(urlparse(url).path)).read_bytes()


def _read_urls_content(requests: list[tuple[str, dict[str, str]]]) -> list[bytes]:
    """Return the content behind each URL, from disk for a `file://` URL and over the network for the rest."""
    remote = [(url, headers) for url, headers in requests if not url.startswith(FILE_SCHEME)]
    fetched = iter(asyncio.run(_fetch_urls(remote)) if remote else [])
    return [
        _read_local_file(url) if url.startswith(FILE_SCHEME) else next(fetched).encode(ENCODING)
        for url, _headers in requests
    ]


def fetch_payloads(
    items: list[RawItem],
    authorize: Callable[[RawItem], str],
    headers: Callable[[RawItem], dict[str, str]] | None = None,
) -> list[RawPayload]:
    """Read each item at the URL `authorize` gives it and pair the bytes back with the item, in order.

    Args:
        items:
            The items to read.

        authorize:
            A callable turning an item into the URL to fetch it from, adding any credential.

        headers:
            An optional callable turning an item into request headers, for credentials that travel as
            headers rather than query parameters.

    Returns:
        The payloads, each pairing an item with its bytes, in the order given.
    """
    header_for = headers if headers is not None else (lambda _item: {})
    contents = _read_urls_content([(authorize(item), header_for(item)) for item in items])
    return [RawPayload(item=item, content=content) for item, content in zip(items, contents, strict=True)]


def read_csv_content(content: bytes) -> pd.DataFrame:
    """Return a data frame read from raw CSV content.

    Args:
        content:
            The raw CSV bytes.

    Returns:
        The parsed data frame.
    """
    text = content.decode(ENCODING)
    names = pd.read_csv(io.StringIO(text), nrows=0, encoding=ENCODING).columns.to_list()
    return pd.read_csv(io.StringIO(text), names=names, skiprows=1, encoding=ENCODING, on_bad_lines='skip')


class BaseSource(ABC):
    """The abstract base class for data sources.

    A source declares the raw items a selection of parameters needs. It turns the returned payloads into long
    snapshots. `sport` names the one sport it carries. `sport` is `None` for a vendor that carries several sports and
    takes the sport of the source it is paired with. The `list_available_params` method returns what the source
    publishes.

    Examples:
        >>> from sportsbet.sources import FootballDataStats, OddsApi
        >>> stats = FootballDataStats()
        >>> stats.name, stats.kind, stats.sport
        ('football_data', 'stats', 'soccer')
        >>> # A vendor selling every sport carries none of its own, and takes the sport it is paired with.
        >>> OddsApi(key_env='ODDS_API_KEY', markets=['h2h']).sport is None
        True
        >>> # Asking what a source publishes declares the items to read.
        >>> items = stats.list_index_items()
        >>> items[0].source
        'football_data'
    """

    name: ClassVar[str]
    kind: ClassVar[str]
    sport: ClassVar[str | None] = None

    @abstractmethod
    def list_index_items(self: Self, selection: ParamGrid | None = None) -> list[RawItem]:
        """Return the items needed to discover what the source publishes.

        Args:
            selection:
                What is being looked for. `None` asks for everything.

        Returns:
            items:
                The items whose payloads describe the catalogue.
        """

    @abstractmethod
    def read_catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the parameter combinations the index payloads describe.

        Args:
            payloads:
                The payloads of the index items.

        Returns:
            params:
                The available `league`, `division` and `year` combinations.
        """

    def list_available_params(self: Self) -> list[dict]:
        """Return the league, division and season combinations the source publishes.

        What a source publishes depends on how it is configured.

        Returns:
            params:
                The available `league`, `division` and `year` combinations.
        """
        return self.read_catalogue(self.fetch_items(self.list_index_items()))

    @abstractmethod
    def list_required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return the raw items the selected parameters need.

        Args:
            params:
                The selected parameter combinations.

            schedule:
                The matches of the selected parameters, with their kick-off instants. An odds source that addresses its
                prices by timestamp needs it. It is `None` for a source that carries its own schedule.

        Returns:
            items:
                The items to read. Deterministic for the same parameters.
        """

    def list_fixtures_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return the raw items the upcoming matches need.

        By default it returns the same items training needs. That suits a source whose season file already carries the
        matches still to be played. It also suits an odds source that prices whatever the schedule lists. A source
        whose upcoming matches live elsewhere, in a separate fixtures file or the season in progress, overrides this
        method.

        Args:
            params:
                The selected parameter combinations.

            schedule:
                The upcoming matches, for an odds source that prices by instant.

        Returns:
            items:
                The items whose payloads yield the upcoming matches.
        """
        return self.list_required_items(params, schedule)

    def needs_schedule(self: Self) -> bool:
        """Return whether the source has to be told when the matches are.

        A source whose events and odds arrive in the same file carries its own schedule and returns `False`.

        Returns:
            needed:
                Whether `list_required_items` needs a schedule.
        """
        return False

    def request_url(self: Self, item: RawItem) -> str:
        """Return the URL to fetch an item from, with the credential added at the moment of the request.

        Args:
            item:
                The item to fetch.

        Returns:
            url:
                Where to fetch it from.
        """
        return item.url

    def request_headers(self: Self, item: RawItem) -> dict[str, str]:
        """Return the headers to send when fetching an item, with credentials added at request time.

        Args:
            item:
                The item to fetch.

        Returns:
            headers:
                The headers to send. The default is none.
        """
        return {}

    def fetch_items(self: Self, items: list[RawItem]) -> list[RawPayload]:
        """Fetch each item and return its payload.

        The default reads each item with a GET at `request_url`, sending `request_headers`. A source whose vendor needs
        more than one request per item, or a method other than GET, overrides this method and keeps the planning methods
        pure.

        Args:
            items:
                The items to read.

        Returns:
            payloads:
                The payloads, each pairing an item with its bytes, in the order given.
        """
        return fetch_payloads(items, self.request_url, self.request_headers)

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


class BaseStatsSource(BaseSource):
    r"""The abstract base class for statistics sources.

    Examples:
        >>> import io
        >>> import pandas as pd
        >>> from sportsbet.sources import BaseStatsSource, RawItem, RawPayload, derive_market_outcomes
        >>> IDENTITY = ['date', 'league', 'division', 'year', 'home_team', 'away_team']
        >>>
        >>> class MyStats(BaseStatsSource):
        ...     '''Statistics from a feed of your own.'''
        ...
        ...     name = 'my_stats'
        ...     sport = 'soccer'
        ...
        ...     def list_index_items(self, selection=None):
        ...         return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json')]
        ...
        ...     def read_catalogue(self, payloads):
        ...         return [{'league': 'Ruritania', 'division': 1, 'year': 2025}]
        ...
        ...     def list_required_items(self, params, schedule=None):
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
        ...         outcomes = derive_market_outcomes(games['home_goals'], games['away_goals'], ['home_win', 'draw',
        ...                                                                               'away_win'])
        ...         postplay = pd.concat([postplay, outcomes], axis=1)
        ...         return pd.concat([preplay, postplay], ignore_index=True)
        >>>
        >>> source = MyStats()
        >>> # It says what it needs, and the dataloader reads it.
        >>> source.list_required_items([{'year': 2025}])[0].url
        'https://example.com/2025.csv'
        >>> csv = b'date,league,division,year,home_team,away_team,home_form,home_goals,away_goals\n'
        >>> csv += b'2025-08-16,Ruritania,1,2025,A,B,0.5,2,1\n'
        >>> item = source.list_required_items([{'year': 2025}])[0]
        >>> snapshots = source.to_snapshots([RawPayload(item=item, content=csv)])
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
        ...     def list_index_items(self, selection=None):
        ...         return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json')]
        ...
        ...     def read_catalogue(self, payloads):
        ...         return [{'league': 'Ruritania', 'division': 1, 'year': 2025}]
        ...
        ...     def list_required_items(self, params, schedule=None):
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
        >>> item = source.list_required_items([{'year': 2025}])[0]
        >>> snapshots = source.to_snapshots([RawPayload(item=item, content=csv)])
        >>> # The markets are the columns, and the provider is a column too.
        >>> snapshots[['provider', 'home_win', 'event_status']].to_dict('records')
        [{'provider': 'acme', 'home_win': 1.8, 'event_status': 'preplay'}]
    """

    kind: ClassVar[str] = 'odds'
