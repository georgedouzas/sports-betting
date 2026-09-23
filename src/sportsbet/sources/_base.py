"""Define the base a data source implements, read its raw content, and validate its snapshots."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from __future__ import annotations

import asyncio
import io
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self
from urllib.parse import urlparse
from urllib.request import url2pathname

import aiohttp
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Timedelta

from ..core import STATUSES, ParamGrid

CONNECTIONS_LIMIT = 20
ENCODING = 'ISO-8859-1'
FILE_SCHEME = 'file://'

async def _fetch_url(client: aiohttp.ClientSession, url: str) -> str:
    """Return the text of a URL, read over the network."""

    async with client.get(url) as response:
        return await response.text(encoding=ENCODING)

async def _fetch_urls(urls: list[str]) -> list[str]:
    """Return the text of several URLs, read over the network at once."""

    async with aiohttp.ClientSession(
        raise_for_status=True,
        connector=aiohttp.TCPConnector(limit=CONNECTIONS_LIMIT),
    ) as client:
        return await asyncio.gather(*[_fetch_url(client, url) for url in urls])

def _read_local_file(url: str) -> bytes:
    """Return the bytes of a `file://` URL, read from disk."""
    return Path(url2pathname(urlparse(url).path)).read_bytes()

def _read_urls_content(urls: list[str]) -> list[bytes]:
    """Return the content behind each URL, from disk for a `file://` URL and over the network for the rest."""
    remote = [url for url in urls if not url.startswith(FILE_SCHEME)]
    fetched = iter(asyncio.run(_fetch_urls(remote)) if remote else [])
    return [_read_local_file(url) if url.startswith(FILE_SCHEME) else next(fetched).encode(ENCODING) for url in urls]

def fetch_payloads(items: list[RawItem], authorize: Callable[[RawItem], str]) -> list[RawPayload]:
    """Read each item at the URL `authorize` gives it and pair the bytes back with the item, in order.

    Args:
        items:
            The items to read.

        authorize:
            A callable turning an item into the URL to fetch it from, adding any credential.

    Returns:
        The payloads, each pairing an item with its bytes, in the order given.
    """
    contents = _read_urls_content([authorize(item) for item in items])
    return [RawPayload(item=item, content=content) for item, content in zip(items, contents, strict=True)]

def read_csv_content(content: bytes) -> pd.DataFrame:
    r"""Return a data frame read from raw CSV content.

    Args:
        content:
            The raw CSV bytes.

    Returns:
        The parsed data frame.

    Examples:
        >>> from sportsbet.sources import read_csv_content
        >>> read_csv_content(b'league,goals\nEngland,2\n')
            league  goals
        0  England      2
    """
    text = content.decode(ENCODING)
    names = pd.read_csv(io.StringIO(text), nrows=0, encoding=ENCODING).columns.to_list()
    return pd.read_csv(io.StringIO(text), names=names, skiprows=1, encoding=ENCODING, on_bad_lines='skip')

def required_col(alias: str | None = None) -> Any:  # noqa: ANN401  # varied defaults
    """Define a required snapshot-identity column.

    Args:
        alias:
            The column name to use when it differs from the field's Python
            identifier.

    Returns:
        A pandera field marking the column as a required snapshot-identity column.

    Examples:
        >>> from sportsbet.sources import BaseStatsSchema, required_col
        >>>
        >>> class MyStatsSchema(BaseStatsSchema):
        ...     'The columns a statistics feed of your own must always carry.'
        ...
        ...     home_team: str = required_col()
        ...     away_team: str = required_col()
        >>>
        >>> # A required column may not be missing.
        >>> MyStatsSchema.to_schema().columns['home_team'].nullable
        False
    """
    return pa.Field(nullable=False, metadata={'snapshot': True}, alias=alias)

def optional_col(include: list[str], fixed: bool, alias: str | None = None) -> Any:  # noqa: ANN401  # varied defaults
    """Define an optional feature or odds column.

    Args:
        include:
            The event statuses at which the column is meaningful.
        fixed:
            Whether the column is time-invariant within a match.
        alias:
            The column name to use when it differs from the field's Python
            identifier.

    Returns:
        A pandera field marking the column as an optional feature or odds column.

    Examples:
        >>> from sportsbet.sources import BaseStatsSchema, optional_col, required_col
        >>>
        >>> class MyStatsSchema(BaseStatsSchema):
        ...     'The columns a statistics feed of your own may carry.'
        ...
        ...     home_team: str = required_col()
        ...     away_team: str = required_col()
        ...     home_goals: float = optional_col(include=['inplay', 'postplay'], fixed=False)
        ...     stadium_capacity: float = optional_col(include=['preplay'], fixed=True)
        >>>
        >>> # There is no score before the match starts.
        >>> MyStatsSchema.to_schema().columns['home_goals'].metadata['include']
        ['inplay', 'postplay']
        >>> # A stadium does not change size at half time.
        >>> MyStatsSchema.to_schema().columns['stadium_capacity'].metadata['fixed']
        True
    """
    return pa.Field(nullable=True, metadata={'include': include, 'fixed': fixed}, alias=alias)

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

class BaseSchema(pa.DataFrameModel):
    """Sport-agnostic base schema for event snapshots."""

    event_status: str = required_col()
    event_time: Timedelta = required_col()

    @pa.dataframe_check
    @classmethod
    def check_event_time_vs_status(cls: type[Self], df: pd.DataFrame) -> pd.Series:
        """Check the event time is consistent with the event status.

        Args:
            df:
                The snapshots to check.

        Returns:
            passing:
                True for every row whose event time agrees with its status.
        """
        preplay_check = (df['event_status'] == 'preplay') & (df['event_time'] >= pd.Timedelta(0))
        inplay_check = (df['event_status'] == 'inplay') & (df['event_time'] > pd.Timedelta(0))
        postplay_check = (df['event_status'] == 'postplay') & (df['event_time'] == pd.Timedelta(0))
        status_check = df['event_status'].isin(STATUSES)
        return status_check & (preplay_check | inplay_check | postplay_check)

    @classmethod
    def list_snapshot_cols(cls: type[Self]) -> list[str]:
        """Return the snapshot-identity columns.

        Returns:
            cols:
                The columns that together identify one snapshot.
        """
        schema = cls.to_schema()
        return [
            name
            for name, col in schema.columns.items()
            if ((col.properties or {}).get('metadata') or {}).get('snapshot', False)
        ]

    @classmethod
    def get_col_metadata(cls: type[Self], col: str) -> dict[str, Any]:
        """Return the `include`, `fixed` and `snapshot` metadata of a column.

        Args:
            col:
                The column to read the metadata of.

        Returns:
            metadata:
                The metadata the schema carries for the column, empty where it carries none.
        """
        schema_col = dict(cls.to_schema().columns)[col]
        return (schema_col.properties or {}).get('metadata') or {}

    @pa.dataframe_check
    @classmethod
    def check_snapshot_unique(cls: type[Self], df: pd.DataFrame) -> bool:
        """Check that no two rows share the same snapshot identity.

        Args:
            df:
                The snapshots to check.

        Returns:
            unique:
                Whether every snapshot identity appears once.
        """
        return not df.duplicated(subset=cls.list_snapshot_cols()).any()

    class Config:
        """Reject a frame carrying a column the schema does not declare."""

        strict = True

class BaseStatsSchema(BaseSchema):
    """Base schema for statistics snapshots.

    Examples:
        >>> from sportsbet.sources import BaseStatsSchema, optional_col, required_col
        >>>
        >>> class MyStatsSchema(BaseStatsSchema):
        ...     'The statistics of a feed of your own.'
        ...
        ...     home_team: str = required_col()
        ...     away_team: str = required_col()
        ...     home_goals: float = optional_col(include=['inplay', 'postplay'], fixed=False)
    """

class BaseOddsSchema(BaseSchema):
    """Base schema for odds snapshots.

    Examples:
        >>> from sportsbet.sources import BaseOddsSchema, optional_col, required_col
        >>>
        >>> class MyOddsSchema(BaseOddsSchema):
        ...     'The odds of a feed of your own.'
        ...
        ...     home_team: str = required_col()
        ...     away_team: str = required_col()
        ...     provider: str = required_col()
        ...     home_win: float = optional_col(include=['preplay', 'inplay'], fixed=False)
        ...     away_win: float = optional_col(include=['preplay', 'inplay'], fixed=False)
    """

    @classmethod
    def list_odds_cols(cls) -> list[str]:
        """Return the market columns.

        Returns:
            cols:
                The columns carrying a price, which is every column that is neither part of the snapshot
                identity nor the provider.
        """
        schema_cols = list(cls.to_schema().columns.keys())
        return [col for col in schema_cols if col not in cls.list_snapshot_cols() and col != 'provider']

    @pa.dataframe_check
    @classmethod
    def check_postplay_missing_odds(cls, df: pd.DataFrame) -> pd.Series:
        """Check that post-match snapshots carry no odds.

        Args:
            df:
                The snapshots to check.

        Returns:
            passing:
                True for every row that is not post-match, and for every post-match row with no price.
        """
        odds_cols = cls.list_odds_cols()
        if not odds_cols:
            return pd.Series(True, index=df.index)
        is_post = df['event_status'].eq('postplay')
        ok_post = df.loc[is_post, odds_cols].isna().all(axis=1)
        out = pd.Series(True, index=df.index)
        out.loc[is_post] = ok_post
        return out

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
        return self.read_catalogue(fetch_payloads(self.list_index_items(), self.request_url))

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
