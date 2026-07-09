"""Download and transform historical and fixtures soccer data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import io
from typing import ClassVar, Self

import aiohttp
import pandas as pd
from sklearn.model_selection import ParameterGrid

from ... import FixturesData, ParamGrid, TrainData
from .._base._dataloader import BaseDataLoader
from ._schema import build_odds_schema, build_stats_schema
from ._utils import (
    EVENT_COLS,
    IDENTITY_COLS,
    derive_metadata,
    odds_columns,
    parse_odds_column,
)

_BASE_URL = 'https://raw.githubusercontent.com/georgedouzas/sports-betting/data/data/soccer/modelling'
STATS_URL = _BASE_URL + '/stats/{league}_{division}_{year}.csv'
ODDS_URL = _BASE_URL + '/odds/{league}_{division}_{year}.csv'
STATS_FIXTURES_URL = _BASE_URL + '/stats/fixtures.csv'
ODDS_FIXTURES_URL = _BASE_URL + '/odds/fixtures.csv'
CONNECTIONS_LIMIT = 20


async def _read_url_content_async(client: aiohttp.ClientSession, url: str) -> str:
    """Read asynchronously the URL content."""

    async with client.get(url) as response:
        with io.StringIO(await response.text(encoding='ISO-8859-1')) as text_io:
            return text_io.getvalue()


async def _read_urls_content_async(urls: list[str]) -> list[str]:
    """Read asynchronously the URLs content."""

    async with aiohttp.ClientSession(
        raise_for_status=True,
        connector=aiohttp.TCPConnector(limit=CONNECTIONS_LIMIT),
    ) as client:
        futures = [_read_url_content_async(client, url) for url in urls]
        return await asyncio.gather(*futures)


def _read_urls_content(urls: list[str]) -> list[str]:
    """Read the URLs content."""
    return asyncio.run(_read_urls_content_async(urls))


def _read_csvs(urls: list[str]) -> list[pd.DataFrame]:
    """Read the CSVs."""
    urls_content = _read_urls_content(urls)
    csvs = []
    for content in urls_content:
        names = pd.read_csv(io.StringIO(content), nrows=0, encoding='ISO-8859-1').columns.to_list()
        csv = pd.read_csv(io.StringIO(content), names=names, skiprows=1, encoding='ISO-8859-1', on_bad_lines='skip')
        csvs.append(csv)
    return csvs


class SoccerDataLoader(BaseDataLoader):
    """Dataloader for soccer data.

    It downloads long event-snapshot `stats` and `odds` data for the selected
    leagues, years and divisions, reads and validates it, derives the available
    providers, markets and per-column metadata from the data itself, and extracts
    moment-aware training and fixtures data. Nothing about the feed is hardcoded:
    the moments come from the stored `event_status`/`event_time`, and each column's
    role is derived from where it actually carries values.

    Read more in the [user guide][user-guide].

    Args:
        param_grid:
            Selects the data to include. Keys are parameters like `'league'`,
            `'division'` or `'year'` and values are sequences of allowed values,
            mirroring scikit-learn's `ParameterGrid`. The default `None` selects
            all available parameters.

    Attributes:
        param_grid_ (ParameterGrid):
            The checked parameters grid.

        drop_na_thres_ (float):
            The checked value of `drop_na_thres`.

        odds_type_ (str | None):
            The checked value of `odds_type`.

        input_cols_ (pd.Index):
            The columns of `X` for training and fixtures data.

        output_cols_ (pd.Index | None):
            The columns of `Y` for training data.

        odds_cols_ (pd.Index):
            The columns of `O` for training and fixtures data.
    """

    # Selectable parameters for the real feed.
    ALL_PARAMS: ClassVar[ParamGrid] = {
        'league': ['England', 'Scotland', 'Germany', 'Italy', 'Spain', 'France', 'Netherlands', 'Greece'],
        'division': [1, 2],
        'year': list(range(2018, 2026)),
    }

    def __init__(self: Self, param_grid: ParamGrid | None = None) -> None:
        self.param_grid = param_grid
        self._provided_snapshots: tuple[pd.DataFrame, pd.DataFrame] | None = None
        self._downloaded: tuple[pd.DataFrame, pd.DataFrame] | None = None

    @classmethod
    def from_dataframe(
        cls: type[Self],
        data: pd.DataFrame,
        *,
        event_status: str,
        event_time: pd.Timedelta,
        param_grid: ParamGrid | None = None,
    ) -> Self:
        """Build a loader from a user's wide match table taken at a single moment.

        Every row of `data` is treated as a snapshot at the caller-declared
        `event_status`/`event_time` — no moment is assumed. `data` must carry the
        identity columns (`date`, `league`, `division`, `year`, `home_team`,
        `away_team`), any number of value columns (goals, market outcomes,
        features), and `{provider}__{market}` odds columns. For several moments,
        provide long snapshots directly or call this per moment.

        Args:
            data:
                One row per match at a single moment.
            event_status:
                The status the rows represent, e.g. `'preplay'` or `'postplay'`.
            event_time:
                The time into the match the rows represent.
            param_grid:
                Optional selection, mirroring the constructor.

        Returns:
            A loader that reads the provided data instead of downloading it.
        """
        loader = cls(param_grid)
        loader._provided_snapshots = cls._wide_to_snapshots(data, event_status, event_time)
        return loader

    @classmethod
    def from_csv(
        cls: type[Self],
        path: str,
        *,
        event_status: str,
        event_time: pd.Timedelta,
        param_grid: ParamGrid | None = None,
    ) -> Self:
        """Build a loader from a CSV of a user's wide match table at a single moment.

        See [`from_dataframe`][sportsbet.datasets.SoccerDataLoader.from_dataframe].
        """
        return cls.from_dataframe(
            pd.read_csv(path),
            event_status=event_status,
            event_time=event_time,
            param_grid=param_grid,
        )

    @classmethod
    def from_snapshots(
        cls: type[Self],
        stats: pd.DataFrame,
        odds: pd.DataFrame,
        *,
        param_grid: ParamGrid | None = None,
    ) -> Self:
        """Build a loader from canonical long `stats` and `odds` snapshots.

        Use this when the data already follows the exported long format, i.e. one
        row per match and moment with explicit `event_status`/`event_time` columns
        (`stats` carrying the values, `odds` carrying `{provider}` and the markets).
        No moment is assumed — every row states its own.

        Args:
            stats:
                Long statistics snapshots.
            odds:
                Long odds snapshots.
            param_grid:
                Optional selection, mirroring the constructor.

        Returns:
            A loader that reads the provided snapshots instead of downloading them.
        """
        loader = cls(param_grid)
        loader._provided_snapshots = (stats, odds)
        return loader

    @staticmethod
    def _wide_to_snapshots(
        data: pd.DataFrame,
        event_status: str,
        event_time: pd.Timedelta,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a wide single-moment frame into long `stats`/`odds` snapshots."""
        odds_cols = odds_columns(list(data.columns))
        stats = data.drop(columns=odds_cols).assign(event_status=event_status, event_time=event_time)
        by_provider: dict[str, dict[str, str]] = {}
        for col in odds_cols:
            provider, market = parse_odds_column(col)
            by_provider.setdefault(provider, {})[market] = col
        records = []
        for _, row in data.iterrows():
            identity = {col: row[col] for col in IDENTITY_COLS}
            for provider, markets in by_provider.items():
                record = {**identity, 'event_status': event_status, 'event_time': event_time, 'provider': provider}
                record.update({market: row[col] for market, col in markets.items()})
                records.append(record)
        return stats, pd.DataFrame(records)

    @classmethod
    def _param_grid_all(cls: type[Self]) -> ParamGrid:
        """Return the full selectable parameter grid for the source."""
        return cls.ALL_PARAMS

    @classmethod
    def get_all_params(cls: type[Self]) -> list[dict]:
        """Return all selectable parameter combinations, without downloading data."""
        return list(ParameterGrid(cls._param_grid_all()))

    def _resolved_grid(self: Self) -> ParamGrid:
        """Merge the selected `param_grid` over the full grid, defaulting omitted dimensions to all."""
        full = self._param_grid_all()
        if self.param_grid is None:
            return full
        if isinstance(self.param_grid, dict) and isinstance(full, dict):
            return {**full, **self.param_grid}
        return self.param_grid

    def _selected_params(self: Self) -> list[dict]:
        """Resolve the parameter combinations selected by `param_grid`."""
        return list(ParameterGrid(self._resolved_grid()))

    @staticmethod
    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate feed frames, skipping empty ones so their dtypes do not leak.

        An empty CSV (e.g. `fixtures.csv` when nothing is upcoming) reads back as all-object columns, which would
        otherwise coerce the identity dtypes of the real data.
        """
        non_empty = [frame for frame in frames if not frame.empty]
        return pd.concat(non_empty, ignore_index=True) if non_empty else frames[0].copy()

    def _snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the long `stats`/`odds` snapshots (downloaded once, or user-provided)."""
        if self._provided_snapshots is not None:
            return self._provided_snapshots
        if self._downloaded is None:
            stats_urls = [STATS_URL.format(**params) for params in self._selected_params()]
            odds_urls = [ODDS_URL.format(**params) for params in self._selected_params()]
            stats = self._concat(_read_csvs([*stats_urls, STATS_FIXTURES_URL]))
            odds = self._concat(_read_csvs([*odds_urls, ODDS_FIXTURES_URL]))
            self._downloaded = (stats, odds)
        return self._downloaded

    @staticmethod
    def _finalize(data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the `date` and `event_time` dtypes of a snapshot frame."""
        data = data.reset_index(drop=True)
        if data.empty:
            return data
        data['date'] = pd.to_datetime(data['date'], utc=True).astype('datetime64[ns, UTC]')
        if not pd.api.types.is_timedelta64_dtype(data['event_time']):
            data['event_time'] = pd.to_timedelta(data['event_time'], unit='m')
        data['event_time'] = data['event_time'].astype('timedelta64[ns]')
        return data

    def _validate_snapshots(self: Self, stats: pd.DataFrame, odds: pd.DataFrame) -> None:
        """Validate that the snapshots carry the required identity and event columns."""
        for name, frame in [('stats', stats), ('odds', odds)]:
            missing = [col for col in EVENT_COLS + IDENTITY_COLS if col not in frame.columns]
            if missing:
                msg = f'The {name} data is missing the required columns: {missing}.'
                raise ValueError(msg)
        if 'provider' not in odds.columns:
            msg = 'The odds data is missing the `provider` column.'
            raise ValueError(msg)

    def get_odds_types(self: Self) -> list[str]:
        """Return the available odds types (providers) derived from the data."""
        _, odds = self._snapshots()
        return sorted(odds['provider'].dropna().unique().tolist())

    def _prepare(self: Self, odds_type: str | None) -> None:
        """Read and validate the snapshots, derive their metadata and build the base inputs."""
        stats, odds = self._snapshots()
        stats = self._finalize(stats)
        odds = self._finalize(odds)
        self._validate_snapshots(stats, odds)
        providers = sorted(odds['provider'].dropna().unique().tolist())
        if odds_type is not None and odds_type not in providers:
            msg = f'Invalid odds type. It should be one of {providers}. Got {odds_type} instead.'
            raise ValueError(msg)
        stats_value_cols = [col for col in stats.columns if col not in EVENT_COLS + IDENTITY_COLS]
        odds_value_cols = [col for col in odds.columns if col not in EVENT_COLS + IDENTITY_COLS + ['provider']]
        stats_metadata = derive_metadata(stats, stats_value_cols)
        odds_metadata = derive_metadata(odds, odds_value_cols)

        odds = odds[odds['provider'] == odds_type] if odds_type is not None else odds.iloc[0:0]
        self.stats = build_stats_schema(stats_metadata).validate(stats)
        self.odds = build_odds_schema(odds_metadata).validate(odds)
        self.stats_schema = build_stats_schema(stats_metadata)
        self.odds_schema = build_odds_schema(odds_metadata)
        self.targets = odds_value_cols
        self.odds_type_ = odds_type
        self.param_grid_ = ParameterGrid(self._resolved_grid())

    def _apply_drop_na(self: Self, X: pd.DataFrame, drop_na_thres: float) -> pd.DataFrame:
        """Drop feature columns whose missingness exceeds `drop_na_thres`."""
        if not X.empty:
            keep = X.columns[X.isna().mean() <= (1.0 - drop_na_thres)]
            X = X[keep]
        self.input_cols_ = X.columns
        self.drop_na_thres_ = drop_na_thres
        return X

    def extract_train_data(
        self: Self,
        *,
        drop_na_thres: float = 0.0,
        odds_type: str | None = None,
        learning_type: str | None = None,
        target_event_status: str | None = None,
        target_event_time: pd.Timedelta | None = None,
    ) -> TrainData:
        """Extract moment-aware training data.

        Read more in the [user guide][dataloader].

        Args:
            drop_na_thres:
                Threshold in `[0.0, 1.0]` controlling how aggressively feature
                columns with missing values are dropped. `0.0` keeps all columns.
            odds_type:
                One of `get_odds_types()`. `None` returns no odds.
            learning_type:
                `'supervised'` (default) or `'unsupervised'` (`Y` is `None`).
            target_event_status:
                `'inplay'` or `'postplay'` (default `'postplay'`).
            target_event_time:
                In-play target time (e.g. `pd.Timedelta('60min')`). Defaults to 0.

        Returns:
            (X, Y, O):
                Moment-aware features, target outcomes, and odds sharing one index.
        """
        self._prepare(odds_type)
        X, Y, O = super().extract_train_data(
            learning_type=learning_type,
            target_event_status=target_event_status,
            target_event_time=target_event_time,
        )
        X = self._apply_drop_na(X, drop_na_thres)
        return X, Y, O

    def extract_fixtures_data(self: Self) -> FixturesData:
        """Extract fixtures data with the training column layout (`Y` is `None`).

        Returns:
            (X, None, O):
                Fixtures features and odds matching the training columns.
        """
        X, Y, O = super().extract_fixtures_data()
        X = X.reindex(columns=self.input_cols_)
        return X, Y, O
