"""Download and transform historical and fixtures soccer data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import io
import warnings
from functools import lru_cache
from typing import ClassVar, Self

import aiohttp
import pandas as pd
from sklearn.model_selection import ParameterGrid

from ... import FixturesData, ParamGrid, TrainData
from .._base._dataloader import BaseDataLoader
from ._schema import SoccerOddsSchema, SoccerStatsSchema
from ._utils import MARKETS, market_outcomes

TRAINING_URL = 'https://raw.githubusercontent.com/georgedouzas/sports-betting/data/data/soccer/modelling/{league}_{division}_{year}.csv'
FIXTURES_URL = 'https://raw.githubusercontent.com/georgedouzas/sports-betting/data/data/soccer/modelling/fixtures.csv'
CONNECTIONS_LIMIT = 20
IDENTITY_COLS = ['date', 'league', 'division', 'year', 'home_team', 'away_team']
PROVIDERS: dict[str, str] = {'bet365': 'B365', 'market_average': 'Avg', 'market_maximum': 'Max'}
FEED_MARKETS: dict[str, str] = {
    'home_win': 'H',
    'draw': 'D',
    'away_win': 'A',
    'over_2.5': '>2.5',
    'under_2.5': '<2.5',
}


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

    It downloads historical and fixtures data for various leagues, years and
    divisions, maps them onto the event-snapshot model, and extracts moment-aware
    training and fixtures data. The real feed provides `preplay` and `postplay`
    snapshots only; an in-play target against it resolves no matches and raises a
    `ValueError` until an in-play source exists (the offline
    [`DummySoccerDataLoader`][sportsbet.datasets.DummySoccerDataLoader] carries
    in-play sample snapshots to exercise that path).

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

    _fixtures_flag = 'fixtures'

    # Selectable parameters for the real feed.
    ALL_PARAMS: ClassVar[ParamGrid] = {
        'league': ['England', 'Scotland', 'Germany', 'Italy', 'Spain', 'France', 'Netherlands', 'Greece'],
        'division': [1, 2],
        'year': list(range(2018, 2026)),
    }

    def __init__(self: Self, param_grid: ParamGrid | None = None) -> None:
        self.param_grid = param_grid

    @classmethod
    def _param_grid_all(cls: type[Self]) -> ParamGrid:
        """Return the full selectable parameter grid for the source."""
        return cls.ALL_PARAMS

    @classmethod
    def get_all_params(cls: type[Self]) -> list[dict]:
        """Return all selectable parameter combinations, without downloading data."""
        return list(ParameterGrid(cls._param_grid_all()))

    def get_odds_types(self: Self) -> list[str]:
        """Return the available odds types (providers)."""
        return list(PROVIDERS)

    def _selected_params(self: Self) -> list[dict]:
        """Resolve the parameter combinations selected by `param_grid`."""
        grid = self._param_grid_all() if self.param_grid is None else self.param_grid
        return list(ParameterGrid(grid))

    @lru_cache  # noqa: B019
    def _raw_data(self: Self) -> pd.DataFrame:
        """Download the wide football-data rows plus a `fixtures` flag column."""
        urls = [TRAINING_URL.format(**params) for params in self._selected_params()]
        training_data = pd.concat(_read_csvs(urls), ignore_index=True) if urls else pd.DataFrame()
        training_data[self._fixtures_flag] = False
        fixtures_data = _read_csvs([FIXTURES_URL])[0]
        fixtures_data[self._fixtures_flag] = True
        data = pd.concat([training_data, fixtures_data], ignore_index=True)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            data['date'] = pd.to_datetime(data['date'], dayfirst=True)
        return data

    def _snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Map the wide source rows onto long `stats`/`odds` snapshot frames."""
        raw = self._raw_data()
        stats_records: list[dict] = []
        odds_records: list[dict] = []
        for _, row in raw.iterrows():
            identity = {col: row[col] for col in IDENTITY_COLS}
            is_fixture = bool(row[self._fixtures_flag])
            base_stats = {**identity, 'home_latest_streak': 0, 'away_latest_streak': 0}
            preplay = {
                **base_stats,
                'event_status': 'preplay',
                'event_time': pd.Timedelta(0),
                'home_goals': 0,
                'away_goals': 0,
                **dict.fromkeys(MARKETS, 0),
            }
            stats_records.append(preplay)
            if not is_fixture:
                home_goals, away_goals = int(row['FTHG']), int(row['FTAG'])
                markets = market_outcomes(pd.Series([home_goals]), pd.Series([away_goals])).iloc[0].to_dict()
                stats_records.append(
                    {
                        **base_stats,
                        'event_status': 'postplay',
                        'event_time': pd.Timedelta(0),
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        **markets,
                    },
                )
            for provider, prefix in PROVIDERS.items():
                odds = {**identity, 'event_status': 'preplay', 'event_time': pd.Timedelta(0), 'provider': provider}
                for market, suffix in FEED_MARKETS.items():
                    col = f'{prefix}{suffix}'
                    odds[market] = float(row[col]) if col in row.index and pd.notna(row[col]) else None
                for market in MARKETS:
                    odds.setdefault(market, None)
                odds_records.append(odds)
        return self._finalize(pd.DataFrame(stats_records)), self._finalize(pd.DataFrame(odds_records))

    @staticmethod
    def _finalize(data: pd.DataFrame) -> pd.DataFrame:
        """Apply shared dtype normalization to a snapshot frame."""
        if not data.empty:
            # Pin nanosecond resolution: newer pandas infers ``us`` from strings.
            data['date'] = pd.to_datetime(data['date'], utc=True).astype('datetime64[ns, UTC]')
        return data.reset_index(drop=True)

    def _prepare(self: Self, odds_type: str | None) -> None:
        """Populate the base-loader inputs from the selected snapshots and odds type."""
        stats, odds = self._snapshots()
        if odds_type is not None and odds_type not in self.get_odds_types():
            msg = f'Invalid odds type. It should be one of {self.get_odds_types()}. Got {odds_type} instead.'
            raise ValueError(msg)
        odds = odds[odds['provider'] == odds_type] if odds_type is not None else odds.iloc[0:0]
        # Markets absent from the feed arrive as all-null object columns; make them float.
        odds = odds.astype(dict.fromkeys(MARKETS, float))
        self.stats = SoccerStatsSchema.validate(stats)
        self.odds = SoccerOddsSchema.validate(odds)
        self.stats_schema = SoccerStatsSchema
        self.odds_schema = SoccerOddsSchema
        self.targets = MARKETS
        self.odds_type_ = odds_type
        self.param_grid_ = ParameterGrid(self._param_grid_all() if self.param_grid is None else self.param_grid)

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
