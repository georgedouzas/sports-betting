"""Download and transform historical and fixtures data."""

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

from ... import FixturesData, OutputsMapping, TrainData
from .._base._dataloader import BaseDataLoader

MODELLING_URL = 'https://github.com/georgedouzas/sports-betting/tree/data/data/soccer/modelling'
TRAINING_URL = 'https://raw.githubusercontent.com/georgedouzas/sports-betting/data/data/soccer/modelling/{league}_{division}_{year}.csv'
FIXTURES_URL = 'https://raw.githubusercontent.com/georgedouzas/sports-betting/data/data/soccer/modelling/fixtures.csv'


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


def _read_csv(url: str) -> pd.DataFrame:
    """Read the CSV."""
    return _read_csvs([url])[0]


class SoccerDataLoader(BaseDataLoader):
    """Dataloader for soccer data.

    It downloads historical and fixtures data for various
    leagues, years and divisions.

    Read more in the [user guide][user-guide].

    Args:
        param_grid:
            It selects the type of information that the data include. The keys of
            dictionaries might be parameters like `'league'` or `'division'` while
            the values are sequences of allowed values. It works in a similar way as the
            `param_grid` parameter of the scikit-learn's ParameterGrid class.
            The default value `None` corresponds to all parameters.

    Attributes:
        param_grid_ (ParameterGrid):
            The checked value of parameters grid. It includes all possible parameters if
            `param_grid` is `None`.

        dropped_na_cols_ (pd.Index):
            The columns with missing values that are dropped.

        drop_na_thres_(float):
            The checked value of `drop_na_thres`.

        odds_type_ (str | None):
            The checked value of `odds_type`.

        input_cols_ (pd.Index):
            The columns of `X_train` and `X_fix`.

        output_cols_ (pd.Index):
            The columns of `Y_train` and `Y_fix`.

        odds_cols_ (pd.Index):
            The columns of `O_train` and `O_fix`.

        target_cols_ (pd.Index):
            The columns used for the extraction of output and odds columns.

        train_data_ (TrainData):
            The tupde (X, Y, O) that represents the training data as extracted from
            the method `extract_train_data`.

        fixtures_data_ (FixturesData):
            The tupde (X, Y, O) that represents the fixtures data as extracted from
            the method `extract_fixtures_data`.

    Exampdes:
        >>> from sportsbet.datasets import SoccerDataLoader
        >>> import pandas as pd
        >>> # Get all available parameters to select the training data
        >>> SoccerDataLoader.get_all_params()
        [{'division': 1, 'league': 'Argentina', ...
        >>> # Select only the traning data for the French and Spanish leagues of 2020 year
        >>> dataloader = SoccerDataLoader(
        ... param_grid={'league': ['England', 'Spain'], 'year':[2020]})
        >>> # Get available odds types
        >>> dataloader.get_odds_types()
        ['market_average', 'market_maximum']
        >>> # Select the market average odds and drop colums with missing values
        >>> X_train, Y_train, O_train = dataloader.extract_train_data(
        ... odds_type='market_average')
        >>> # Odds data include the selected market average odds
        >>> O_train.columns
        Index(['odds__market_average__home_win__full_time_goals',...
        >>> # Extract the corresponding fixtures data
        >>> X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
        >>> # Training and fixtures input and odds data have the same column names
        >>> pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
        >>> pd.testing.assert_index_equal(O_train.columns, O_fix.columns)
        >>> # Fixtures data have always no output
        >>> Y_fix is None
        True
    """

    OVER_UNDER_SCORES: ClassVar[dict[str, float]] = {
        '2.5': 2.5,
        '3.5': 3.5,
        '4.5': 4.5,
        '5.5': 5.5,
        '6.5': 6.5,
        '7.5': 7.5,
        '8.5': 8.5,
        '9.5': 9.5,
    }

    @property
    def _outputs_mapping(self: Self) -> OutputsMapping:
        return {
            ('home_team_goals', 'away_team_goals'): {
                'home_win': lambda data: data['home_team_goals'] > data['away_team_goals'],
                'away_win': lambda data: data['away_team_goals'] > data['home_team_goals'],
                'draw': lambda data: data['home_team_goals'] == data['away_team_goals'],
                'over_2.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['2.5'],
                'under_2.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['2.5'],
                'over_3.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['3.5'],
                'under_3.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['3.5'],
                'over_4.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['4.5'],
                'under_4.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['4.5'],
                'over_5.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['5.5'],
                'under_5.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['5.5'],
                'over_6.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['6.5'],
                'under_6.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['6.5'],
                'over_7.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['7.5'],
                'under_7.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['7.5'],
                'over_8.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['8.5'],
                'under_8.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['8.5'],
                'over_9.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['9.5'],
                'under_9.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['9.5'],
            },
        }

    @property
    def _required_cols(self: Self) -> list[str]:
        return ['league', 'division', 'year', 'home_team', 'away_team']

    @property
    def _stages(self: Self) -> list[str]:
        return [
            '0 min',
            *[f'{minute!s} min' for minute in range(1, 46)],
            'half_time',
            *[f'{minute!s} min' for minute in range(45, 90)],
            'full_time',
        ]

    def _get_stats_data(self: Self) -> pd.DataFrame:
        return pd.DataFrame()

    def _get_odds_data(self: Self) -> pd.DataFrame:
        return pd.DataFrame()

    @lru_cache  # noqa: B019
    def _get_data(self: Self) -> pd.DataFrame:
        urls = [TRAINING_URL.format(**params) for params in self.param_grid_]
        training_data = pd.concat(_read_csvs(urls))
        training_data['fixtures'] = False
        fixtures_data = _read_csv(FIXTURES_URL)
        fixtures_data['fixtures'] = True
        data = (pd.concat([training_data, fixtures_data]) if not fixtures_data.empty else training_data).reset_index(
            drop=True,
        )
        try:
            data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
        except ValueError:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)
        return data

    def extract_train_data(self: Self) -> TrainData:
        """Extract the training data.

        Read more in the [user guide][dataloader].

        It returns historical data that can be used to create a betting
        strategy based on heuristics or machine learning models.

        The data contain information about the matches that belong
        in two categories. The first category includes any information
        known before the start of the match, i.e. the training data `X`
        and the odds data `O`. The second category includes the outcomes of
        matches i.e. the multi-output targets `Y`.

        The method selects only the the data allowed by the `param_grid`
        parameter of the initialization method. Additionally, columns with missing
        values are dropped through the `drop_na_thres` parameter, while the
        types of odds returned is defined by the `odds_type` parameter.

        Args:
            drop_na_thres:
                The threshold that specifies the input columns to drop. It is a float in
                the `[0.0, 1.0]` range. Higher values result in dropping more values.
                The default value `drop_na_thres=0.0` keeps all columns while the
                maximum value `drop_na_thres=1.0` keeps only columns with non
                missing values.

            odds_type:
                The selected odds type. It should be one of the available odds columns
                prefixes returned by the method `get_odds_types`. If `odds_type=None`
                then no odds are returned.

        Returns:
            (X, Y, O):
                Each of the components represent the training input data `X`, the
                multi-output targets `Y` and the corresponding odds `O`, respectively.
        """
        return super().extract_train_data()

    def extract_fixtures_data(self: Self) -> FixturesData:
        """Extract the fixtures data.

        Read more in the [user guide][dataloader].

        It returns fixtures data that can be used to make predictions for
        upcoming matches based on a betting strategy.

        Before calling the `extract_fixtures_data` method for
        the first time, the `extract_training_data` should be called, in
        order to match the columns of the input, output and odds data.

        The data contain information about the matches known before the
        start of the match, i.e. the training data `X` and the odds
        data `O`. The multi-output targets `Y` is always equal to `None`
        and are only included for consistency with the method `extract_train_data`.

        The `param_grid` parameter of the initialization method has no effect
        on the fixtures data.

        Returns:
            (X, None, O):
                Each of the components represent the fixtures input data `X`, the
                multi-output targets `Y` equal to `None` and the
                corresponding odds `O`, respectively.
        """
        return super().extract_fixtures_data()
