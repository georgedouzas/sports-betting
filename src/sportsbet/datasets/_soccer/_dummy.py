"""Dataloader for dummy data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import ClassVar

import numpy as np
import polars as pl
import pytz
from typing_extensions import Self

from ... import FixturesData, TrainData
from ._base import BaseSoccerDataLoader


class DummySoccerDataLoader(BaseSoccerDataLoader):
    """Dataloader for soccer dummy data.

    The data are provided only for convenience, since they require
    no downloading, and to familiarize the user with the methods
    of the dataloader objects.

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
            The tuple (X, Y, O) that represents the training data as extracted from
            the method `extract_train_data`.

        fixtures_data_ (FixturesData):
            The tuple (X, Y, O) that represents the fixtures data as extracted from
            the method `extract_fixtures_data`.

    Examples:
        >>> from sportsbet.datasets import DummySoccerDataLoader
        >>> import pandas as pd
        >>> # Get all available parameters to select the training data
        >>> DummySoccerDataLoader.get_all_params()
        [{'division': 1, 'year': 1998}, ...
        >>> # Select only the traning data for the Spanish league
        >>> dataloader = DummySoccerDataLoader(param_grid={'league': ['Spain']})
        >>> # Get available odds types
        >>> dataloader.get_odds_types()
        ['interwetten', 'williamhill']
        >>> # Select the odds of Interwetten bookmaker for training data
        >>> X_train, Y_train, O_train = dataloader.extract_train_data(
        ... odds_type='interwetten')
        >>> # Extract the corresponding fixtures data
        >>> X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
        >>> # Training and fixtures input and odds data have the same column names
        >>> pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
        >>> pd.testing.assert_index_equal(O_train.columns, O_fix.columns)
        >>> # Fixtures data have always no output
        >>> Y_fix is None
        True
    """

    DATE_CURRENT = datetime.now(tz=pytz.utc) + timedelta(2)
    DATES: ClassVar[list[date | None]] = [
        date(2017, 3, 17),
        None,
        date(2019, 3, 17),
        date(1997, 4, 5),
        date(1999, 4, 3),
        date(1997, 7, 5),
        date(1998, 4, 3),
        date(1998, 4, 3),
        DATE_CURRENT.date(),
        DATE_CURRENT.date(),
        date(2000, 4, 3),
        date(2001, 4, 6),
    ]
    MATCHES_DATA: pl.DataFrame = pl.DataFrame(
        {
            'date': DATES,
            'year': [date.year if date is not None else None for date in DATES],
            'league': [
                'Greece',
                'Greece',
                'Greece',
                'Spain',
                'Spain',
                'England',
                'England',
                None,
                None,
                'France',
                'France',
                'France',
            ],
            'home_team': [
                'Olympiakos',
                None,
                'Panathinaikos',
                'Real Madrid',
                'Barcelona',
                'Arsenal',
                'Liverpool',
                'Liverpool',
                'Barcelona',
                'Monaco',
                'Lens',
                'PSG',
            ],
            'away_team': [
                'Panathinaikos',
                None,
                'AEK',
                'Barcelona',
                'Real Madrid',
                'Liverpool',
                'Arsenal',
                'Arsenal',
                'Real Madrid',
                'PSG',
                'Monaco',
                'Lens',
            ],
            'division': [1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1],
        },
    )
    FULL_TIME_DATA: pl.DataFrame = pl.concat(
        [
            MATCHES_DATA,
            pl.DataFrame(
                {
                    'stage': ['full_time'] * len(DATES),
                    'home_team_goals': [1, None, 1, 2, 2, None, 1, 1, None, None, 2, 1],
                    'away_team_goals': [1, None, 0, 1, 2, 2, 3, 2, None, None, 1, 2],
                },
            ),
        ],
        how='horizontal',
    )
    HALF_TIME_DATA: pl.DataFrame = pl.concat(
        [
            MATCHES_DATA,
            pl.DataFrame(
                {
                    'stage': ['half_time'] * len(DATES),
                    'home_team_goals': [0, None, 0, 1, 2, None, 0, 1, None, None, 0, 0],
                    'away_team_goals': [0, None, 0, 1, 1, 2, 1, 2, None, None, 1, 2],
                },
            ),
        ],
        how='horizontal',
    )
    PRE_HALF_TIME_DATA: pl.DataFrame = pl.concat(
        [
            MATCHES_DATA,
            pl.DataFrame(
                {
                    'stage': ['15 min'] * len(DATES),
                    'home_team_goals': [0, None, 0, 0, 0, None, 0, 0, None, None, 0, 0],
                    'away_team_goals': [0, None, 0, 0, 0, 0, 0, 0, None, None, 0, 0],
                },
            ),
        ],
        how='horizontal',
    )
    STATS_DATA = pl.concat([FULL_TIME_DATA, HALF_TIME_DATA, PRE_HALF_TIME_DATA])
    PROVIDERS_DATA = pl.DataFrame(
        {
            'provider': ['interwetten', 'williamhill', 'pinnacle'],
        },
    )
    MARKETS_DATA = pl.DataFrame(
        {
            'market': ['home_win', 'draw', 'over_2.5', 'over_3.5'],
        },
    )
    STAGES_DATA = pl.DataFrame(
        {
            'stage': ['full_time', '15 min', '55 min'],
        },
    )
    VALUES_DATA = pl.DataFrame(
        {
            'value': np.random.choice(
                [1.5, 2.0, 2.5, 3.0, None],
                len(MATCHES_DATA) * len(PROVIDERS_DATA) * len(MARKETS_DATA) * len(STAGES_DATA),
            ).tolist(),
        },
    )
    ODDS_DATA = pl.concat(
        [
            MATCHES_DATA.join(PROVIDERS_DATA, how='cross')
            .join(MARKETS_DATA, how='cross')
            .join(STAGES_DATA, how='cross'),
            VALUES_DATA,
        ],
        how='horizontal',
    ).with_columns(
        datetime=pl.col('date').cast(pl.Datetime)
        - pl.Series(
            [
                timedelta(days=np.random.randint(1, 4), hours=np.random.randint(6, 12))
                for _ in range(len(MATCHES_DATA) * len(PROVIDERS_DATA) * len(MARKETS_DATA) * len(STAGES_DATA))
            ],
        ),
    )

    def __init__(self: Self, prediction_stage: str | None = None) -> None:
        self.prediction_stage = prediction_stage

    def _get_stats_data(self: Self) -> pl.DataFrame:
        return self.STATS_DATA

    def _get_odds_data(self: Self) -> pl.DataFrame:
        return self.ODDS_DATA

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
