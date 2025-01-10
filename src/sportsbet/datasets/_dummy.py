"""Dataloader for dummy data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from datetime import datetime, timedelta
from typing import ClassVar

import numpy as np
import pandas as pd
import pytz
from sklearn.model_selection import ParameterGrid
from typing_extensions import Self

from .. import FixturesData, Outputs, ParamGrid, Schema, TrainData
from ._base import BaseDataLoader

OVER_UNDER = 2.5


class DummySoccerDataLoader(BaseDataLoader):
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

    DATE = datetime.now(tz=pytz.utc) + timedelta(2)
    SCHEMA: ClassVar[Schema] = [
        ('division', np.int64),
        ('league', object),
        ('date', np.datetime64),
        ('year', np.int64),
        ('home_team', object),
        ('away_team', object),
        ('home_soccer_index', float),
        ('away_soccer_index', float),
        ('target__home_team__full_time_goals', np.int64),
        ('target__away_team__full_time_goals', np.int64),
        ('odds__interwetten__home_win__full_time_goals', float),
        ('odds__interwetten__draw__full_time_goals', float),
        ('odds__interwetten__away_win__full_time_goals', float),
        ('odds__williamhill__home_win__full_time_goals', float),
        ('odds__williamhill__draw__full_time_goals', float),
        ('odds__williamhill__away_win__full_time_goals', float),
        (f'odds__pinnacle__over_{OVER_UNDER}__full_time_goals', float),
        (f'odds__pinnacle__under_{OVER_UNDER}__full_time_goals', float),
    ]

    OUTPUTS: ClassVar[Outputs] = [
        (
            'output__home_win__full_time_goals',
            lambda outputs: outputs['target__home_team__full_time_goals']
            > outputs['target__away_team__full_time_goals'],
        ),
        (
            'output__away_win__full_time_goals',
            lambda outputs: outputs['target__home_team__full_time_goals']
            < outputs['target__away_team__full_time_goals'],
        ),
        (
            'output__draw__full_time_goals',
            lambda outputs: outputs['target__home_team__full_time_goals']
            == outputs['target__away_team__full_time_goals'],
        ),
        (
            f'output__over_{OVER_UNDER}__full_time_goals',
            lambda outputs: outputs['target__home_team__full_time_goals']
            + outputs['target__away_team__full_time_goals']
            > OVER_UNDER,
        ),
        (
            f'output__under_{OVER_UNDER}__full_time_goals',
            lambda outputs: outputs['target__home_team__full_time_goals']
            + outputs['target__away_team__full_time_goals']
            < OVER_UNDER,
        ),
    ]

    DATA = pd.DataFrame(
        {
            'division': [1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 3.0, 1.0, 1.0],
            'league': [
                'Greece',
                'Greece',
                'Greece',
                'Spain',
                'Spain',
                'England',
                'England',
                np.nan,
                np.nan,
                'France',
                'France',
                'France',
            ],
            'date': pd.to_datetime(
                [
                    '17/3/2017',
                    'NaT',
                    '17/3/2019',
                    '5/4/1997',
                    '3/4/1999',
                    '5/7/1997',
                    '3/4/1998',
                    '3/4/1998',
                    DATE.date().strftime('%d/%m/%Y'),
                    DATE.date().strftime('%d/%m/%Y'),
                    '3/4/2000',
                    '6/4/2001',
                ],
                format='%d/%m/%Y',
            ),
            'year': [
                2017,
                np.nan,
                2019,
                1997,
                1999,
                1997,
                1998,
                1998,
                DATE.year,
                DATE.year,
                2000,
                2001,
            ],
            'home_team': [
                'Olympiakos',
                np.nan,
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
                np.nan,
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
            'target__home_team__full_time_goals': [
                1,
                np.nan,
                1,
                2,
                2,
                np.nan,
                1,
                1,
                np.nan,
                np.nan,
                2,
                1,
            ],
            'target__away_team__full_time_goals': [
                1,
                np.nan,
                0,
                1,
                2,
                2,
                3,
                2,
                np.nan,
                np.nan,
                1,
                2,
            ],
            'odds__interwetten__home_win__full_time_goals': [
                2.0,
                1.5,
                2,
                1.5,
                2.5,
                3,
                2,
                np.nan,
                3,
                1.5,
                2.0,
                3.0,
            ],
            'odds__interwetten__draw__full_time_goals': [
                2,
                2,
                2,
                3.5,
                4.5,
                2.5,
                4.5,
                2.5,
                2.5,
                3.5,
                2.5,
                2.5,
            ],
            'odds__interwetten__away_win__full_time_goals': [
                2,
                2,
                3,
                2.5,
                2,
                2,
                3.5,
                3.5,
                2,
                2.5,
                3.0,
                2.0,
            ],
            'odds__williamhill__home_win__full_time_goals': [
                2,
                1.5,
                3.5,
                2.5,
                2.0,
                3.0,
                2.0,
                4.0,
                3.5,
                2.5,
                2.5,
                2.5,
            ],
            'odds__williamhill__draw__full_time_goals': [
                2,
                np.nan,
                1.5,
                2.5,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                2.5,
                1.5,
                2.5,
                3.0,
            ],
            'odds__williamhill__away_win__full_time_goals': [
                2,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                2.0,
                2.5,
                3.0,
                2.5,
            ],
            'odds__pinnacle__over_2.5__full_time_goals': [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            'fixtures': [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
            ],
        },
    )

    def __init__(self: Self, param_grid: ParamGrid | None = None) -> None:
        super().__init__(param_grid)

    @classmethod
    def _get_full_param_grid(cls: type[DummySoccerDataLoader]) -> ParameterGrid:
        return ParameterGrid(
            [
                {'league': ['Greece'], 'division': [1], 'year': [2017, 2019]},
                {'league': ['Spain'], 'division': [1], 'year': [1997]},
                {'league': ['Spain'], 'division': [2], 'year': [1999]},
                {'league': ['England'], 'division': [2], 'year': [1997]},
                {'league': ['England'], 'division': [3], 'year': [1998]},
                {'league': ['France'], 'division': [1], 'year': [2000]},
                {'league': ['France'], 'division': [1], 'year': [2001]},
                {'division': [1], 'year': [1998]},
            ],
        )

    def _get_data(self: Self) -> pd.DataFrame:
        return self.DATA

    def extract_train_data(
        self: Self,
        drop_na_thres: float = 0.0,
        odds_type: str | None = None,
    ) -> TrainData:
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
        return super().extract_train_data(drop_na_thres, odds_type)

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
