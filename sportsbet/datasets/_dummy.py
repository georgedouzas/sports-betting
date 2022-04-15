"""
Dataloader for dummy data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from ._base import _BaseDataLoader


class DummySoccerDataLoader(_BaseDataLoader):
    """Dataloader for soccer dummy data.

    The data are provided only for convenience, since they require
    no downloading, and to familiarize the user with the methods
    of the dataloader objects.

    Read more in the :ref:`user guide <user_guide>`.

    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such parameter, default=None
        It selects the type of information that the data include. The keys of
        dictionaries might be parameters like ``'league'`` or ``'division'`` while
        the values are sequences of allowed values. It works in a similar way as the
        ``param_grid`` parameter of the :class:`~sklearn.model_selection.ParameterGrid`
        class. The default value ``None`` corresponds to all parameters.

    Examples
    --------
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
    >>> # Select the odds of Interwetten bookmaker
    >>> X_train, Y_train, O_train = dataloader.extract_train_data(
    ... odds_type='interwetten')
    >>> # Training input data
    >>> print(X_train) # doctest: +NORMALIZE_WHITESPACE
                division league  year ... odds__williamhill__draw__full_time_goals
    date
    1997-05-04         1  Spain  1997 ...                                      2.5
    1999-03-04         2  Spain  1999 ...                                      NaN
    >>> # Training output data
    >>> print(Y_train)
       output__home_win__full_time_goals ... output__away_win__full_time_goals
    0                               True ...                             False
    1                              False ...                             False
    >>> # Training odds data
    >>> print(O_train)
       odds__interwetten__home_win__full_time_goals ...
    0                                           1.5 ...
    1                                           2.5 ...
    >>> # Extract the corresponding fixtures data
    >>> X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
    >>> # Training and fixtures input and odds data have the same column names
    >>> pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
    >>> pd.testing.assert_index_equal(O_train.columns, O_fix.columns)
    >>> # Fixtures data have always no output
    >>> Y_fix is None
    True
    """

    DATE = pd.Timestamp(datetime.now()) + pd.to_timedelta(1, 'd')
    PARAMS = ParameterGrid(
        [
            {'league': ['Greece'], 'division': [1], 'year': [2017, 2019]},
            {'league': ['Spain'], 'division': [1], 'year': [1997]},
            {'league': ['Spain'], 'division': [2], 'year': [1999]},
            {'league': ['England'], 'division': [2], 'year': [1997]},
            {'league': ['England'], 'division': [3], 'year': [1998]},
            {'league': ['France'], 'division': [1], 'year': [2000]},
            {'league': ['France'], 'division': [1], 'year': [2001]},
            {'division': [1], 'year': [1998]},
        ]
    )
    SCHEMA = [
        ('division', int),
        ('league', object),
        ('date', np.datetime64),
        ('year', int),
        ('home_team', object),
        ('away_team', object),
        ('home_soccer_index', float),
        ('away_soccer_index', float),
        ('target__home_team__full_time_goals', int),
        ('target__away_team__full_time_goals', int),
        ('odds__interwetten__home_win__full_time_goals', float),
        ('odds__interwetten__draw__full_time_goals', float),
        ('odds__interwetten__away_win__full_time_goals', float),
        ('odds__williamhill__home_win__full_time_goals', float),
        ('odds__williamhill__draw__full_time_goals', float),
        ('odds__williamhill__away_win__full_time_goals', float),
        ('odds__pinnacle__over_2.5__full_time_goals', float),
        ('odds__pinnacle__under_2.5__full_time_goals', float),
    ]
    OUTPUTS = [
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
            'output__over_2.5__full_time_goals',
            lambda outputs: outputs['target__home_team__full_time_goals']
            + outputs['target__away_team__full_time_goals']
            > 2.5,
        ),
        (
            'output__under_2.5__full_time_goals',
            lambda outputs: outputs['target__home_team__full_time_goals']
            + outputs['target__away_team__full_time_goals']
            < 2.5,
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
            'date': [
                pd.Timestamp('17/3/2017'),
                np.nan,
                pd.Timestamp('17/3/2019'),
                pd.Timestamp('5/4/1997'),
                pd.Timestamp('3/4/1999'),
                pd.Timestamp('5/7/1997'),
                pd.Timestamp('3/4/1998'),
                pd.Timestamp('3/4/1998'),
                DATE,
                DATE,
                pd.Timestamp('3/4/2000'),
                pd.Timestamp('6/4/2001'),
            ],
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
        }
    )

    def _get_data(self):
        return self.DATA
