"""
Dataloader for dummy data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from .._base import _BaseDataLoader


class DummySoccerDataLoader(_BaseDataLoader):
    """Dataloader for dummy data.

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
    >>> pd.DataFrame(DummySoccerDataLoader.get_all_params())
       division   league  year
    0         1   Greece  2017
    1         1   Greece  2019
    2         1    Spain  1997
    3         2    Spain  1999
    4         2  England  1997
    5         3  England  1998
    6         1   France  2000
    7         1   France  2001
    >>> # Select only the traning data for the Spanish league
    >>> dataloader = DummySoccerDataLoader(param_grid={'league': ['Spain']})
    >>> # Get available odds types
    >>> dataloader.get_odds_types()
    ['interwetten', 'williamhill']
    >>> # Select the odds of Interwetten bookmaker
    >>> X_train, Y_train, O_train = dataloader.extract_train_data(
    ... odds_type='interwetten')
    >>> # Training input data
    >>> X_train
                division league  year    home_team ... williamhill__away_win__odds
    date ...
    1997-05-04         1  Spain  1997  Real Madrid ...                         NaN
    1999-03-04         2  Spain  1999    Barcelona ...                         NaN
    >>> # Training output data
    >>> Y_train
       away_win__full_time_goals  draw__full_time_goals  home_win__full_time_goals
    0                      False                  False                       True
    1                      False                   True                      False
    >>> # Training odds data
    >>> O_train
       interwetten__away_win__odds  interwetten__draw__odds  interwetten__home_win__odds
    0                          2.5                      3.5                          1.5
    1                          2.0                      4.5                          2.5
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
    PARAMS = [
        {'league': ['Greece'], 'division': [1], 'year': [2017, 2019]},
        {'league': ['Spain'], 'division': [1], 'year': [1997]},
        {'league': ['Spain'], 'division': [2], 'year': [1999]},
        {'league': ['England'], 'division': [2], 'year': [1997]},
        {'league': ['England'], 'division': [3], 'year': [1998]},
        {'league': ['France'], 'division': [1], 'year': [2000]},
        {'league': ['France'], 'division': [1], 'year': [2001]},
    ]
    SCHEMA = [
        ('division', int),
        ('league', object),
        ('date', np.datetime64),
        ('home_team', object),
        ('away_team', object),
        ('home_team__full_time_goals', int),
        ('away_team__full_time_goals', int),
        ('interwetten__home_win__odds', float),
        ('interwetten__draw__odds', float),
        ('interwetten__away_win__odds', float),
        ('williamhill__home_win__odds', float),
        ('williamhill__draw__odds', float),
        ('williamhill__away_win__odds', float),
        ('year', int),
    ]
    OUTCOMES = [
        (
            'home_win__full_time_goals',
            lambda outputs: outputs['home_team__full_time_goals']
            > outputs['away_team__full_time_goals'],
        ),
        (
            'away_win__full_time_goals',
            lambda outputs: outputs['home_team__full_time_goals']
            < outputs['away_team__full_time_goals'],
        ),
        (
            'draw__full_time_goals',
            lambda outputs: outputs['home_team__full_time_goals']
            == outputs['away_team__full_time_goals'],
        ),
        (
            'over_2.5_goals__full_time_goals',
            lambda outputs: outputs['home_team__full_time_goals']
            + outputs['away_team__full_time_goals']
            > 2.5,
        ),
        (
            'under_2.5_goals__full_time_goals',
            lambda outputs: outputs['home_team__full_time_goals']
            + outputs['away_team__full_time_goals']
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
            'home_team__full_time_goals': [
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
            'away_team__full_time_goals': [
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
            'interwetten__home_win__odds': [
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
            'interwetten__draw__odds': [
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
            'interwetten__away_win__odds': [
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
            'williamhill__home_win__odds': [
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
            'williamhill__draw__odds': [
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
            'williamhill__away_win__odds': [
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

    @classmethod
    def _get_schema(cls):
        return cls.SCHEMA

    @classmethod
    def _get_outcomes(cls):
        return cls.OUTCOMES

    @classmethod
    def _get_params(cls):
        return ParameterGrid(cls.PARAMS)

    def _get_data(self):
        return self.DATA

    def set_data(self, data):
        self.DATA = data
        return self
