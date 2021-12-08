"""
Dataloder for dummy data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from sportsbet.datasets._base import _BaseDataLoader


class DummyDataLoader(_BaseDataLoader):
    """Dataloader for dummy data."""

    PARAMS = [
        {'league': ['Greece'], 'division': [1], 'year': [2017, 2019]},
        {'league': ['Spain'], 'division': [1], 'year': [1997]},
        {'league': ['Spain'], 'division': [2], 'year': [1999]},
        {'league': ['England'], 'division': [2], 'year': [1997]},
        {'league': ['England'], 'division': [3], 'year': [1998]},
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
        ('william_hill__home_win__odds', float),
        ('william_hill__draw__odds', float),
        ('william_hill__away_win__odds', float),
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
            'division': [1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 3.0],
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
                pd.Timestamp('5/4/2021'),
                pd.Timestamp('10/4/2021'),
            ],
            'year': [2017, np.nan, 2019, 1997, 1999, 1997, 1998, 1998, 2021, 2021],
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
            ],
            'away_team__full_time_goals': [1, np.nan, 0, 1, 2, 2, 1, 2, np.nan, np.nan],
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
            ],
            'interwetten__draw__odds': [2, 2, 2, 3.5, 4.5, 2.5, 4.5, 2.5, 2.5, 3.5],
            'interwetten__away_win__odds': [2, 2, 3, 2.5, 2, 2, 3.5, 3.5, 2, 2.5],
            'william_hill__home_win__odds': [
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
            ],
            'william_hill__draw__odds': [
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
            ],
            'william_hill__away_win__odds': [
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
            ],
        }
    )

    def __init__(self, param_grid=None, data=DATA):
        super(DummyDataLoader, self).__init__(
            param_grid,
        )
        self.data = data

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
        return self.data
