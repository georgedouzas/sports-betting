"""
Download and transform historical and fixtures data 
for various leagues from FiveThirtyEight. 

FiveThirtyEight: https://github.com/fivethirtyeight/data/tree/master/soccer-spi
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from sklearn.model_selection import ParameterGrid

from . import TARGETS
from .._utils import (
    _DataLoader
)

URL = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
REMOVED = [
    ('season', None, None)
]
CREATED = [
    (None, 'division', int),
    (None, 'match_quality', float)
]
RENAMED = [
    ('league', 'league', object),
    ('team1', 'home_team', object),
    ('team2', 'away_team', object),
    ('date', 'date', np.datetime64),
    ('spi1', 'home_team_soccer_power_index', float),
    ('spi2', 'away_team_soccer_power_index', float),
    ('prob1', 'home_team_probability_win', float),
    ('prob2', 'away_team_probability_win', float),
    ('probtie', 'probability_draw', float),
    ('proj_score1', 'home_team_projected_score', float),
    ('proj_score2', 'away_team_projected_score', float),
    ('importance1', 'home_team_match_importance', float),
    ('importance2', 'away_team_match_importance', float),
]
INPUT = REMOVED + CREATED + RENAMED
OUTPUT = [
    ('score1', 'home_team__full_time_goals', int), 
    ('score2', 'away_team__full_time_goals', int),
    ('xg1', 'home_team__full_time_shot_expected_goals', float),
    ('xg2', 'away_team__full_time_shot_expected_goals', float),
    ('nsxg1', 'home_team__full_time_non_shot_expected_goals', float),
    ('nsxg2', 'away_team__full_time_non_shot_expected_goals', float),
    ('adj_score1', 'home_team__full_time_adjusted_goals', float),
    ('adj_score2', 'away_team__full_time_adjusted_goals', float)
]
CONFIG = INPUT + OUTPUT
LEAGUES_MAPPING = {
    4582: ('USA-Women', 1),
    9541: ('USA-Women', 1),
    2160: ('United-Soccer-League', 1),
    1818: ('Champions-League', 1),
    1820: ('Europa-League', 1),
    1843: ('France', 1),
    2411: ('England', 1),
    1869: ('Spain', 1),
    1854: ('Italy', 1),
    1845: ('Germany', 1),
    1951: ('USA', 1),
    1874: ('Sweden', 1),
    1859: ('Norway', 1),
    2105: ('Brazil', 1),
    1866: ('Russia', 1),
    1952: ('Mexico', 1),
    1975: ('Mexico', 1),
    1827: ('Austria', 1),
    1879: ('Switzerland', 1),
    1844: ('France', 2),
    1846: ('Germany', 2),
    2412: ('England', 2),
    2417: ('Scotland', 1),
    1864: ('Portugal', 1),
    1849: ('Netherlands', 1),
    1882: ('Turkey', 1),
    1871: ('Spain', 2),
    1856: ('Italy', 2),
    5641: ('Argentina', 1),
    1837: ('Denmark', 1),
    1832: ('Belgium', 1),
    1947: ('Japan', 1),
    1979: ('China', 1),
    2413: ('England', 3),
    1983: ('South-Africa', 1),
    2414: ('England', 4),
    1884: ('Greece', 1),
    1948: ('Australia', 1)
}


class _FTEDataLoader(_DataLoader):
    """Data loader for FiveThirtyEight data."""

    def _fetch_full_param_grid(self):
        full_param_grid = pd.DataFrame([{'league': league, 'division': division} for league, division in LEAGUES_MAPPING.values()]).drop_duplicates().to_dict('records')
        self.full_param_grid_ = ParameterGrid([{'league': [param_grid['league']], 'division': [param_grid['division']] } for param_grid in full_param_grid])
        return self
    
    def _fetch_data(self):
        data = pd.read_csv(URL, parse_dates=['date'])
        data[['league', 'division']] = pd.DataFrame(
            data['league_id'].apply(lambda lid: LEAGUES_MAPPING[lid]).values.tolist()
        )
        data['match_quality'] = 2 / (1 / data['spi1'] + 1 / data['spi2'])
        data['test'] = data['score1'].isna() & data['score2'].isna()
        self.data_ = data
        return self


def load_from_five_thirty_eight_soccer_data(
    param_grid=None,
    drop_na_cols=None,
    drop_na_rows=None,
    testing_duration=None,
    return_only_params=False
):
    """Load and return FiveThirtyEight soccer data for model training and testing.

    parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such parameter, default=None
        The parameter grid to explore, as a dictionary mapping data parameters
        to sequences of allowed values. An empty dict signifies default
        parameters. A sequence of dicts signifies a sequence of grids to search,
        and is useful to avoid exploring parameter combinations that do not
        exist. The default value corresponds to all parameters.
    
    drop_na_cols : float, default=None
        The threshold of input columns to drop. It is a float in the [0.0,
        1.0] range. The default value ``None ``corresponds to ``0.0`` i.e.
        all columns are kept while the value ``1.0`` keeps only columns with
        non missing values.
    
    drop_na_rows : bool, default=None
        The threshold of rows with missing values to drop. It is a
        float in the [0.0, 1.0] range. The default value ``None``
        corresponds to ``0.0`` i.e. all rows are kept while the value
        ``1.0`` keeps only rows with non missing values.
    
    testing_duration : int, default=None
        The number of future weeks to include in the testing data. The
        default value corresponds to one week.
    
    return_only_params : bool, default=False
        When set to ``True`` only the available parameter grid is returned.

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        training : (X, Y, O) tuple
            A tuple of 'X' and 'Y', both as pandas
            DataFrames, that represent the training input data and 
            multi-output targets, respectively.
        testing : (X, None, O) tuple
            A pandas DataFrame that represents the testing input data.
        removed : :class:`sklearn.utils.Bunch`
            The dropped columns and rows as attributes.
        params : :class:`sklearn.utils.Bunch`
            The selected and available parameter grids as pandas DataFrames.
    """
    data_loader = _FTEDataLoader(config=CONFIG, targets=TARGETS, param_grid=param_grid, drop_na_cols=drop_na_cols, drop_na_rows=drop_na_rows, testing_duration=testing_duration)
    return data_loader.load(return_only_params)
