"""
Download and transform historical and fixtures data
for various leagues from FiveThirtyEight.

FiveThirtyEight: https://github.com/fivethirtyeight/data/tree/master/soccer-spi
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from . import OUTCOMES
from .._base import _BaseDataLoader, _read_csv

URL = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
LEAGUES_MAPPING = {
    7921: ('FAWSL', 1),
    10281: ('Europa', 1),
    4582: ('NWSL', 1),
    9541: ('NWSL', 1),
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
    1948: ('Australia', 1),
}


def _extract_data():
    data = _read_csv(URL, parse_dates='date').copy()
    data[['league', 'division']] = pd.DataFrame(
        data['league_id'].apply(lambda lid: LEAGUES_MAPPING[lid]).values.tolist()
    )
    data['year'] = data['season'] + 1
    return data


class FTEDataLoader(_BaseDataLoader):
    """Dataloader for FiveThirtyEight data.

    Read more in the :ref:`user guide <user_guide>`.

    parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such parameter, default=None
        The parameter grid to explore, as a dictionary mapping data parameters
        to sequences of allowed values. An empty dict signifies default
        parameters. A sequence of dicts signifies a sequence of grids to search,
        and is useful to avoid exploring parameter combinations that do not
        exist. The default value corresponds to all parameters.
    """

    _removed_cols = ['season', 'league_id']
    _cols_mapping = {
        'team1': 'home_team',
        'team2': 'away_team',
        'date': 'date',
        'spi1': 'home_team_soccer_power_index',
        'spi2': 'away_team_soccer_power_index',
        'prob1': 'home_team_probability_win',
        'prob2': 'away_team_probability_win',
        'probtie': 'probability_draw',
        'proj_score1': 'home_team_projected_score',
        'proj_score2': 'away_team_projected_score',
        'importance1': 'home_team_match_importance',
        'importance2': 'away_team_match_importance',
        'score1': 'home_team__full_time_goals',
        'score2': 'away_team__full_time_goals',
        'xg1': 'home_team__full_time_shot_expected_goals',
        'xg2': 'away_team__full_time_shot_expected_goals',
        'nsxg1': 'home_team__full_time_non_shot_expected_goals',
        'nsxg2': 'away_team__full_time_non_shot_expected_goals',
        'adj_score1': 'home_team__full_time_adjusted_goals',
        'adj_score2': 'away_team__full_time_adjusted_goals',
    }

    @classmethod
    def _get_schema(cls):
        return [
            ('year', int),
            ('division', int),
            ('match_quality', float),
            ('league', object),
            ('home_team', object),
            ('away_team', object),
            ('date', np.datetime64),
            ('home_team_soccer_power_index', float),
            ('away_team_soccer_power_index', float),
            ('home_team_probability_win', float),
            ('away_team_probability_win', float),
            ('probability_draw', float),
            ('home_team_projected_score', float),
            ('away_team_projected_score', float),
            ('home_team_match_importance', float),
            ('away_team_match_importance', float),
            ('home_team__full_time_goals', int),
            ('away_team__full_time_goals', int),
            ('home_team__full_time_shot_expected_goals', float),
            ('away_team__full_time_shot_expected_goals', float),
            ('home_team__full_time_non_shot_expected_goals', float),
            ('away_team__full_time_non_shot_expected_goals', float),
            ('home_team__full_time_adjusted_goals', float),
            ('away_team__full_time_adjusted_goals', float),
        ]

    @classmethod
    def _get_outcomes(cls):
        return OUTCOMES

    @classmethod
    @lru_cache
    def _get_params(cls):
        data = _extract_data()
        full_param_grid = (
            data[['league', 'division', 'year']].drop_duplicates().to_dict('records')
        )
        return ParameterGrid(
            [
                {name: [val] for name, val in params.items()}
                for params in full_param_grid
            ]
        )

    @lru_cache
    def _get_data(self):
        data = _extract_data()
        data['match_quality'] = 2 / (1 / data['spi1'] + 1 / data['spi2'])
        data['fixtures'] = data['score1'].isna() & data['score2'].isna()
        data = data.drop(columns=self._removed_cols).rename(columns=self._cols_mapping)
        return data
