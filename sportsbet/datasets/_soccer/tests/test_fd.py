"""
Test the _fd module.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import pytest

from sportsbet.datasets._utils import (
    _DataLoader,
    _create_names_mapping_table,
    _combine_odds
)

CONFIG = [
    ('Div', 'division', int, None),
    ('Country', 'league', object, None),
    ('Date', 'date', np.datetime64, None),
    ('Home', 'home_team', object, None),
    ('Away', 'away_team', object, None),
    ('HG', 'full_time_home_team_goals', int, 'full_time_goals'),
    ('AG', 'full_time_away_team_goals', int, 'full_time_goals'),
    ('Feature', None, None, None),
    ('IWH', 'interwetten_home_win_odds', float, None),
    ('IWD', 'interwetten_draw_odds', float, None),
    ('IWA', 'interwetten_away_win_odds', float, None),
    ('WHH', 'william_hill_home_win_odds', float, None),
    ('WHD', 'william_hill_draw_odds', float, None),
    ('WHA', 'william_hill_away_win_odds', float, None),
    (None, 'year', int, None)
]
TARGETS = [
    ('home_win', lambda scores: scores.values[:, 0] > scores.values[:, 1], 'full_time_goals'),
    ('away_win', lambda scores: scores.values[:, 1] > scores.values[:, 1], 'full_time_goals'),
    ('draw', lambda scores: scores.values[:, 0] == scores.values[:, 1], 'full_time_goals'),
    ('over_2.5_goals', lambda scores: scores.values[:, 0] + scores.values[:, 1] > 2.5, 'full_time_goals'),
    ('under_2.5_goals', lambda scores: scores.values[:, 0] + scores.values[:, 1] < 2.5, 'full_time_goals'),
    ('home_win', lambda scores: scores.values[:, 0] > scores.values[:, 1] + 1.0, 'full_time_adjusted_goals'),
    ('away_win', lambda scores: scores.values[:, 1] > scores.values[:, 0] + 1.0, 'full_time_adjusted_goals'),
    ('draw', lambda scores: np.abs(scores.values[:, 0]- scores.values[:, 1]) <= 1.0, 'full_time_adjusted_goals'),
    ('over_2.5_goals', lambda scores: scores.values[:, 0] + scores.values[:, 1] > 2.5, 'full_time_adjusted_goals'),
    ('under_2.5_goals', lambda scores: scores.values[:, 0] + scores.values[:, 1] < 2.5, 'full_time_adjusted_goals')
]
FULL_PARAM_GRID = [
    {
        'league': ['Greece', 'Spain'], 
        'division': [1, 2],
        'year': range(1993, 2020)
    },
    {
        'league': ['England'], 
        'division': [1, 2, 3, 4, 5],
        'year': range(1990, 2020)
    }
]
CURRENT_DATE = datetime.now()
DATA = pd.DataFrame(
    {
        'Div':  [np.nan, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 3.0],
        'Country': [np.nan, 'Greece', 'Spain', 'Spain', 'England', 'England', np.nan, np.nan, 'France'],
        'Date': [np.nan, pd.Timestamp('17/3/2019'), pd.Timestamp('5/4/1997'), pd.Timestamp('3/4/1999'), pd.Timestamp('5/7/1997'), pd.Timestamp('3/4/1998'), pd.Timestamp('3/4/1998'), CURRENT_DATE + timedelta(days=5), CURRENT_DATE + timedelta(weeks=5)],
        'Home': [np.nan, 'Panathinaikos', 'Real Madrid', 'Barcelona', 'Arsenal', 'Liverpool', 'Liverpool', 'Barcelona', 'Monaco'],
        'Away': [np.nan, 'AEK', 'Barcelona', 'Real Madrid', 'Liverpool', 'Arsenal', 'Arsenal', 'Real Madrid', 'PSG'],
        'HG': [np.nan, 1, 2, 2, np.nan, 1, 1, np.nan, np.nan],
        'AG': [np.nan, 0, 1, 2, 2, 1, 2, np.nan, np.nan],
        'Feature': [np.nan, 2.0, np.nan, 3.0, np.nan, np.nan, 5.0, 1.0, 6.0],
        'IWH': [np.nan, 2, 1.5, 2.5, 3, 2, np.nan, 3, 1.5],
        'IWD': [np.nan, 2, 3.5, 4.5, 2.5, 4.5, 2.5, 2.5, 3.5],
        'IWA': [np.nan, 3, 2.5, 2, 2, 3.5, 3.5, 2, 2.5],
        'WHH': [np.nan, 3.5, 2.5, np.nan, 3.0, 2.0, 4.0, 3.5, 2.5],
        'WHD': [np.nan, 1.5, 2.5, np.nan, np.nan, np.nan, np.nan, 2.5, 1.5],
        'WHA': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2.0, 2.5]
    }
)


class DataLoader(_DataLoader):
    """Data loader for testing."""

    def _fetch_data(self, data=None):
        if data is None:
            self.data_ = DATA.copy()
            self.data_['year'] = self.data_['Date'].apply(lambda date: date.year)
            self.data_['test'] = np.concatenate([np.repeat(False, len(DATA) - 2), [True, True]])
        else:
            self.data_ = data
        return self
    
    def _fetch_full_param_grid(self, data=None):
        self.full_param_grid_ = ParameterGrid(FULL_PARAM_GRID)
        return self


def test_data_loader_init():
    """Test the initialization of data loader."""
    data_loader = DataLoader(CONFIG, TARGETS)
    assert data_loader.config == CONFIG
    assert data_loader.targets == TARGETS
    assert data_loader.param_grid is None
    assert data_loader.drop_na_cols is None
    assert data_loader.drop_na_rows is None
    assert data_loader.odds_type is None
    assert data_loader.testing_duration is None


def test_data_loader_raise_error_data():
    """Test the raise of error of data loader for data."""
    with pytest.raises(TypeError, match='Attribute `data_` should be a pandas dataframe. Got list instead.'):
        DataLoader(CONFIG, TARGETS).load(data=[4, 5])
    with pytest.raises(ValueError, match='Attribute `data_` should be a pandas dataframe with positive size.'):
        DataLoader(CONFIG, TARGETS).load(data=pd.DataFrame())
    with pytest.raises(KeyError, match='Attribute `data_` should include a boolean column `test`.'):
        DataLoader(CONFIG, TARGETS).load(data=pd.DataFrame({'feature': [3, 4]}))


def test_data_loader_attributes():
    """Test the attributes of data loader."""
    data_loader = DataLoader(CONFIG, TARGETS, odds_type='interwetten')
    data_loader.load()
    assert list(data_loader.param_grid_) ==  list(data_loader.full_param_grid_)
    assert data_loader.drop_na_cols_ == 0.0
    assert data_loader.drop_na_rows_ == 0.0
    assert data_loader.testing_duration_ == 1
    assert data_loader.odds_type_ == 'interwetten'
    pd.testing.assert_index_equal(data_loader.cols_odds_, pd.Index([
        'interwetten_home_win_odds',
        'interwetten_away_win_odds',
        'interwetten_draw_odds'
    ], dtype=object))
    pd.testing.assert_index_equal(data_loader.cols_outputs_, pd.Index([
        'full_time_home_team_goals',
        'full_time_away_team_goals'
    ], dtype=object))
    pd.testing.assert_index_equal(data_loader.dropped_na_rows_, pd.Index([0], dtype=int))
    pd.testing.assert_index_equal(data_loader.dropped_na_cols_, pd.Index([], dtype=object))


def test_data_loader_param_grid():
    """Test the use of param_grid parameter of data loader."""
    data_loader = DataLoader(CONFIG, TARGETS, param_grid={'league': ['Greece'], 'division': [1, 2]})
    data_loader.load()
    expected_param_grid = data_loader.param_grid.copy()
    expected_param_grid.update({'year': data_loader.full_param_grid_.param_grid[0]['year']})
    assert list(data_loader.param_grid_) == list(ParameterGrid(expected_param_grid))


def test_data_loader_drop_na_cols():
    """Test the use of drop_na_cols parameter of data loader."""
    data_loader = DataLoader(CONFIG, TARGETS, drop_na_cols=1.0, odds_type='interwetten')
    data_loader.load()
    pd.testing.assert_index_equal(data_loader.dropped_na_cols_, pd.Index([
        'league',
        'william_hill_home_win_odds',
        'william_hill_draw_odds',
        'william_hill_away_win_odds'
    ], dtype=object))


def test_data_loader_drop_na_rows():
    """Test the use of drop_na_rows parameter of data loader."""
    data_loader = DataLoader(CONFIG, TARGETS, drop_na_rows=1.0, odds_type='interwetten')
    data_loader.load()
    pd.testing.assert_index_equal(data_loader.dropped_na_rows_, pd.Index([0, 1, 2, 3, 4, 5, 6], dtype=int))
    data_loader = DataLoader(CONFIG, TARGETS, drop_na_rows=0.0, odds_type='interwetten')
    data_loader.load()
    pd.testing.assert_index_equal(data_loader.dropped_na_rows_, pd.Index([0], dtype=int))


def test_data_loader_param_grid_raise_error():
    """Test the raise of errors of data loader for parameters grid."""
    with pytest.raises(ValueError):
        DataLoader(CONFIG, TARGETS, param_grid=[{'league': ['England']}, {'league': ['Greece'], 'division': [1, 2, 3]}]).load()


def test_data_loader_parameters_raise_error():
    """Test the raise of errors of data loader for various parameters."""
    with pytest.raises(TypeError):
        DataLoader(CONFIG, TARGETS, drop_na_cols=1).load()
    with pytest.raises(ValueError):
        DataLoader(CONFIG, TARGETS, drop_na_cols=1.4).load()
    with pytest.raises(TypeError):
        DataLoader(CONFIG, TARGETS, drop_na_rows=1).load()
    with pytest.raises(ValueError):
        DataLoader(CONFIG, TARGETS, drop_na_rows=-0.2).load()
    with pytest.raises(TypeError):
        DataLoader(CONFIG, TARGETS, testing_duration=0.5).load()
    with pytest.raises(ValueError):
        DataLoader(CONFIG, TARGETS, testing_duration=0).load()
    with pytest.raises(ValueError):
        DataLoader(CONFIG, TARGETS, odds_type='market_average').load()


def test_data_loader_load():
    """Test the load method of data loader."""
    
    # Create data loader
    data_loader = DataLoader(CONFIG, TARGETS, param_grid={'league': ['Greece', 'Spain'], 'division': [1]}, odds_type='interwetten')
    
    # Get data
    bunch = data_loader.load()
    X_test, Y_test, O_test = bunch.testing.X, bunch.testing.Y, bunch.testing.O
    X_train, Y_train, O_train = bunch.training.X, bunch.training.Y, bunch.training.O
    
    # Expected data
    data_train = pd.DataFrame(
        {
            'division':  [1, 1],
            'league': ['Greece', 'Spain'],
            'date': [pd.Timestamp('17/3/2019'), pd.Timestamp('5/4/1997')],
            'home_team': ['Panathinaikos', 'Real Madrid'],
            'away_team': ['AEK', 'Barcelona'],
            'full_time_home_team_goals': [1, 2],
            'full_time_away_team_goals': [0, 1],
            'interwetten_home_win_odds': [2.0, 1.5],
            'interwetten_draw_odds': [2.0, 3.5],
            'interwetten_away_win_odds': [3.0, 2.5],
            'william_hill_home_win_odds': [3.5, 2.5],
            'william_hill_draw_odds': [1.5, 2.5],
            'william_hill_away_win_odds': [np.nan, np.nan],
            'year': [2019, 1997]
        }
    )
    data_test = pd.DataFrame(
        {
            'division':  [4, 3],
            'league': [None, 'France'],
            'date': [CURRENT_DATE + timedelta(days=5), CURRENT_DATE + timedelta(weeks=5)],
            'home_team': ['Barcelona', 'Monaco'],
            'away_team': ['Real Madrid', 'PSG'],
            'interwetten_home_win_odds': [3.0, 1.5],
            'interwetten_draw_odds': [2.5, 3.5],
            'interwetten_away_win_odds': [2.0, 2.5],       
            'william_hill_home_win_odds': [3.5, 2.5],
            'william_hill_draw_odds': [2.5, 1.5],
            'william_hill_away_win_odds': [2.0, 2.5],
            'year': [(CURRENT_DATE + timedelta(days=5)).year, (CURRENT_DATE + timedelta(weeks=5)).year]
        }
    )
    X_test_expected = pd.DataFrame(
        {
            'division':  [4],
            'league': [None],
            'date': [CURRENT_DATE + timedelta(days=5)],
            'home_team': ['Barcelona'],
            'away_team': ['Real Madrid'],
            'interwetten_home_win_odds': [3.0],
            'interwetten_draw_odds': [2.5],
            'interwetten_away_win_odds': [2.0],
            'william_hill_home_win_odds': [3.5],
            'william_hill_draw_odds': [2.5],
            'william_hill_away_win_odds': [2.0],
            'year': [(CURRENT_DATE + timedelta(days=5)).year]
        }
    )
    O_test_expected = pd.DataFrame(
        {
            'interwetten_home_win_odds': [3.0],
            'interwetten_away_win_odds': [2.0],
            'interwetten_draw_odds': [2.5]
        }
    )
    X_train_expected = pd.DataFrame(
        {
            'division':  [1, 1],
            'league': ['Greece', 'Spain'],
            'date': [pd.Timestamp('17/3/2019'), pd.Timestamp('5/4/1997')],
            'home_team': ['Panathinaikos', 'Real Madrid'],
            'away_team': ['AEK', 'Barcelona'],
            'interwetten_home_win_odds': [2.0, 1.5],
            'interwetten_draw_odds': [2.0, 3.5],
            'interwetten_away_win_odds': [3.0, 2.5],
            'william_hill_home_win_odds': [3.5, 2.5],
            'william_hill_draw_odds': [1.5, 2.5],
            'william_hill_away_win_odds': [np.nan, np.nan],
            'year': [2019, 1997]
        }
    )
    Y_train_expected = pd.DataFrame(
        {
            'home_win__full_time_goals': [True, True],
            'away_win__full_time_goals': [False, False],
            'draw__full_time_goals': [False, False],
        }
    )
    O_train_expected = pd.DataFrame(
        {
            'interwetten_home_win_odds': [2.0, 1.5],
            'interwetten_away_win_odds': [3.0, 2.5],
            'interwetten_draw_odds': [2.0, 3.5]
        }
    )
    
    # Assertions
    pd.testing.assert_frame_equal(data_loader.data_train_, data_train)
    pd.testing.assert_frame_equal(data_loader.data_test_, data_test)
    pd.testing.assert_frame_equal(X_test, X_test_expected)
    assert Y_test is None
    pd.testing.assert_frame_equal(O_test, O_test_expected)
    pd.testing.assert_frame_equal(X_train, X_train_expected)
    pd.testing.assert_frame_equal(Y_train, Y_train_expected)
    pd.testing.assert_frame_equal(O_train, O_train_expected)
    pd.testing.assert_index_equal(X_train.columns, X_test.columns)
    pd.testing.assert_index_equal(O_train.columns, O_test.columns)
    pd.testing.assert_frame_equal(bunch.params.selected, pd.DataFrame(data_loader.param_grid_))
    pd.testing.assert_frame_equal(bunch.params.available, pd.DataFrame(data_loader.full_param_grid_))
    pd.testing.assert_index_equal(bunch.dropped.cols, pd.Index([], dtype=object))
    pd.testing.assert_index_equal(bunch.dropped.rows, pd.Index([0], dtype=int))


def test_data_loader_return_only_params():
    """Test the load method of data loader when only the
    available parameter grid is returned."""
    data_loader = DataLoader(CONFIG, TARGETS, param_grid={'league': ['Greece', 'Spain'], 'division': [1]}, odds_type='interwetten')
    bunch = data_loader.load(return_only_params=True)
    assert bunch.training is None
    assert bunch.testing is None
    assert bunch.dropped.cols is None
    assert bunch.dropped.rows is None
    assert bunch.params.selected is None
    pd.testing.assert_frame_equal(bunch.params.available, pd.DataFrame(data_loader.full_param_grid_))


def test_create_names_mapping():
    """Test the creation of names mapping tables."""
    left_data = pd.DataFrame({'date': [1, 1, 2, 2], 'league': ['A', 'B', 'A', 'B'], 'team1': ['PAOK', 'AEK', 'Panathinaikos', 'Olympiakos'], 'team2': ['AEK', 'Panathinaikos', 'Olympiakos', 'PAOK']})
    right_data = pd.DataFrame({'Date': [1, 1, 2, 2], 'Div': ['A', 'B', 'A', 'B'], 'HomeTeam': ['PAOK Salonika', 'AEK Athens', 'Panathinaikos', 'Olympiakos Piraeus'], 'AwayTeam': ['AEK Athens', 'Panathinaikos', 'Olympiakos Piraeus', 'PAOK Salonika']})
    names_mapping = _create_names_mapping_table(left_data, right_data)
    pd.testing.assert_frame_equal(names_mapping, pd.DataFrame({'left_team': ['AEK', 'Olympiakos', 'PAOK', 'Panathinaikos'], 'right_team': ['AEK Athens', 'Olympiakos Piraeus', 'PAOK Salonika', 'Panathinaikos']}))


@pytest.mark.parametrize('odds,combined_odds', [
    (np.array([[2.0, 4.0], [5.0, 2.0]]), np.array([4.0 / 3.0, 10.0 / 7.0])),
    (np.array([[2.5, 2.5], [4.0, 4.0]]), np.array([2.5 / 2.0, 4.0 / 2.0]))
])
def test_combine_odds(odds, combined_odds):
    """Test the generation of combined odds."""
    np.testing.assert_array_equal(_combine_odds(odds), combined_odds)
