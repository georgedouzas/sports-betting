"""
Test the _utils module.
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
    ('Div', 'division', int),
    ('Country', 'league', object),
    ('Date', 'date', np.datetime64),
    ('Home', 'home_team', object),
    ('Away', 'away_team', object),
    ('HG', 'home_team__full_time_goals', int),
    ('AG', 'away_team__full_time_goals', int),
    ('Feature', None, None),
    ('IWH', 'interwetten__home_win__odds', float),
    ('IWD', 'interwetten__draw__odds', float),
    ('IWA', 'interwetten__away_win__odds', float),
    ('WHH', 'william_hill__home_win__odds', float),
    ('WHD', 'william_hill__draw__odds', float),
    ('WHA', 'william_hill__away_win__odds', float),
    (None, 'year', int)
]
TARGETS = [
    ('home_win__full_time_goals', lambda outputs: outputs['home_team__full_time_goals'] > outputs['away_team__full_time_goals']),
    ('away_win__full_time_goals', lambda outputs: outputs['home_team__full_time_goals'] < outputs['away_team__full_time_goals']),
    ('draw__full_time_goals', lambda outputs: outputs['home_team__full_time_goals'] == outputs['away_team__full_time_goals']),
    ('over_2.5_goals__full_time_goals', lambda outputs: outputs['home_team__full_time_goals'] + outputs['away_team__full_time_goals'] > 2.5),
    ('under_2.5_goals__full_time_goals', lambda outputs: outputs['home_team__full_time_goals'] + outputs['away_team__full_time_goals'] < 2.5),

    ('home_win__full_time_adjusted_goals', lambda outputs: outputs['home_team__full_time_goals'] > outputs['away_team__full_time_goals'] + 1.0),
    ('away_win__full_time_adjusted_goals', lambda outputs: outputs['away_team__full_time_goals'] > outputs['home_team__full_time_goals'] + 1.0),
    ('draw__full_time_adjusted_goals', lambda outputs: np.abs(outputs['home_team__full_time_goals'] -  outputs['away_team__full_time_goals']) <= 1.0),
    ('over_2.5_goals__full_time_adjusted_goals', lambda outputs: outputs['home_team__full_time_goals'] + outputs['away_team__full_time_goals'] > 2.5),
    ('under_2.5_goals__full_time_adjusted_goals', lambda outputs: outputs['home_team__full_time_goals'] + outputs['away_team__full_time_goals'] < 2.5)
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
    
    def _fetch_full_param_grid(self, data):
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


@pytest.mark.parametrize('odds_type', [None, 'interwetten', 'william_hill'])
def test_data_loader_check_parameters(odds_type):
    """Test the check of parameters of data loader."""
    data_loader = DataLoader(CONFIG, TARGETS, odds_type=odds_type)
    data_loader.load()
    assert data_loader.drop_na_cols_ == 0.0
    assert data_loader.drop_na_rows_ == 0.0
    assert data_loader.testing_duration_ == 1
    assert data_loader.odds_type_ == ('' if odds_type is None else odds_type)


def test_data_loader_check_parameters_raise_error():
    """Test the raise of errors for check of parameters of data loader."""
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
    with pytest.raises(ValueError, match='Parameter `odds_type` should be a prefix of available odds columns. Got bet365 instead.'):
        DataLoader(CONFIG, TARGETS, odds_type='bet365').load()


@pytest.mark.parametrize('param_grid', [None, {'league': ['Greece'], 'division': [1, 2]}])
def test_data_loader_check_param_grid(param_grid):
    """Test the check of parameters grid of data loader."""
    data_loader = DataLoader(CONFIG, TARGETS, param_grid=param_grid)
    data_loader.load()
    if param_grid is None:
        assert list(data_loader.param_grid_) == list(data_loader.full_param_grid_)
    else:
        expected_param_grid = param_grid.copy()
        expected_param_grid.update({'year': data_loader.full_param_grid_.param_grid[0]['year']})
        assert list(data_loader.param_grid_) == list(ParameterGrid(expected_param_grid))


def test_data_loader_check_param_grid_raise_error():
    """Test the raise of error of parameters grid of data loader."""
    with pytest.raises(ValueError):
        DataLoader(CONFIG, TARGETS, param_grid=[{'league': ['England']}, {'league': ['Greece'], 'division': [1, 2, 3]}]).load()


def test_data_loader_check_data_raise_error():
    """Test the raise of error of data_ attribute of data loader."""
    with pytest.raises(TypeError, match='Attribute `data_` should be a pandas dataframe. Got list instead.'):
        DataLoader(CONFIG, TARGETS).load(data=[4, 5])
    with pytest.raises(ValueError, match='Attribute `data_` should be a pandas dataframe with positive size.'):
        DataLoader(CONFIG, TARGETS).load(data=pd.DataFrame())
    with pytest.raises(KeyError, match='Attribute `data_` should include a boolean column `test`.'):
        DataLoader(CONFIG, TARGETS).load(data=pd.DataFrame({'feature': [3, 4]}))
    with pytest.raises(ValueError, match='All columns of attribute `data_` should be included in the schema.'):
        DataLoader(CONFIG, TARGETS).load(data=pd.DataFrame({'feature': [3, 4], 'test': [True, False]}))


@pytest.mark.parametrize('odds_type', [None, 'interwetten', 'william_hill'])
def test_data_loader_check_cols(odds_type):
    """Test the check of columns of data loader."""
    
    # Create data loader and load data
    data_loader = DataLoader(CONFIG, TARGETS, odds_type=odds_type)
    data_loader.load()

    # Assert inputs
    pd.testing.assert_index_equal(data_loader.cols_inputs_, pd.Index([
        'division',
        'league',
        'date',
        'home_team',
        'away_team',
        'interwetten__home_win__odds',
        'interwetten__draw__odds',
        'interwetten__away_win__odds',
        'william_hill__home_win__odds',
        'william_hill__draw__odds',
        'william_hill__away_win__odds',
        'year'
    ], dtype=object))
    
    # Assert odds
    if odds_type is None:
        cols_odds = []
    elif odds_type == 'interwetten':
        cols_odds = [
            'interwetten__home_win__odds',
            'interwetten__draw__odds',
            'interwetten__away_win__odds'
        ]
    elif odds_type == 'william_hill':
        cols_odds = [
            'william_hill__home_win__odds',
            'william_hill__draw__odds',
            'william_hill__away_win__odds'
        ]
    pd.testing.assert_index_equal(data_loader.cols_odds_, pd.Index(cols_odds, dtype=object))
    
    # Assert outputs
    pd.testing.assert_index_equal(data_loader.cols_outputs_, pd.Index([
        'home_team__full_time_goals',
        'away_team__full_time_goals'
    ], dtype=object))


@pytest.mark.parametrize('drop_na_cols', [0.0, 1.0])
def test_data_loader_dropped_na_cols(drop_na_cols):
    """Test the dropped columns of data loader."""
    data_loader = DataLoader(CONFIG, TARGETS, drop_na_cols=drop_na_cols)
    data_loader.load()
    if drop_na_cols == 0.0:
        pd.testing.assert_index_equal(data_loader.dropped_na_cols_, pd.Index([], dtype=object))
    elif drop_na_cols == 1.0:
        pd.testing.assert_index_equal(data_loader.dropped_na_cols_, pd.Index([
            'league',
            'interwetten__home_win__odds',
            'william_hill__home_win__odds',
            'william_hill__draw__odds',
            'william_hill__away_win__odds'
        ], dtype=object))


@pytest.mark.parametrize('drop_na_rows', [0.0, 1.0])
def test_data_loader_dropped_na_rows(drop_na_rows):
    """Test the dropped rows of data loader."""
    data_loader = DataLoader(CONFIG, TARGETS, drop_na_rows=drop_na_rows)
    data_loader.load()
    if drop_na_rows == 0.0:
        pd.testing.assert_index_equal(data_loader.dropped_na_rows_, pd.Index([0], dtype=int))
    elif drop_na_rows == 1.0:
        pd.testing.assert_index_equal(data_loader.dropped_na_rows_, pd.Index([0, 1, 2, 3, 4, 5, 6], dtype=int))
    

def test_data_loader_load():
    """Test the load method of data loader."""
    
    # Create data loader
    data_loader = DataLoader(CONFIG, TARGETS, param_grid={'league': ['Greece', 'Spain'], 'division': [1]}, odds_type='interwetten', drop_na_cols=1.0)
    
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
