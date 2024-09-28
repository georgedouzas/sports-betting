"""Test the DummySoccerDataLoader class."""

import re
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid

from sportsbet.datasets import DummySoccerDataLoader


def test_get_all_params():
    """Test all parameters."""
    dataloader = DummySoccerDataLoader()
    all_params = dataloader.get_all_params()
    assert all_params == [
        {'division': 1, 'year': 1998},
        {'division': 1, 'league': 'France', 'year': 2000},
        {'division': 1, 'league': 'France', 'year': 2001},
        {'division': 1, 'league': 'Greece', 'year': 2017},
        {'division': 1, 'league': 'Greece', 'year': 2019},
        {'division': 1, 'league': 'Spain', 'year': 1997},
        {'division': 2, 'league': 'England', 'year': 1997},
        {'division': 2, 'league': 'Spain', 'year': 1999},
        {'division': 3, 'league': 'England', 'year': 1998},
    ]


def test_get_odds_types():
    """Test all parameters."""
    dataloader = DummySoccerDataLoader()
    assert dataloader.get_odds_types() == ['interwetten', 'williamhill']


def test_param_grid_default():
    """Test the default parameters grid."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    params = pd.DataFrame(dataloader.param_grid_)
    expected_params = pd.DataFrame(
        ParameterGrid([{param: [val] for param, val in params.items()} for params in dataloader.get_all_params()]),
    )
    cols = list(params.columns)
    pd.testing.assert_frame_equal(
        params[cols].sort_values(cols, ignore_index=True),
        expected_params[cols].sort_values(cols, ignore_index=True),
    )


def test_param_grid():
    """Test the parameters grid."""
    dataloader = DummySoccerDataLoader(param_grid={'division': [1]})
    dataloader.extract_train_data()
    params = pd.DataFrame(dataloader.param_grid_)
    expected_params = pd.DataFrame(
        ParameterGrid([{param: [val] for param, val in params.items()} for params in dataloader.get_all_params()]),
    )
    expected_params = expected_params[expected_params["division"] == 1]
    cols = list(params.columns)
    pd.testing.assert_frame_equal(
        params[cols].sort_values(cols, ignore_index=True),
        expected_params[cols].sort_values(cols, ignore_index=True),
    )


def test_param_grid_false_names():
    """Test the raise of value error for parameters grid for false names."""
    false_param_grid = {'Division': [4], 'league': ['Greece']}
    dataloader = DummySoccerDataLoader(false_param_grid)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Parameter grid includes the parameters name(s) ['Division'] that "
            "are not not allowed by available data.",
        ),
    ):
        dataloader.extract_train_data()


def test_param_grid_false_values():
    """Test the raise of value error for parameters grid for false values."""
    false_param_grid = {'division': [4], 'league': ['Greece']}
    dataloader = DummySoccerDataLoader(false_param_grid)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Parameter grid includes the parameters value(s) "
            "{'division': 4, 'league': 'Greece'} that are not allowed by "
            "available data.",
        ),
    ):
        dataloader.extract_train_data()


def test_drop_na_thres_default():
    """Test default value for drop na threshold."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    min_thres = 0.0
    assert dataloader.drop_na_thres_ == min_thres


@pytest.mark.parametrize('drop_na_thres', [1, 0])
def test_drop_na_thres_raise_type_error(drop_na_thres):
    """Test the raise of type error for check of drop na threshold."""
    dataloader = DummySoccerDataLoader()
    with pytest.raises(TypeError):
        dataloader.extract_train_data(drop_na_thres)


@pytest.mark.parametrize('drop_na_thres', [1.5, -0.4])
def test_drop_na_thres_raise_value_error(drop_na_thres):
    """Test the raise of value error for check of drop na threshold."""
    dataloader = DummySoccerDataLoader()
    drop_na_thres_min = 0.0
    suffix = '>= 0.0' if drop_na_thres < drop_na_thres_min else '<= 1.0'
    with pytest.raises(ValueError, match=f'drop_na_thres == {drop_na_thres}, must be {suffix}.'):
        dataloader.extract_train_data(drop_na_thres)


def test_odds_type_default():
    """Test default value for odds type."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    assert dataloader.odds_type_ is None


def test_odds_type():
    """Test value of odds type."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data(odds_type='interwetten')
    assert dataloader.odds_type_ == 'interwetten'


def test_odds_type_raise_type_error():
    """Test the raise of type error for check of odds type."""
    dataloader = DummySoccerDataLoader()
    with pytest.raises(
        TypeError,
        match='Parameter `odds_type` should be a prefix of available odds columns. Got `5` instead.',
    ):
        dataloader.extract_train_data(odds_type=cast(str, 5))


def test_odds_type_raise_value_error():
    """Test the raise of value error for check of odds type."""
    dataloader = DummySoccerDataLoader()
    with pytest.raises(
        ValueError,
        match='Parameter `odds_type` should be a prefix of available odds columns. Got `pinnacle` instead.',
    ):
        dataloader.extract_train_data(odds_type='pinnacle')


def test_drop_na_cols_default():
    """Test the dropped columns of data loader for the default value."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    pd.testing.assert_index_equal(
        dataloader.dropped_na_cols_,
        pd.Index(['odds__pinnacle__over_2.5__full_time_goals'], dtype=object),
    )


def test_drop_na_cols():
    """Test the dropped columns of data loader."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data(drop_na_thres=1.0)
    pd.testing.assert_index_equal(
        dataloader.dropped_na_cols_,
        pd.Index(
            [
                'league',
                'odds__interwetten__home_win__full_time_goals',
                'odds__williamhill__draw__full_time_goals',
                'odds__williamhill__away_win__full_time_goals',
                'odds__pinnacle__over_2.5__full_time_goals',
            ],
            dtype='object',
        ),
    )


def test_input_cols_default():
    """Test the input columns for default values."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    pd.testing.assert_index_equal(
        dataloader.input_cols_,
        pd.Index(
            [
                col
                for col in DummySoccerDataLoader.DATA.columns
                if col
                not in (
                    'target__home_team__full_time_goals',
                    'target__away_team__full_time_goals',
                    'fixtures',
                    'date',
                    'odds__pinnacle__over_2.5__full_time_goals',
                )
            ],
            dtype=object,
        ),
    )


def test_input_cols():
    """Test the input columns."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data(drop_na_thres=1.0)
    pd.testing.assert_index_equal(
        dataloader.input_cols_,
        pd.Index(
            [
                col
                for col in DummySoccerDataLoader.DATA.columns
                if col
                not in (
                    'target__home_team__full_time_goals',
                    'target__away_team__full_time_goals',
                    'fixtures',
                    'odds__williamhill__draw__full_time_goals',
                    'odds__williamhill__away_win__full_time_goals',
                    'odds__pinnacle__over_2.5__full_time_goals',
                    'date',
                    'league',
                    'odds__interwetten__home_win__full_time_goals',
                )
            ],
            dtype=object,
        ),
    )


def test_output_cols_default():
    """Test the output columns for default parameters."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    pd.testing.assert_index_equal(
        dataloader.output_cols_,
        pd.Index(
            [
                'output__home_win__full_time_goals',
                'output__away_win__full_time_goals',
                'output__draw__full_time_goals',
                'output__over_2.5__full_time_goals',
                'output__under_2.5__full_time_goals',
            ],
            dtype=object,
        ),
    )


def test_output_cols():
    """Test the output columns."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data(odds_type='interwetten')
    pd.testing.assert_index_equal(
        dataloader.output_cols_,
        pd.Index(
            [
                'output__home_win__full_time_goals',
                'output__draw__full_time_goals',
                'output__away_win__full_time_goals',
            ],
            dtype=object,
        ),
    )


def test_odds_cols_default():
    """Test the odds columns for default parameters."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    pd.testing.assert_index_equal(
        dataloader.odds_cols_,
        pd.Index([], dtype=object),
    )


def test_odds_cols():
    """Test the odds columns."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data(odds_type='williamhill')
    pd.testing.assert_index_equal(
        dataloader.odds_cols_,
        pd.Index(
            [
                'odds__williamhill__home_win__full_time_goals',
                'odds__williamhill__draw__full_time_goals',
                'odds__williamhill__away_win__full_time_goals',
            ],
        ),
    )


def test_extract_train_data_default():
    """Test the the train data columns for default parameters."""
    dataloader = DummySoccerDataLoader()
    X_train, Y_train, O_train = dataloader.extract_train_data()
    pd.testing.assert_frame_equal(
        X_train,
        pd.DataFrame(
            {
                'division': [1, 3, 1, 2, 1, 1, 1, 1],
                'league': [
                    'Spain',
                    'England',
                    np.nan,
                    'Spain',
                    'France',
                    'France',
                    'Greece',
                    'Greece',
                ],
                'year': [1997, 1998, 1998, 1999, 2000, 2001, 2017, 2019],
                'home_team': [
                    'Real Madrid',
                    'Liverpool',
                    'Liverpool',
                    'Barcelona',
                    'Lens',
                    'PSG',
                    'Olympiakos',
                    'Panathinaikos',
                ],
                'away_team': [
                    'Barcelona',
                    'Arsenal',
                    'Arsenal',
                    'Real Madrid',
                    'Monaco',
                    'Lens',
                    'Panathinaikos',
                    'AEK',
                ],
                'odds__interwetten__home_win__full_time_goals': [
                    1.5,
                    2.0,
                    np.nan,
                    2.5,
                    2.0,
                    3.0,
                    2.0,
                    2.0,
                ],
                'odds__interwetten__draw__full_time_goals': [
                    3.5,
                    4.5,
                    2.5,
                    4.5,
                    2.5,
                    2.5,
                    2.0,
                    2.0,
                ],
                'odds__interwetten__away_win__full_time_goals': [
                    2.5,
                    3.5,
                    3.5,
                    2.0,
                    3.0,
                    2.0,
                    2.0,
                    3.0,
                ],
                'odds__williamhill__home_win__full_time_goals': [
                    2.5,
                    2.0,
                    4.0,
                    2.0,
                    2.5,
                    2.5,
                    2.0,
                    3.5,
                ],
                'odds__williamhill__draw__full_time_goals': [
                    2.5,
                    np.nan,
                    np.nan,
                    np.nan,
                    2.5,
                    3.0,
                    2.0,
                    1.5,
                ],
                'odds__williamhill__away_win__full_time_goals': [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    3.0,
                    2.5,
                    2.0,
                    np.nan,
                ],
            },
        ).set_index(
            pd.DatetimeIndex(
                [
                    '5/4/1997',
                    '3/4/1998',
                    '3/4/1998',
                    '3/4/1999',
                    '3/4/2000',
                    '6/4/2001',
                    '17/3/2017',
                    '17/3/2019',
                ],
                name='date',
                dayfirst=True,
            ),
        ),
    )
    pd.testing.assert_frame_equal(
        Y_train,
        pd.DataFrame(
            {
                'output__home_win__full_time_goals': [
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                ],
                'output__away_win__full_time_goals': [
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                'output__draw__full_time_goals': [
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                ],
                'output__over_2.5__full_time_goals': [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                ],
                'output__under_2.5__full_time_goals': [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                ],
            },
        ),
    )
    assert O_train is None


def test_extract_train_data():
    """Test the the train data."""
    dataloader = DummySoccerDataLoader(param_grid={'league': ['Greece']})
    X_train, Y_train, O_train = dataloader.extract_train_data(drop_na_thres=1.0, odds_type='interwetten')
    assert O_train is not None
    pd.testing.assert_frame_equal(
        X_train,
        pd.DataFrame(
            {
                'division': [1, 1],
                'league': ['Greece', 'Greece'],
                'year': [2017, 2019],
                'home_team': ['Olympiakos', 'Panathinaikos'],
                'away_team': ['Panathinaikos', 'AEK'],
                'odds__interwetten__home_win__full_time_goals': [2.0, 2.0],
                'odds__interwetten__draw__full_time_goals': [2.0, 2.0],
                'odds__interwetten__away_win__full_time_goals': [2.0, 3.0],
                'odds__williamhill__home_win__full_time_goals': [2.0, 3.5],
                'odds__williamhill__draw__full_time_goals': [2.0, 1.5],
            },
        ).set_index(pd.DatetimeIndex(['17/3/2017', '17/3/2019'], name='date', dayfirst=True)),
    )
    pd.testing.assert_frame_equal(
        Y_train,
        pd.DataFrame(
            {
                'output__home_win__full_time_goals': [False, True],
                'output__draw__full_time_goals': [True, False],
                'output__away_win__full_time_goals': [False, False],
            },
        ),
    )
    pd.testing.assert_frame_equal(
        O_train,
        pd.DataFrame(
            {
                'odds__interwetten__home_win__full_time_goals': [2.0, 2.0],
                'odds__interwetten__draw__full_time_goals': [2.0, 2.0],
                'odds__interwetten__away_win__full_time_goals': [2.0, 3.0],
            },
        ),
    )
    assert Y_train.shape == O_train.shape


def test_extract_fixtures_data():
    """Test the fixtures data."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data(odds_type='interwetten', drop_na_thres=1.0)
    X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
    assert Y_fix is None
    assert O_fix is not None
    pd.testing.assert_frame_equal(
        X_fix,
        pd.DataFrame(
            {
                'division': [4, 3],
                'year': [
                    DummySoccerDataLoader.DATE.year,
                    DummySoccerDataLoader.DATE.year,
                ],
                'home_team': ['Barcelona', 'Monaco'],
                'away_team': ['Real Madrid', 'PSG'],
                'odds__interwetten__draw__full_time_goals': [2.5, 3.5],
                'odds__interwetten__away_win__full_time_goals': [2.0, 2.5],
                'odds__williamhill__home_win__full_time_goals': [3.5, 2.5],
            },
        ).set_index(
            pd.DatetimeIndex(
                [
                    DummySoccerDataLoader.DATE.date().strftime('%d/%m/%Y'),
                    DummySoccerDataLoader.DATE.date().strftime('%d/%m/%Y'),
                ],
                name='date',
                dayfirst=True,
            ),
        ),
    )
    pd.testing.assert_frame_equal(
        O_fix,
        pd.DataFrame(
            {
                'odds__interwetten__home_win__full_time_goals': [3.0, 1.5],
                'odds__interwetten__draw__full_time_goals': [2.5, 3.5],
                'odds__interwetten__away_win__full_time_goals': [2.0, 2.5],
            },
        ),
    )


def test_extract_fixtures_data_raise_error():
    """Test the raise of error when fixtures data are extracted."""
    dataloader = DummySoccerDataLoader()
    with pytest.raises(
        AttributeError,
        match='Extract the training data before extracting the fixtures data.',
    ):
        dataloader.extract_fixtures_data()
