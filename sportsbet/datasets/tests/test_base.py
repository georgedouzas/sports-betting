"""
Test the _base module.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import pytest

from sportsbet.datasets import DummySoccerDataLoader


def test_get_all_params():
    """Test all parameters."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    assert list(dataloader.get_all_params()) == list(
        ParameterGrid(DummySoccerDataLoader.PARAMS)
    )


def test_default_param_grid():
    """Test the default parameters grid."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    assert list(dataloader.param_grid_) == list(dataloader.get_all_params())


def test_param_grid():
    """Test the parameters grid."""
    dataloader = DummySoccerDataLoader(param_grid={'division': [1]})
    dataloader.extract_train_data()
    assert list(dataloader.param_grid_) == list(
        ParameterGrid(
            [
                params
                for params in DummySoccerDataLoader.PARAMS
                if params['division'] == [1]
            ]
        )
    )


def test_param_grid_raise_value_error():
    """Test the raise of value error for parameters grid."""
    dataloader = DummySoccerDataLoader(
        param_grid={'division': [4], 'league': ['Greece']}
    )
    with pytest.raises(
        ValueError,
        match='Parameter grid includes values not allowed by available data.',
    ):
        dataloader.extract_train_data()


def test_drop_na_thres_default():
    """Test default value for drop na threshold."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    assert dataloader.drop_na_thres_ == 0.0


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
    with pytest.raises(ValueError):
        dataloader.extract_train_data(drop_na_thres)


def test_odds_type_default():
    """Test default value for odds type."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    assert dataloader.odds_type_ is None


def test_odds_type():
    """Test check for odds type."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data(odds_type='interwetten')
    assert dataloader.odds_type_ == 'interwetten'


def test_odds_type_raise_type_error():
    """Test the raise of type error for check of odds type."""
    dataloader = DummySoccerDataLoader()
    with pytest.raises(
        TypeError,
        match='Parameter `odds_type` should be a string or None. Got int instead.',
    ):
        dataloader.extract_train_data(odds_type=5)


def test_odds_type_raise_value_error():
    """Test the raise of value error for check of odds type."""
    dataloader = DummySoccerDataLoader()
    with pytest.raises(
        ValueError,
        match='Parameter `odds_type` should be a prefix of available odds columns. '
        'Got bet365 instead.',
    ):
        dataloader.extract_train_data(odds_type='bet365')


def test_data_non_df():
    """Test the raise of error of check data."""
    dataloader = DummySoccerDataLoader().set_data([4, 5])
    with pytest.raises(
        TypeError,
        match='Data should be a pandas dataframe. Got list instead.',
    ):
        dataloader.extract_train_data()


def test_data_empty_df():
    """Test the raise of error of check data."""
    dataloader = DummySoccerDataLoader().set_data(pd.DataFrame())
    with pytest.raises(
        ValueError,
        match='Data should be a pandas dataframe with positive size.',
    ):
        dataloader.extract_train_data()


def test_data_fixtures_col():
    """Test the raise of error of check data."""
    dataloader = DummySoccerDataLoader().set_data(pd.DataFrame({'Div': [3, 4]}))
    with pytest.raises(
        KeyError,
        match='Data should include a boolean column `fixtures` to distinguish between '
        'train and fixtures data.',
    ):
        dataloader.extract_train_data()


def test_data_date_col():
    """Test the raise of error of check data."""
    dataloader = DummySoccerDataLoader().set_data(
        pd.DataFrame({'Div': [3, 4], 'fixtures': [True, True]})
    )
    with pytest.raises(
        KeyError,
        match='Data should include a datetime column `date` to represent the date.',
    ):
        dataloader.extract_train_data()


def test_schema_output_cols_default():
    """Test the schema output columns."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    pd.testing.assert_index_equal(
        dataloader.schema_output_cols_,
        pd.Index(['home_team__full_time_goals', 'away_team__full_time_goals']),
    )


def test_schema_input_cols_default():
    """Test the schema input columns."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    pd.testing.assert_index_equal(
        dataloader.schema_input_cols_,
        pd.Index(
            [
                'division',
                'league',
                'date',
                'home_team',
                'away_team',
                'interwetten__home_win__odds',
                'interwetten__draw__odds',
                'interwetten__away_win__odds',
                'williamhill__home_win__odds',
                'williamhill__draw__odds',
                'williamhill__away_win__odds',
                'year',
            ]
        ),
    )


def test_schema_odds_cols_default():
    """Test the schema odds columns."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    pd.testing.assert_index_equal(
        dataloader.schema_odds_cols_,
        pd.Index(
            [
                'interwetten__home_win__odds',
                'interwetten__draw__odds',
                'interwetten__away_win__odds',
                'williamhill__home_win__odds',
                'williamhill__draw__odds',
                'williamhill__away_win__odds',
            ]
        ),
    )


def test_drop_na_cols_default():
    """Test the dropped columns of data loader for the default value."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    pd.testing.assert_index_equal(
        dataloader.dropped_na_cols_, pd.Index([], dtype=object)
    )


def test_drop_na_cols():
    """Test the dropped columns of data loader."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data(drop_na_thres=1.0)
    pd.testing.assert_index_equal(
        dataloader.dropped_na_cols_,
        pd.Index(
            ['williamhill__draw__odds', 'williamhill__away_win__odds'], dtype=object
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
                    'home_team__full_time_goals',
                    'away_team__full_time_goals',
                    'fixtures',
                    'date',
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
                    'home_team__full_time_goals',
                    'away_team__full_time_goals',
                    'fixtures',
                    'williamhill__draw__odds',
                    'williamhill__away_win__odds',
                    'date',
                )
            ],
            dtype=object,
        ),
    )


def test_output_cols():
    """Test the output columns."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data()
    pd.testing.assert_index_equal(
        dataloader.output_cols_,
        pd.Index(
            ['home_team__full_time_goals', 'away_team__full_time_goals'], dtype=object
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
                'williamhill__away_win__odds',
                'williamhill__draw__odds',
                'williamhill__home_win__odds',
            ]
        ),
    )


def test_extract_train_data_default():
    """Test the the train data columns for default parameters."""
    dataloader = DummySoccerDataLoader()
    X, Y, Odds = dataloader.extract_train_data()
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame(
            {
                'division': [1, 3, 2, 1, 1, 1, 1],
                'league': [
                    'Spain',
                    'England',
                    'Spain',
                    'France',
                    'France',
                    'Greece',
                    'Greece',
                ],
                'year': [1997, 1998, 1999, 2000, 2001, 2017, 2019],
                'home_team': [
                    'Real Madrid',
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
                    'Real Madrid',
                    'Monaco',
                    'Lens',
                    'Panathinaikos',
                    'AEK',
                ],
                'interwetten__home_win__odds': [1.5, 2.0, 2.5, 2.0, 3.0, 2.0, 2.0],
                'interwetten__draw__odds': [3.5, 4.5, 4.5, 2.5, 2.5, 2.0, 2.0],
                'interwetten__away_win__odds': [2.5, 3.5, 2.0, 3.0, 2.0, 2.0, 3.0],
                'williamhill__home_win__odds': [2.5, 2.0, 2.0, 2.5, 2.5, 2.0, 3.5],
                'williamhill__draw__odds': [2.5, np.nan, np.nan, 2.5, 3.0, 2.0, 1.5],
                'williamhill__away_win__odds': [
                    np.nan,
                    np.nan,
                    np.nan,
                    3.0,
                    2.5,
                    2.0,
                    np.nan,
                ],
            }
        ).set_index(
            pd.Index(
                [
                    pd.Timestamp('5/4/1997'),
                    pd.Timestamp('3/4/1998'),
                    pd.Timestamp('3/4/1999'),
                    pd.Timestamp('3/4/2000'),
                    pd.Timestamp('6/4/2001'),
                    pd.Timestamp('17/3/2017'),
                    pd.Timestamp('17/3/2019'),
                ],
                name='date',
            )
        ),
    )
    pd.testing.assert_frame_equal(
        Y,
        pd.DataFrame(
            {
                'away_win__full_time_goals': [
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                'draw__full_time_goals': [
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                ],
                'home_win__full_time_goals': [
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                ],
                'over_2.5_goals__full_time_goals': [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                ],
                'under_2.5_goals__full_time_goals': [
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                ],
            }
        ),
    )
    assert Odds is None


def test_extract_train_data():
    """Test the the train data."""
    dataloader = DummySoccerDataLoader()
    X, Y, Odds = dataloader.extract_train_data(odds_type='interwetten')
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame(
            {
                'division': [1, 3, 2, 1, 1, 1, 1],
                'league': [
                    'Spain',
                    'England',
                    'Spain',
                    'France',
                    'France',
                    'Greece',
                    'Greece',
                ],
                'year': [1997, 1998, 1999, 2000, 2001, 2017, 2019],
                'home_team': [
                    'Real Madrid',
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
                    'Real Madrid',
                    'Monaco',
                    'Lens',
                    'Panathinaikos',
                    'AEK',
                ],
                'interwetten__home_win__odds': [1.5, 2.0, 2.5, 2.0, 3.0, 2.0, 2.0],
                'interwetten__draw__odds': [3.5, 4.5, 4.5, 2.5, 2.5, 2.0, 2.0],
                'interwetten__away_win__odds': [2.5, 3.5, 2.0, 3.0, 2.0, 2.0, 3.0],
                'williamhill__home_win__odds': [2.5, 2.0, 2.0, 2.5, 2.5, 2.0, 3.5],
                'williamhill__draw__odds': [2.5, np.nan, np.nan, 2.5, 3.0, 2.0, 1.5],
                'williamhill__away_win__odds': [
                    np.nan,
                    np.nan,
                    np.nan,
                    3.0,
                    2.5,
                    2.0,
                    np.nan,
                ],
            }
        ).set_index(
            pd.Index(
                [
                    pd.Timestamp('5/4/1997'),
                    pd.Timestamp('3/4/1998'),
                    pd.Timestamp('3/4/1999'),
                    pd.Timestamp('3/4/2000'),
                    pd.Timestamp('6/4/2001'),
                    pd.Timestamp('17/3/2017'),
                    pd.Timestamp('17/3/2019'),
                ],
                name='date',
            )
        ),
    )
    pd.testing.assert_frame_equal(
        Y,
        pd.DataFrame(
            {
                'away_win__full_time_goals': [
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                'draw__full_time_goals': [
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                ],
                'home_win__full_time_goals': [
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                ],
            }
        ),
    )
    pd.testing.assert_frame_equal(
        Odds,
        pd.DataFrame(
            {
                'interwetten__away_win__odds': [2.5, 3.5, 2.0, 3.0, 2.0, 2.0, 3.0],
                'interwetten__draw__odds': [3.5, 4.5, 4.5, 2.5, 2.5, 2.0, 2.0],
                'interwetten__home_win__odds': [1.5, 2.0, 2.5, 2.0, 3.0, 2.0, 2.0],
            }
        ),
    )
    assert Y.shape == Odds.shape


def test_extract_fixtures_data():
    """Test the fixtures data."""
    dataloader = DummySoccerDataLoader()
    dataloader.extract_train_data(odds_type='interwetten')
    X, Y, Odds = dataloader.extract_fixtures_data()
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame(
            {
                'division': [4, 3],
                'league': [np.nan, 'France'],
                'year': [
                    DummySoccerDataLoader.DATE.year,
                    DummySoccerDataLoader.DATE.year,
                ],
                'home_team': ['Barcelona', 'Monaco'],
                'away_team': [
                    'Real Madrid',
                    'PSG',
                ],
                'interwetten__home_win__odds': [3.0, 1.5],
                'interwetten__draw__odds': [2.5, 3.5],
                'interwetten__away_win__odds': [2.0, 2.5],
                'williamhill__home_win__odds': [3.5, 2.5],
                'williamhill__draw__odds': [2.5, 1.5],
                'williamhill__away_win__odds': [2.0, 2.5],
            }
        ).set_index(
            pd.DatetimeIndex(
                [
                    DummySoccerDataLoader.DATE,
                    DummySoccerDataLoader.DATE,
                ],
                name='date',
            )
        ),
    )
    assert Y is None
    pd.testing.assert_frame_equal(
        Odds,
        pd.DataFrame(
            {
                'interwetten__away_win__odds': [2.0, 2.5],
                'interwetten__draw__odds': [2.5, 3.5],
                'interwetten__home_win__odds': [3.0, 1.5],
            }
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
