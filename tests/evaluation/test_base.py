"""Test the base Bettor class."""

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from sportsbet.evaluation._base import BaseBettor
from tests.evaluation import O_train, TestBettor, X_train, Y_train


def test_abstract_class_raise_error():
    """Test abstract method missing implementation."""

    class TestBettor(BaseBettor):
        pass

    with pytest.raises(
        TypeError,
        match='Can\'t instantiate abstract class TestBettor',
    ):
        TestBettor()


def test_fit_input_output_data_length_value_error():
    """Test raising an error on inconsistent data length."""
    bettor = TestBettor()
    with pytest.raises(
        ValueError,
        match=re.escape(
            f'Found input variables with inconsistent numbers of samples: [{X_train.shape[0] -1}, {Y_train.shape[0]}]',
        ),
    ):
        bettor.fit(X_train.iloc[:-1], Y_train)


def test_bet_input_odds_data_length_value_error():
    """Test raising an error on inconsistent data length."""
    bettor = TestBettor()
    bettor.fit(X_train, Y_train)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f'Found input variables with inconsistent numbers of samples: [{X_train.shape[0] -1}, {O_train.shape[0]}]',
        ),
    ):
        bettor.bet(X_train.iloc[:-1], O_train)


@pytest.mark.parametrize('X', [X_train.to_numpy(), pd.DataFrame(X_train.to_numpy())])
def test_fit_input_data_type_error(X):
    """Test raising an error on the wrong input data."""
    bettor = TestBettor()
    with pytest.raises(
        TypeError,
        match='Input data `X` should be pandas dataframe with a date index.',
    ):
        bettor.fit(X, Y_train)


@pytest.mark.parametrize('Y', [Y_train.to_numpy(), Y_train.to_numpy().tolist()])
def test_fit_output_data_type_error(Y):
    """Test raising an error on the wrong output data."""
    bettor = TestBettor()
    with pytest.raises(
        TypeError,
        match='Output data `Y` should be pandas dataframe.',
    ):
        bettor.fit(X_train, Y)


def test_fit_output_data_cols_names_value_error():
    """Test raising an error on the wrong output data columns names."""
    Y = Y_train.rename(columns={'output__home_win__full_time_goals': 'home_win__full_time_goals'})
    bettor = TestBettor()
    with pytest.raises(
        ValueError,
        match='Output data column names should follow a naming',
    ):
        bettor.fit(X_train, Y)


def test_fit_output_data_cols_prefixes_value_error():
    """Test raising an error on the wrong output data columns prefixes."""
    Y = Y_train.rename(columns={'output__home_win__full_time_goals': 'outputs__home_win__full_time_goals'})
    bettor = TestBettor()
    with pytest.raises(
        ValueError,
        match='Prefixes of output data column names should be equal to `output`.',
    ):
        bettor.fit(X_train, Y)


@pytest.mark.parametrize('O', [O_train.to_numpy(), O_train.to_numpy().tolist()])
def test_fit_odds_data_type_error(O):
    """Test raising an error on the wrong odds data."""
    bettor = TestBettor()
    bettor.fit(X_train, Y_train)
    with pytest.raises(
        TypeError,
        match='Odds data `O` should be pandas dataframe.',
    ):
        bettor.bet(X_train, O)


def test_fit_odds_data_cols_names_value_error():
    """Test raising an error on the wrong odds data columns names."""
    O = O_train.rename(columns={'odds__williamhill__home_win__full_time_goals': 'home_win__full_time_goals'})
    bettor = TestBettor()
    bettor.fit(X_train, Y_train)
    with pytest.raises(
        ValueError,
        match='Odds data column names should follow a naming',
    ):
        bettor.bet(X_train, O)


def test_fit_odds_data_cols_prefixes_value_error():
    """Test raising an error on the wrong odds data columns prefixes."""
    O = O_train.rename(
        columns={'odds__williamhill__home_win__full_time_goals': 'odd__williamhill__home_win__full_time_goals'},
    )
    bettor = TestBettor()
    bettor.fit(X_train, Y_train)
    with pytest.raises(
        ValueError,
        match='Prefixes of odds data column names should be equal to `odds`.',
    ):
        bettor.bet(X_train, O)


@pytest.mark.parametrize(
    'betting_markets',
    ['home_win__full_time_goals', ('home_win__full_time_goals', 'over_2.5__full_time_goals')],
)
def test_fit_betting_markets_raise_type_error(betting_markets):
    """Test raising an error on the wrong betting markets type."""
    bettor = TestBettor(betting_markets=betting_markets)
    with pytest.raises(
        TypeError,
        match='Parameter `betting_markets` should be a list of betting market names.',
    ):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize(
    'betting_markets',
    [['under_2.5__full_time_goals'], ['home_win__full_time_goals', 'over_2.5__full_time_goals']],
)
def test_fit_betting_markets_raise_value_error(betting_markets):
    """Test raising an error on the wrong betting markets value."""
    bettor = TestBettor(betting_markets=betting_markets)
    with pytest.raises(
        ValueError,
        match='Parameter `betting_markets` does not contain valid names.',
    ):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize('init_cash', [[4.5], 'init_cash'])
def test_fit_init_cash_raise_type_error(init_cash):
    """Test raising a type error on the fit method."""
    bettor = TestBettor(init_cash=init_cash)
    with pytest.raises(
        TypeError,
        match=f'init_cash must be an instance of {{float, int}}, not {type(init_cash).__name__}.',
    ):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize('init_cash', [0.0, -30.0])
def test_fit_init_cash_raise_value_error(init_cash):
    """Test raising a value error on the fit method."""
    bettor = TestBettor(init_cash=init_cash)
    with pytest.raises(ValueError, match=f"init_cash == {init_cash}, must be > 0.0."):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize('stake', [[20.0], 'stake'])
def test_fit_stake_raise_type_error(stake):
    """Test raising a type error on the fit method."""
    bettor = TestBettor(stake=stake)
    with pytest.raises(
        TypeError,
        match=f'stake must be an instance of {{float, int}}, not {type(stake).__name__}.',
    ):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize('stake', [0.0, -30.0])
def test_fit_stake_raise_value_error(stake):
    """Test raising a value error on the fit method."""
    bettor = TestBettor(stake=stake)
    with pytest.raises(ValueError, match=f"stake == {stake}, must be > 0.0."):
        bettor.fit(X_train, Y_train)


def test_fit_default():
    """Test the fit method with default parameters."""
    default_betting_markets = np.array(
        ['home_win__full_time_goals', 'draw__full_time_goals', 'away_win__full_time_goals'],
    )
    default_init_cash = 1e4
    default_stake = 50.0
    bettor = TestBettor()
    bettor.fit(X_train, Y_train)
    assert np.array_equal(bettor.betting_markets_, default_betting_markets)
    assert bettor.init_cash_ == default_init_cash
    assert bettor.stake_ == default_stake


@pytest.mark.parametrize(
    'betting_markets',
    [['draw__full_time_goals', 'away_win__full_time_goals'], ['draw__full_time_goals']],
)
def test_fit_betting_markets(betting_markets):
    """Test the fit method and betting markets."""
    bettor = TestBettor(betting_markets=betting_markets)
    bettor.fit(X_train, Y_train)
    assert np.array_equal(bettor.betting_markets_, betting_markets)


@pytest.mark.parametrize('init_cash', [100, 1e3])
def test_fit_init_cash(init_cash):
    """Test the fit method and initialization of cash."""
    bettor = TestBettor(init_cash=init_cash)
    bettor.fit(X_train, Y_train)
    assert bettor.init_cash_ == float(init_cash)


@pytest.mark.parametrize('init_cash', [100, 1e3])
def test_fit_stake(init_cash):
    """Test the fit method and stakes."""
    bettor = TestBettor(init_cash=init_cash)
    bettor.fit(X_train, Y_train)
    assert bettor.init_cash_ == float(init_cash)


def test_bet_no_fit():
    """Test the bet method if the bettor is not fitted."""
    bettor = TestBettor()
    with pytest.raises(NotFittedError):
        bettor.bet(X_train, O_train)


def test_bet():
    """Test the bet method."""
    bettor = TestBettor()
    bettor.fit(X_train, Y_train)
    expected_value_bets = np.array(
        [
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [True, False, False],
        ],
    )
    assert np.array_equal(bettor.bet(X_train, O_train), expected_value_bets)
