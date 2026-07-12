"""Test the base Bettor class."""

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError

from sportsbet.evaluation import ClassifierBettor, complementary_events
from sportsbet.evaluation._base import BaseBettor, latest_odds_column, market_base
from tests.evaluation import O_train, TestBettor, X_train, Y_train


def test_market_base():
    """Test the market base helper drops the status/time suffix."""
    assert market_base('home_win__postplay__0min') == 'home_win'
    assert market_base('over_2.5__inplay__60min') == 'over_2.5'


def test_latest_odds_column():
    """Test the latest odds column helper picks the most recent snapshot."""
    columns = [
        'bet365__home_win__preplay__0min',
        'bet365__home_win__inplay__90min',
        'bet365__home_win__inplay__30min',
        'market_average__home_win__inplay__60min',
    ]
    assert latest_odds_column(columns, 'home_win', provider='bet365') == 'bet365__home_win__inplay__90min'
    assert latest_odds_column(columns, 'draw') is None


def test_abstract_class_raise_error():
    """Test abstract method missing implementation."""

    class IncompleteBettor(BaseBettor):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteBettor"):
        IncompleteBettor()


def test_fit_input_output_data_length_value_error():
    """Test raising an error on inconsistent data length."""
    bettor = TestBettor()
    with pytest.raises(ValueError, match='inconsistent numbers of samples'):
        bettor.fit(X_train.iloc[:-1], Y_train)


def test_bet_input_odds_data_length_value_error():
    """Test raising an error on inconsistent data length."""
    bettor = TestBettor()
    bettor.fit(X_train, Y_train)
    with pytest.raises(ValueError, match='inconsistent numbers of samples'):
        bettor.bet(X_train.iloc[:-1], O_train)


@pytest.mark.parametrize('X', [X_train.to_numpy(), pd.DataFrame(X_train.to_numpy())])
def test_fit_input_data_type_error(X):
    """Test raising an error on the wrong input data."""
    bettor = TestBettor()
    with pytest.raises(TypeError, match=re.escape('Input data `X` should be pandas dataframe with a date index.')):
        bettor.fit(X, Y_train)


@pytest.mark.parametrize('Y', [Y_train.to_numpy(), Y_train.to_numpy().tolist()])
def test_fit_output_data_type_error(Y):
    """Test raising an error on the wrong output data."""
    bettor = TestBettor()
    with pytest.raises(TypeError, match=re.escape('Output data `Y` should be pandas dataframe.')):
        bettor.fit(X_train, Y)


def test_fit_output_data_cols_names_value_error():
    """Test raising an error on output columns that break the naming grammar."""
    Y = Y_train.rename(columns={'home_win__postplay__0min': 'home_win'})
    bettor = TestBettor()
    with pytest.raises(ValueError, match='Output data column names should follow a naming'):
        bettor.fit(X_train, Y)


@pytest.mark.parametrize('O', [O_train.to_numpy(), O_train.to_numpy().tolist()])
def test_fit_odds_data_type_error(O):
    """Test raising an error on the wrong odds data."""
    bettor = TestBettor()
    bettor.fit(X_train, Y_train)
    with pytest.raises(TypeError, match=re.escape('Odds data `O` should be pandas dataframe.')):
        bettor.bet(X_train, O)


def test_fit_odds_data_cols_names_value_error():
    """Test raising an error on odds columns that break the naming grammar."""
    O = O_train.rename(columns={O_train.columns[0]: 'home_win__postplay__0min'})
    bettor = TestBettor()
    bettor.fit(X_train, Y_train)
    with pytest.raises(ValueError, match='Odds data column names should follow a naming'):
        bettor.bet(X_train, O)


def test_fit_odds_data_providers_value_error():
    """Test raising an error when odds columns mix providers."""
    renamed = O_train.columns[0].replace('market_average', 'other', 1)
    O = O_train.rename(columns={O_train.columns[0]: renamed})
    bettor = TestBettor()
    bettor.fit(X_train, Y_train)
    with pytest.raises(ValueError, match='Providers of odds data column names should be unique'):
        bettor.bet(X_train, O)


@pytest.mark.parametrize('betting_markets', ['home_win', ('home_win', 'draw')])
def test_fit_betting_markets_raise_type_error(betting_markets):
    """Test raising an error on the wrong betting markets type."""
    bettor = TestBettor(betting_markets=betting_markets)
    msg = re.escape('Parameter `betting_markets` should be a list of betting market names.')
    with pytest.raises(TypeError, match=msg):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize('betting_markets', [['not_a_market'], ['home_win', 'unknown']])
def test_fit_betting_markets_raise_value_error(betting_markets):
    """Test raising an error on the wrong betting markets value."""
    bettor = TestBettor(betting_markets=betting_markets)
    msg = re.escape('Parameter `betting_markets` does not contain valid names.')
    with pytest.raises(ValueError, match=msg):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize('init_cash', [[4.5], 'init_cash'])
def test_fit_init_cash_raise_type_error(init_cash):
    """Test raising a type error on the fit method."""
    bettor = TestBettor(init_cash=init_cash)
    msg = f'init_cash must be an instance of {{float, int}}, not {type(init_cash).__name__}.'
    with pytest.raises(TypeError, match=msg):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize('init_cash', [0.0, -30.0])
def test_fit_init_cash_raise_value_error(init_cash):
    """Test raising a value error on the fit method."""
    bettor = TestBettor(init_cash=init_cash)
    with pytest.raises(ValueError, match=f'init_cash == {init_cash}, must be > 0.0.'):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize('stake', [[20.0], 'stake'])
def test_fit_stake_raise_type_error(stake):
    """Test raising a type error on the fit method."""
    bettor = TestBettor(stake=stake)
    msg = f'stake must be an instance of {{float, int}}, not {type(stake).__name__}.'
    with pytest.raises(TypeError, match=msg):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize('stake', [0.0, -30.0])
def test_fit_stake_raise_value_error(stake):
    """Test raising a value error on the fit method."""
    bettor = TestBettor(stake=stake)
    with pytest.raises(ValueError, match=f'stake == {stake}, must be > 0.0.'):
        bettor.fit(X_train, Y_train)


def test_fit_default():
    """Test the fit method with default parameters uses the Y market bases."""
    default_init_cash, default_stake = 1e4, 50.0
    bettor = TestBettor()
    bettor.fit(X_train, Y_train)
    expected = np.array([market_base(col) for col in Y_train.columns])
    assert np.array_equal(bettor.betting_markets_, expected)
    assert bettor.init_cash_ == default_init_cash
    assert bettor.stake_ == default_stake


@pytest.mark.parametrize('betting_markets', [['draw', 'away_win'], ['home_win']])
def test_fit_betting_markets(betting_markets):
    """Test the fit method and betting markets subset."""
    bettor = TestBettor(betting_markets=betting_markets)
    bettor.fit(X_train, Y_train)
    assert np.array_equal(bettor.betting_markets_, betting_markets)
    assert len(bettor.feature_names_out_) == len(betting_markets)


def test_bet_no_fit():
    """Test the bet method if the bettor is not fitted."""
    bettor = TestBettor()
    with pytest.raises(NotFittedError):
        bettor.bet(X_train, O_train)


def test_bet_shape_and_dtype():
    """Test the bet method returns a boolean matrix aligned to the betting markets."""
    bettor = TestBettor()
    bettor.fit(X_train, Y_train)
    B = bettor.bet(X_train, O_train)
    assert B.shape == (len(X_train), bettor.betting_markets_.size)
    assert B.dtype == bool


SOCCER = ['home_win', 'draw', 'away_win', 'over_2.5', 'under_2.5']
BASKETBALL = ['home_win', 'away_win', 'over_220.5', 'under_220.5']
BOTH_GROUPS = 2


def test_complementary_events_group_the_outcome_of_a_match():
    """Test the outcome group is whichever of a home win, a draw and an away win the data carries."""
    assert ['home_win', 'draw', 'away_win'] in complementary_events(SOCCER)


def test_complementary_events_group_a_line_at_whatever_it_is():
    """Test over and under are complementary at any line, not only at the ones named in advance."""
    assert ['over_1.5', 'under_1.5'] in complementary_events(['over_1.5', 'under_1.5'])
    assert ['over_220.5', 'under_220.5'] in complementary_events(BASKETBALL)


def test_complementary_events_have_no_draw_when_the_data_has_none():
    """Test a sport that cannot be drawn has a two-way outcome, which is derived rather than declared."""
    assert ['home_win', 'away_win'] in complementary_events(BASKETBALL)
    assert len(complementary_events(BASKETBALL)) == BOTH_GROUPS


def test_a_home_win_and_an_away_win_are_not_complementary_when_a_draw_exists():
    """Test they are complementary in a sport that cannot be drawn and are not in one that can.

    Only the data knows which sport it is, which is why the groups cannot be named in advance.
    """
    assert ['home_win', 'away_win'] not in complementary_events(SOCCER)


def test_an_unknown_line_no_longer_breaks_a_bet(bettor_data):
    """Test a market outside the ones named in advance can be bet on.

    It used to be dropped from every group, so nothing was left to bet and the bet failed to be built at all.
    """
    X, Y, O = bettor_data(['over_1.5', 'under_1.5'])
    bettor = ClassifierBettor(DummyClassifier(strategy='prior')).fit(X, Y, O)
    value_bets = bettor.bet(X, O)
    assert value_bets.shape == (len(X), 2)


def test_the_probabilities_of_a_line_are_normalized(bettor_data):
    """Test the probabilities of a market outside the ones named in advance sum to one."""
    X, Y, O = bettor_data(['over_1.5', 'under_1.5'])
    bettor = ClassifierBettor(DummyClassifier(strategy='prior')).fit(X, Y, O)
    assert bettor.predict_proba(X).sum(axis=1) == pytest.approx(1.0)


@pytest.fixture
def bettor_data():
    """Build training data for a set of betting markets."""

    def build(markets, rows=4):
        index = pd.DatetimeIndex(pd.to_datetime([f'2024-01-0{row + 1}' for row in range(rows)]), name='date')
        X = pd.DataFrame({'feature': range(rows)}, index=index).astype(float)
        Y = pd.DataFrame(
            {f'{market}__postplay__0min': [row % 2 for row in range(rows)] for market in markets},
            index=index,
        )
        O = pd.DataFrame({f'market_average__{market}__postplay__0min': [1.9] * rows for market in markets}, index=index)
        return X, Y, O

    return build
