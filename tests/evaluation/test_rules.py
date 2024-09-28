"""Test the OddsComparisonBettor class."""

import numpy as np
import pytest

from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation import OddsComparisonBettor

X_train, Y_train, O_train = DummySoccerDataLoader().extract_train_data(odds_type='williamhill')


@pytest.mark.parametrize('odds_types', ['market_average', ('bet365',), ['williamhill', None]])
def test_fit_raise_type_error_odds_types(odds_types):
    """Test raising a type error on odds types."""
    bettor = OddsComparisonBettor(odds_types=odds_types)
    with pytest.raises(
        TypeError,
        match='Parameter `odds_types` should be either',
    ):
        bettor.fit(X_train, Y_train)


def test_fit_raise_value_error_no_odds():
    """Test raising a value error when odds columns are missing."""
    X = X_train[[col for col in X_train.columns if not col.startswith('odds__')]]
    bettor = OddsComparisonBettor()
    with pytest.raises(
        ValueError,
        match='Input data do not include any odds columns.',
    ):
        bettor.fit(X, Y_train)


@pytest.mark.parametrize('odds_types', [['market_average'], ['bet365']])
def test_fit_raise_value_error_odds_types(odds_types):
    """Test raising a value error on wrong odds types."""
    bettor = OddsComparisonBettor(odds_types=odds_types)
    with pytest.raises(
        ValueError,
        match='Parameter `odds_types` should be either',
    ):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize('alpha', ['alpha', None, 3])
def test_fit_raise_type_error_alpha(alpha):
    """Test raising a type error on alpha."""
    bettor = OddsComparisonBettor(alpha=alpha)
    with pytest.raises(
        TypeError,
        match='alpha must be an instance of float, not',
    ):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize('alpha', [-0.3, 1.4])
def test_backtest_raise_value_error_alpha(alpha):
    """Test raising a value error on alpha."""
    bettor = OddsComparisonBettor(alpha=alpha)
    with pytest.raises(
        ValueError,
        match=f'alpha == {alpha}, must be',
    ):
        bettor.fit(X_train, Y_train)


def test_fit_check_odds_types_default():
    """Test the checked odds types default value."""
    bettor = OddsComparisonBettor()
    bettor = bettor.fit(X_train, Y_train)
    assert bettor.odds_types_ == ['interwetten', 'williamhill']


@pytest.mark.parametrize('odds_types', [['williamhill'], ['interwetten'], ['interwetten', 'williamhill']])
def test_fit_check_odds_types(odds_types):
    """Test the checked odds types value."""
    bettor = OddsComparisonBettor(odds_types=odds_types)
    bettor.fit(X_train, Y_train)
    assert bettor.odds_types_ == odds_types


def test_bet():
    """Test the bet method."""
    assert O_train is not None
    bettor = OddsComparisonBettor(odds_types=['williamhill'])
    bettor.fit(X_train, Y_train)
    expected_value_bets = np.array(
        [
            [False, True, False],
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, True, False],
        ],
    )
    assert np.array_equal(bettor.bet(X_train, O_train), expected_value_bets)
