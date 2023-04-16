"""Test the RulesBettor class."""

import numpy as np
import pytest
from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation import OddsComparisonBettor

X_train, Y_train, O_train = DummySoccerDataLoader().extract_train_data(odds_type='williamhill')


@pytest.mark.parametrize('odds_types', ['market_average', ('bet365',), ['williamhill', None]])
def test_backtest_raise_type_error_odds_types(odds_types):
    """Test raising a type error on odds types."""
    with pytest.raises(
        TypeError,
        match='Parameter `odds_type` should be either',
    ):
        OddsComparisonBettor(odds_types=odds_types).backtest(X_train, Y_train, O_train)


def test_backtest_raise_value_error_no_odds():
    """Test raising a value error when odds columns are missing."""
    X = X_train[[col for col in X_train.columns if not col.startswith('odds__')]]
    with pytest.raises(
        ValueError,
        match='Input data do not include any odds columns.',
    ):
        OddsComparisonBettor().backtest(X, Y_train, O_train)


@pytest.mark.parametrize('odds_types', [['market_average'], ['bet365']])
def test_backtest_raise_value_error_odds_types(odds_types):
    """Test raising a value error on odds types."""
    with pytest.raises(
        ValueError,
        match='Parameter `odds_type` should be either',
    ):
        OddsComparisonBettor(odds_types=odds_types).backtest(X_train, Y_train, O_train)


@pytest.mark.parametrize('alpha', ['alpha', None, 3])
def test_backtest_raise_type_error_alpha(alpha):
    """Test raising a type error on alpha."""
    with pytest.raises(
        TypeError,
        match='alpha must be an instance of float, not',
    ):
        OddsComparisonBettor(alpha=alpha).backtest(X_train, Y_train, O_train)


@pytest.mark.parametrize('alpha', [-0.3, 1.4])
def test_backtest_raise_value_error_alpha(alpha):
    """Test raising a value error on alpha."""
    with pytest.raises(
        ValueError,
        match=f'alpha == {alpha}, must be',
    ):
        OddsComparisonBettor(alpha=alpha).backtest(X_train, Y_train, O_train)


def test_fit_check_odds_types_default():
    """Test the checked odds types default value."""
    bettor = OddsComparisonBettor().fit(X_train, Y_train)
    assert bettor.odds_types_ == ['interwetten', 'williamhill']


@pytest.mark.parametrize('odds_types', [['williamhill'], ['interwetten'], ['interwetten', 'williamhill']])
def test_fit_check_odds_types(odds_types):
    """Test the checked odds types value."""
    bettor = OddsComparisonBettor(odds_types=odds_types).fit(X_train, Y_train)
    assert bettor.odds_types_ == odds_types


def test_bet():
    """Test the bet method."""
    assert O_train is not None
    bettor = OddsComparisonBettor(odds_types=['williamhill']).fit(X_train, Y_train)
    np.testing.assert_array_equal(bettor.bet(X_train, O_train), np.array([False, False, False]) * (O_train > 1))
