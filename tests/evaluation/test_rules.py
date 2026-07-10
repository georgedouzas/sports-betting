"""Test the OddsComparisonBettor class."""

import re

import numpy as np
import pytest

from sportsbet.evaluation import OddsComparisonBettor
from tests.evaluation import O_fix, O_train, X_fix, X_train, Y_train


def test_score_returns_finite():
    """Test score works when the bettor appends the full odds data to its features."""
    bettor = OddsComparisonBettor(alpha=0.03).fit(X_train, Y_train, O_train)
    assert np.isfinite(bettor.score(X_train, Y_train, O_train))


@pytest.mark.parametrize('odds_types', ['market_average', ('market_average',), ['market_average', None]])
def test_fit_raise_type_error_odds_types(odds_types):
    """Test raising a type error on odds types."""
    bettor = OddsComparisonBettor(odds_types=odds_types)
    with pytest.raises(TypeError, match='Parameter `odds_types` should be either'):
        bettor.fit(X_train, Y_train, O_train)


def test_fit_raise_value_error_no_odds():
    """Test raising a value error when odds columns are missing (no O provided)."""
    bettor = OddsComparisonBettor()
    with pytest.raises(ValueError, match=re.escape('Input data do not include any odds columns.')):
        bettor.fit(X_train, Y_train)


@pytest.mark.parametrize('odds_types', [['williamhill'], ['unknown']])
def test_fit_raise_value_error_odds_types(odds_types):
    """Test raising a value error on odds types absent from the data."""
    bettor = OddsComparisonBettor(odds_types=odds_types)
    with pytest.raises(ValueError, match='Parameter `odds_types` should be either'):
        bettor.fit(X_train, Y_train, O_train)


@pytest.mark.parametrize('alpha', ['alpha', None, 3])
def test_fit_raise_type_error_alpha(alpha):
    """Test raising a type error on alpha."""
    bettor = OddsComparisonBettor(alpha=alpha)
    with pytest.raises(TypeError, match='alpha must be an instance of float, not'):
        bettor.fit(X_train, Y_train, O_train)


@pytest.mark.parametrize('alpha', [-0.3, 1.4])
def test_fit_raise_value_error_alpha(alpha):
    """Test raising a value error on alpha."""
    bettor = OddsComparisonBettor(alpha=alpha)
    with pytest.raises(ValueError, match=f'alpha == {alpha}, must be'):
        bettor.fit(X_train, Y_train, O_train)


def test_fit_check_odds_types_default():
    """Test the default odds types resolve to the single provider present in the odds."""
    bettor = OddsComparisonBettor().fit(X_train, Y_train, O_train)
    assert bettor.odds_types_ == ['market_average']


def test_fit_check_odds_types_explicit():
    """Test an explicit valid odds type is accepted."""
    bettor = OddsComparisonBettor(odds_types=['market_average']).fit(X_train, Y_train, O_train)
    assert bettor.odds_types_ == ['market_average']


def test_bet_parses_new_grammar():
    """Test the bettor derives probabilities and value bets from the new odds grammar (CR-2)."""
    bettor = OddsComparisonBettor(alpha=0.03)
    bettor.fit(X_train, Y_train, O_train)
    proba = bettor.predict_proba(bettor._append_odds_data(X_train, O_train))
    assert proba.shape == (len(X_train), bettor.betting_markets_.size)
    B = bettor.bet(X_fix, O_fix)
    assert B.shape == (len(X_fix), bettor.betting_markets_.size)
    assert B.dtype == bool
