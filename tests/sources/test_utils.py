"""Tests for the utils module."""

import re

import pandas as pd
import pytest

from sportsbet.sources import derive_market_outcomes


def test_derive_market_outcomes():
    """Derive an outcome column per market."""
    home_points = [2, 1, 0]
    away_points = [1, 1, 3]
    markets = ['home_win', 'draw', 'away_win', 'over_2.5']
    expected_market_outcomes = pd.DataFrame.from_dict(
        {
            'home_win': [1, 0, 0],
            'draw': [0, 1, 0],
            'away_win': [0, 0, 1],
            'over_2.5': [1, 0, 1],
        },
    )
    market_outcomes = derive_market_outcomes(home_points, away_points, markets)
    assert market_outcomes.equals(expected_market_outcomes)


def test_derive_market_outcomes_no_draw():
    """Derive two-way outcomes for a sport with no draw."""
    home_points = [100, 95, 78]
    away_points = [90, 97, 80]
    markets = ['home_win', 'away_win', 'over_188']
    expected_market_outcomes = pd.DataFrame.from_dict(
        {
            'home_win': [1, 0, 0],
            'away_win': [0, 1, 1],
            'over_188': [1, 1, 0],
        },
    )
    market_outcomes = derive_market_outcomes(home_points, away_points, markets)
    assert market_outcomes.equals(expected_market_outcomes)


def test_derive_market_outcomes_different_length():
    """Raise when home and away points differ in length."""
    home_points = [2, 1]
    away_points = [1, 1, 3]
    markets = ['home_win', 'draw', 'away_win', 'over_2.5']
    with pytest.raises(
        ValueError,
        match=re.escape('Found input variables with inconsistent numbers of samples: [2, 3]'),
    ):
        derive_market_outcomes(home_points, away_points, markets)


def test_derive_market_outcomes_wrong_types():
    """Raise when points cannot be cast to integers."""
    home_points = [1, 2, 1]
    away_points = ['point', '1', '3']
    markets = ['home_win', 'draw', 'away_win', 'over_2.5']
    with pytest.raises(ValueError, match=re.escape('invalid literal for int() with base 10: np.str_(\'point\')')):
        derive_market_outcomes(home_points, away_points, markets)
