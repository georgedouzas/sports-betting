"""
Test the data module.
"""

import numpy as np
import pytest

from sportsbet.soccer.data import (
    LEAGUES_MAPPING,
    combine_odds,
    check_leagues_ids
)


@pytest.mark.parametrize('odds,combined_odds', [
    (np.array([[2.0, 4.0], [5.0, 2.0]]), np.array([4.0 / 3.0, 10.0 / 7.0])),
    (np.array([[2.5, 2.5], [4.0, 4.0]]), np.array([2.5 / 2.0, 4.0 / 2.0]))
])
def test_combine_odds(odds, combined_odds):
    """Test the generation of combined odds."""
    np.testing.assert_array_equal(combine_odds(odds), combined_odds)


@pytest.mark.parametrize('leagues_ids', [
    ('E0', 'B1'),
    all
])
def test_check_leagues_ids_type(leagues_ids):
    """Test the check of leagues ids type."""
    with pytest.raises(TypeError):
        check_leagues_ids(leagues_ids)


@pytest.mark.parametrize('leagues_ids', [
    ['Gr', 'B1'],
    'All'
])
def test_check_leagues_ids_value(leagues_ids):
    """Test the check of leagues ids value."""
    with pytest.raises(ValueError):
        check_leagues_ids(leagues_ids)


@pytest.mark.parametrize('leagues_ids', [
    ['E0', 'B1'],
    'all'
])
def test_check_leagues_ids(leagues_ids):
    """Test the check of leagues ids."""
    checked_leagues_ids = set(check_leagues_ids(leagues_ids))
    if leagues_ids != 'all':
        assert checked_leagues_ids == set(leagues_ids)
    else:
        assert checked_leagues_ids == set(LEAGUES_MAPPING.keys())

