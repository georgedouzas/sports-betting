"""
Test the data module.
"""

import numpy as np
import pandas as pd
import pytest

from sportsbet.soccer.data import (
    LEAGUES_MAPPING,
    combine_odds,
    check_leagues_ids,
    create_spi_tables,
    create_fd_tables,
    create_names_mapping_table,
    create_modeling_tables
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


@pytest.mark.parametrize('leagues_ids', [
    ['E0', 'B1', 'E1'],
    'all'
])
def test_create_spi_tables(leagues_ids):
    """Test the creation of spi tables."""
    leagues_ids = check_leagues_ids(leagues_ids)
    spi_historical, spi_fixtures = create_spi_tables(leagues_ids)
    assert set(spi_historical.columns) == set(spi_fixtures.columns)
    assert set(spi_historical.league.unique()) == set(leagues_ids)
    assert set(spi_fixtures.league.unique()) == set(leagues_ids)
    assert spi_historical[['score1', 'score2']].isna().sum().sum() == 0
    assert spi_fixtures[['score1', 'score2']].isna().sum().sum() == 2 * len(spi_fixtures)


def test_create_fd_tables():
    """Test the creation of fd tables."""
    leagues_ids = ['E0', 'B1', 'E1']
    leagues_ids = check_leagues_ids(leagues_ids)
    fd_historical, fd_fixtures = create_fd_tables(leagues_ids)
    assert set(fd_historical.columns).difference(fd_fixtures.columns) == set(['season'])
    assert set(fd_historical.Div.unique()) == set(leagues_ids)
    assert set(fd_fixtures.Div.unique()).issubset(leagues_ids)


def test_create_names_mapping():
    """Test the creation of names mapping tables."""
    left_data = pd.DataFrame({'date': [1, 1, 2, 2], 'league': ['A', 'B', 'A', 'B'], 'team1': ['PAOK', 'AEK', 'Panathinaikos', 'Olympiakos'], 'team2': ['AEK', 'Panathinaikos', 'Olympiakos', 'PAOK']})
    right_data = pd.DataFrame({'Date': [1, 1, 2, 2], 'Div': ['A', 'B', 'A', 'B'], 'HomeTeam': ['PAOK Salonika', 'AEK Athens', 'Panathinaikos', 'Olympiakos Piraeus'], 'AwayTeam': ['AEK Athens', 'Panathinaikos', 'Olympiakos Piraeus', 'PAOK Salonika']})
    names_mapping = create_names_mapping_table(left_data, right_data)
    pd.testing.assert_frame_equal(names_mapping, pd.DataFrame({'left_team': ['AEK', 'Olympiakos', 'PAOK', 'Panathinaikos'], 'right_team': ['AEK Athens', 'Olympiakos Piraeus', 'PAOK Salonika', 'Panathinaikos']}))


def test_create_modeling_tables():
    """Test the creation of modelling tables."""
    