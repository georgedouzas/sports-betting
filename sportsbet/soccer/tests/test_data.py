"""
Test the data module.
"""

import numpy as np
import pandas as pd
import pytest

from sportsbet.soccer import TARGETS
from sportsbet.soccer.data import (
    LEAGUES_MAPPING,
    combine_odds,
    check_leagues_ids,
    create_spi_tables,
    create_fd_tables,
    create_names_mapping_table,
    create_modeling_tables,
    SPI_KEYS,
    FD_KEYS,
    INPUT_COLS,
    OUTPUT_COLS
)
LEAGUES_IDS = ['E0', 'B1', 'E1']
SPI_HISTORICAL, SPI_FIXTURES = create_spi_tables(LEAGUES_IDS)
FD_HISTORICAL, FD_FIXTURES = create_fd_tables(LEAGUES_IDS)


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


def test_create_spi_tables():
    """Test the creation of spi tables."""
    assert set(SPI_HISTORICAL.columns) == set(SPI_FIXTURES.columns)
    assert set(SPI_HISTORICAL.league.unique()) == set(LEAGUES_IDS)
    assert set(SPI_FIXTURES.league.unique()) == set(LEAGUES_IDS)
    assert SPI_HISTORICAL[['score1', 'score2']].isna().sum().sum() == 0
    assert SPI_FIXTURES[['score1', 'score2']].isna().sum().sum() == 2 * len(SPI_FIXTURES)


def test_create_fd_tables():
    """Test the creation of fd tables."""
    FD_HISTORICAL, FD_FIXTURES = create_fd_tables(LEAGUES_IDS)
    assert set(FD_HISTORICAL.columns).difference(FD_FIXTURES.columns) == set(['season'])
    assert set(FD_HISTORICAL.Div.unique()) == set(LEAGUES_IDS)
    assert set(FD_FIXTURES.Div.unique()).issubset(LEAGUES_IDS)


def test_create_names_mapping():
    """Test the creation of names mapping tables."""
    left_data = pd.DataFrame({'date': [1, 1, 2, 2], 'league': ['A', 'B', 'A', 'B'], 'team1': ['PAOK', 'AEK', 'Panathinaikos', 'Olympiakos'], 'team2': ['AEK', 'Panathinaikos', 'Olympiakos', 'PAOK']})
    right_data = pd.DataFrame({'Date': [1, 1, 2, 2], 'Div': ['A', 'B', 'A', 'B'], 'HomeTeam': ['PAOK Salonika', 'AEK Athens', 'Panathinaikos', 'Olympiakos Piraeus'], 'AwayTeam': ['AEK Athens', 'Panathinaikos', 'Olympiakos Piraeus', 'PAOK Salonika']})
    names_mapping = create_names_mapping_table(left_data, right_data)
    pd.testing.assert_frame_equal(names_mapping, pd.DataFrame({'left_team': ['AEK', 'Olympiakos', 'PAOK', 'Panathinaikos'], 'right_team': ['AEK Athens', 'Olympiakos Piraeus', 'PAOK Salonika', 'Panathinaikos']}))


def test_create_modeling_tables():
    """Test the creation of modelling tables."""
    names_mapping = create_names_mapping_table(SPI_HISTORICAL[SPI_KEYS], FD_HISTORICAL[FD_KEYS])
    X, y, odds, X_test, odds_test = create_modeling_tables(SPI_HISTORICAL, SPI_FIXTURES, FD_HISTORICAL, FD_FIXTURES, names_mapping)
    assert X.columns[1:].tolist() == X_test.columns.tolist()
    assert set(X.columns) == set(['season'] + SPI_KEYS + INPUT_COLS + ['quality', 'importance', 'rating', 'sum_proj_score'])
    if X_test.size > 0:
        assert max(X.date) < min(X_test.date)
    assert odds.columns.tolist() == odds_test.columns.tolist()
    assert set(odds.columns).issuperset([target for target, *_ in TARGETS])
    assert set(y.columns) == set(OUTPUT_COLS + ['avg_score1', 'avg_score2'])