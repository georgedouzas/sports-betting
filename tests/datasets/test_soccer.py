"""Tests for the soccer dataloader (network mocked)."""

import numpy as np
import pandas as pd
import pytest

from sportsbet.datasets import SoccerDataLoader

# A small wide football-data-style frame: two historical matches and one fixture.
_RAW = pd.DataFrame(
    {
        'date': pd.to_datetime(['2024-08-16', '2024-08-17', '2025-09-01'], utc=True),
        'league': ['England', 'England', 'England'],
        'division': [1, 1, 1],
        'year': [2025, 2025, 2025],
        'home_team': ['Man United', 'Arsenal', 'Chelsea'],
        'away_team': ['Fulham', 'Liverpool', 'Everton'],
        'FTHG': [1.0, 2.0, np.nan],
        'FTAG': [0.0, 2.0, np.nan],
        'B365H': [1.70, 2.10, 1.90],
        'B365D': [3.40, 3.30, 3.50],
        'B365A': [4.20, 3.10, 3.80],
        'B365>2.5': [1.90, 1.80, 1.95],
        'B365<2.5': [1.95, 2.00, 1.90],
        'AvgH': [1.68, 2.05, 1.88],
        'AvgD': [3.35, 3.25, 3.45],
        'AvgA': [4.10, 3.05, 3.75],
        'Avg>2.5': [1.88, 1.78, 1.93],
        'Avg<2.5': [1.93, 1.98, 1.88],
        'fixtures': [False, False, True],
    },
)


@pytest.fixture
def loader(monkeypatch):
    """A SoccerDataLoader whose network download is replaced by the wide fixture frame."""
    monkeypatch.setattr(SoccerDataLoader, '_raw_data', lambda self: _RAW.copy())
    return SoccerDataLoader(param_grid={'league': ['England']})


def test_get_all_params_offline():
    """Test parameter discovery requires no download."""
    params = SoccerDataLoader.get_all_params()
    assert {'league': 'England', 'division': 1, 'year': 2018} in params
    assert all({'league', 'division', 'year'} == set(param) for param in params)


def test_get_odds_types(loader):
    """Test odds type discovery."""
    assert loader.get_odds_types() == ['bet365', 'market_average', 'market_maximum']


def test_extract_train_data_maps_feed_to_snapshots(loader):
    """Test the wide feed maps to moment-aware X, Y, O with market targets."""
    X, Y, O = loader.extract_train_data(odds_type='bet365')
    # Historical (non-fixture) matches resolve at postplay
    n_historical = int((~_RAW['fixtures']).sum())
    assert len(X) == len(Y) == len(O) == n_historical
    assert X.index.equals(Y.index)
    assert X.index.equals(O.index)
    assert 'home_win__postplay__0min' in Y.columns
    assert {col.split('__')[0] for col in O.columns} == {'bet365'}


def test_extract_train_data_odds_type_selects_provider(loader):
    """Test odds_type selects a single provider's odds."""
    _, _, O = loader.extract_train_data(odds_type='market_average')
    assert {col.split('__')[0] for col in O.columns} == {'market_average'}


def test_extract_train_data_no_odds(loader):
    """Test odds_type None returns no odds columns."""
    _, _, O = loader.extract_train_data(odds_type=None)
    assert O.shape[1] == 0


def test_extract_train_data_invalid_odds_type(loader):
    """Test an invalid odds type is rejected."""
    with pytest.raises(ValueError, match='Invalid odds type'):
        loader.extract_train_data(odds_type='williamhill')


def test_extract_train_data_inplay_target_has_no_data(loader):
    """Test an in-play target against the pre/post-only feed raises (no in-play data)."""
    with pytest.raises(ValueError, match='No resolvable events'):
        loader.extract_train_data(
            odds_type='bet365',
            target_event_status='inplay',
            target_event_time=pd.Timedelta('60min'),
        )


def test_extract_fixtures_data_matches_columns(loader):
    """Test fixtures reproduce training columns and carry the upcoming match."""
    X_train, _, O_train = loader.extract_train_data(odds_type='bet365')
    X_fix, Y_fix, O_fix = loader.extract_fixtures_data()
    assert Y_fix is None
    assert list(zip(X_fix['home_team'], X_fix['away_team'], strict=True)) == [('Chelsea', 'Everton')]
    pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
    pd.testing.assert_index_equal(O_train.columns, O_fix.columns)
