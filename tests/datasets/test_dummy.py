"""Tests for the offline dummy soccer dataloader."""

import pandas as pd
import pytest

from sportsbet.datasets import DummySoccerDataLoader, load_dataloader


def test_get_all_params():
    """Test the source publishes the parameters of the bundled data."""
    stats_source, *_ = DummySoccerDataLoader().sources
    params = stats_source.available_params()
    assert {'league': 'England', 'division': 1, 'year': 2025} in params
    assert {'league': 'Spain', 'division': 1, 'year': 2025} in params


def test_get_odds_types():
    """Test the discovery of odds types offline."""
    assert DummySoccerDataLoader().get_odds_types() == ['market_average', 'market_maximum']


def test_extract_train_data_grammar_and_alignment():
    """Test training extraction returns aligned, grammar-following X, Y, O."""
    loader = DummySoccerDataLoader(param_grid={'league': ['England']})
    X, Y, O = loader.extract_train_data(odds_type='market_average')
    assert isinstance(X.index, pd.DatetimeIndex)
    assert X.index.equals(Y.index)
    assert X.index.equals(O.index)
    assert 'home_win__postplay__0min' in Y.columns
    assert {col.split('__')[0] for col in O.columns} == {'market_average'}
    assert {'league', 'home_team', 'away_team'}.issubset(X.columns)


def test_extract_train_data_invalid_odds_type():
    """Test an invalid odds type is rejected."""
    loader = DummySoccerDataLoader()
    with pytest.raises(ValueError, match='Invalid odds type'):
        loader.extract_train_data(odds_type='does_not_exist')


def test_extract_train_data_unsupervised():
    """Test unsupervised extraction returns Y as None."""
    loader = DummySoccerDataLoader(param_grid={'league': ['England']})
    X, Y, O = loader.extract_train_data(odds_type='market_average', learning_type='unsupervised')
    assert Y is None
    assert X.index.equals(O.index)


def test_extract_train_data_inplay_no_leakage():
    """Test an in-play target excludes post-target snapshots."""
    loader = DummySoccerDataLoader(param_grid={'league': ['England']})
    X, Y, _ = loader.extract_train_data(
        odds_type='market_average',
        target_event_status='inplay',
        target_event_time=pd.Timedelta('60min'),
    )
    assert all('90min' not in col for col in X.columns)
    assert 'home_win__inplay__60min' in Y.columns


def test_extract_fixtures_data_matches_training_columns():
    """Test fixtures data reproduces the training columns and carries no targets."""
    loader = DummySoccerDataLoader(param_grid={'league': ['England']})
    X_train, _, O_train = loader.extract_train_data(odds_type='market_average')
    X_fix, Y_fix, O_fix = loader.extract_fixtures_data()
    assert Y_fix is None
    assert len(X_fix) >= 1  # the Arsenal vs Chelsea fixture
    pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
    pd.testing.assert_index_equal(O_train.columns, O_fix.columns)


def test_drop_na_thres_applies_to_training_and_fixtures():
    """Test drop_na_thres drops sparse columns consistently for training and fixtures."""
    loader = DummySoccerDataLoader(param_grid={'league': ['England']})
    X_train, _, _ = loader.extract_train_data(odds_type='market_average', drop_na_thres=1.0)
    X_fix, _, _ = loader.extract_fixtures_data()
    pd.testing.assert_index_equal(X_train.columns, X_fix.columns)


def test_save_reload_reproduces_fixtures_columns(tmp_path):
    """Test a saved and reloaded loader reproduces identical fixtures columns (FR-015)."""
    loader = DummySoccerDataLoader(param_grid={'league': ['England']})
    loader.extract_train_data(odds_type='market_average')
    X_fix, _, O_fix = loader.extract_fixtures_data()
    path = str(tmp_path / 'loader.pkl')
    loader.save(path)
    reloaded = load_dataloader(path)
    X_fix2, _, O_fix2 = reloaded.extract_fixtures_data()
    pd.testing.assert_index_equal(X_fix.columns, X_fix2.columns)
    pd.testing.assert_index_equal(O_fix.columns, O_fix2.columns)
