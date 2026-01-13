"""Tests for the dataloader module."""

import pandas as pd
import pandera.pandas as pa
import pytest

from sportsbet.datasets import BaseDataLoader, required_col


def test_base_dataloader_initialization(stats, odds, stats_schema, odds_schema):
    """Test BaseDataLoader initialization."""
    targets = ['home_goals', 'away_goals']
    dataloader = BaseDataLoader(stats, odds, stats_schema, odds_schema, targets)
    assert dataloader.stats is stats
    assert dataloader.odds is odds
    assert dataloader.stats_schema is stats_schema
    assert dataloader.odds_schema is odds_schema
    assert dataloader.targets is targets


def test_extract_train_data_fails_on_invalid_stats(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on invalid stats validation."""
    targets = ['home_goals', 'away_goals']
    stats_wrong = stats.copy()
    stats_wrong.loc[0, 'event_status'] = 'halftime'
    dataloader = BaseDataLoader(stats_wrong, odds, stats_schema, odds_schema, targets)
    with pytest.raises(pa.errors.SchemaError):
        dataloader.extract_train_data()


def test_extract_train_data_fails_on_invalid_odds(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on invalid odds validation."""
    targets = ['home_goals', 'away_goals']
    odds_wrong = odds.copy()
    odds_wrong.loc[0, 'event_status'] = 'halftime'
    dataloader = BaseDataLoader(stats, odds_wrong, stats_schema, odds_schema, targets)
    with pytest.raises(pa.errors.SchemaError):
        dataloader.extract_train_data()


def test_extract_train_data_fails_on_mismatched_snapshot_cols(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data when stats and odds snapshot missmatch."""
    targets = ['home_goals', 'away_goals']

    class MismatchedOddsSchema(odds_schema):
        """Odds schema with different snapshot columns."""

        extra_col: str = required_col()

    odds = odds.assign(extra_col='value')
    dataloader = BaseDataLoader(stats, odds, stats_schema, MismatchedOddsSchema, targets)
    with pytest.raises(AssertionError, match="Stats and odds snapshots columns do not match"):
        dataloader.extract_train_data()


def test_extract_train_data_fails_on_no_inplay_postplay_events(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data when no inplay or postplay events exist."""
    targets = ['home_goals', 'away_goals']
    stats_preplay_only = stats.copy()
    stats_preplay_only = stats_preplay_only.loc[stats_preplay_only['event_status'] == 'preplay']
    dataloader = BaseDataLoader(stats_preplay_only, odds, stats_schema, odds_schema, targets)
    with pytest.raises(ValueError, match='No `inplay` or `postplay` events were found'):
        dataloader.extract_train_data()


def test_extract_train_data_fails_on_invalid_learning_type(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on invalid learning type."""
    targets = ['home_goals', 'away_goals']
    dataloader = BaseDataLoader(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(ValueError, match="Invalid learning type"):
        dataloader.extract_train_data(learning_type='invalid')


def test_extract_train_data_fails_on_non_string_learning_type(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on non-string learning type."""
    targets = ['home_goals', 'away_goals']
    dataloader = BaseDataLoader(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(TypeError):
        dataloader.extract_train_data(learning_type=123)


def test_extract_train_data_fails_on_invalid_event_status(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on invalid event status."""
    targets = ['home_goals', 'away_goals']
    dataloader = BaseDataLoader(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(ValueError, match="Invalid learning type"):
        dataloader.extract_train_data(target_event_status='invalid')


def test_extract_train_data_fails_on_non_string_event_status(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on non-string event status."""
    targets = ['home_goals', 'away_goals']
    dataloader = BaseDataLoader(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(TypeError):
        dataloader.extract_train_data(target_event_status=123)


def test_extract_train_data_fails_on_negative_event_time(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on negative event time."""
    targets = ['home_goals', 'away_goals']
    dataloader = BaseDataLoader(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(ValueError, match="The event time should be positive"):
        dataloader.extract_train_data(target_event_time=pd.Timedelta('-10min'))


def test_extract_train_data_fails_on_non_timedelta_event_time(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on non-Timedelta event time."""
    targets = ['home_goals', 'away_goals']
    dataloader = BaseDataLoader(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(TypeError):
        dataloader.extract_train_data(target_event_time='10min')


def test_extract_train_data_returns_x_y_o(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data returns X, Y, O with correct structure."""
    targets = ['home_goals', 'away_goals']
    dataloader = BaseDataLoader(stats, odds, stats_schema, odds_schema, targets)
    X, Y, O = dataloader.extract_train_data()
    
    # Check return types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(Y, pd.DataFrame)
    assert O is None
    
    # Check Y contains only target columns
    assert list(Y.columns) == targets
    
    # Check X has wide format with event columns
    assert any('__' in col for col in X.columns if col not in stats_schema.snapshot_cols())
    
    # Check both have same number of rows
    assert len(X) == len(Y)
