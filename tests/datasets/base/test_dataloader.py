"""Tests for the dataloader module."""

import pandas as pd
import pandera.pandas as pa
import pytest

from sportsbet.datasets import BaseDataLoader, required_col


class _EngineDataLoader(BaseDataLoader):
    """A concrete dataloader whose snapshots and schemas are supplied directly.

    It exercises the extraction engine of the public `BaseDataLoader` with ready
    components, so `_prepare` (which would otherwise derive the schemas) is a no-op.
    """

    def _snapshots(self):
        return self.stats, self.odds

    def _prepare(self, odds_type):
        """Skip snapshot preparation; the components are provided directly."""


def _from_components(stats, odds, stats_schema, odds_schema, targets):
    """Build an engine loader directly from ready snapshots and schemas."""
    loader = _EngineDataLoader()
    loader.stats = stats
    loader.odds = odds
    loader.stats_schema = stats_schema
    loader.odds_schema = odds_schema
    loader.targets = targets
    return loader


def test_base_dataloader_initialization(stats, odds, stats_schema, odds_schema):
    """Test BaseDataLoader initialization."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
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
    dataloader = _from_components(stats_wrong, odds, stats_schema, odds_schema, targets)
    with pytest.raises(pa.errors.SchemaError):
        dataloader.extract_train_data()


def test_extract_train_data_fails_on_invalid_odds(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on invalid odds validation."""
    targets = ['home_goals', 'away_goals']
    odds_wrong = odds.copy()
    odds_wrong.loc[0, 'event_status'] = 'halftime'
    dataloader = _from_components(stats, odds_wrong, stats_schema, odds_schema, targets)
    with pytest.raises(pa.errors.SchemaError):
        dataloader.extract_train_data()


def test_extract_train_data_fails_on_mismatched_snapshot_cols(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data when stats and odds snapshot missmatch."""
    targets = ['home_goals', 'away_goals']

    class MismatchedOddsSchema(odds_schema):
        """Odds schema with different snapshot columns."""

        extra_col: str = required_col()

    odds = odds.assign(extra_col='value')
    dataloader = _from_components(stats, odds, stats_schema, MismatchedOddsSchema, targets)
    with pytest.raises(AssertionError, match="Stats and odds snapshots columns do not match"):
        dataloader.extract_train_data()


def test_extract_train_data_fails_on_no_inplay_postplay_events(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data when no inplay or postplay events exist."""
    targets = ['home_goals', 'away_goals']
    stats_preplay_only = stats.copy()
    stats_preplay_only = stats_preplay_only.loc[stats_preplay_only['event_status'] == 'preplay']
    dataloader = _from_components(stats_preplay_only, odds, stats_schema, odds_schema, targets)
    with pytest.raises(ValueError, match='No `inplay` or `postplay` events were found'):
        dataloader.extract_train_data()


def test_extract_train_data_fails_on_invalid_learning_type(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on invalid learning type."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(ValueError, match="Invalid learning type"):
        dataloader.extract_train_data(learning_type='invalid')


def test_extract_train_data_fails_on_non_string_learning_type(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on non-string learning type."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(TypeError):
        dataloader.extract_train_data(learning_type=123)


def test_extract_train_data_fails_on_invalid_event_status(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on invalid event status."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(ValueError, match="Invalid target event status"):
        dataloader.extract_train_data(target_event_status='invalid')


def test_extract_train_data_fails_on_non_string_event_status(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on non-string event status."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(TypeError):
        dataloader.extract_train_data(target_event_status=123)


def test_extract_train_data_fails_on_negative_event_time(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on negative event time."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(ValueError, match="The event time should be positive"):
        dataloader.extract_train_data(target_event_time=pd.Timedelta('-10min'))


def test_extract_train_data_fails_on_non_timedelta_event_time(stats, odds, stats_schema, odds_schema):
    """Test extract_train_data fails on non-Timedelta event time."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(TypeError):
        dataloader.extract_train_data(target_event_time='10min')


def test_extract_train_data_returns_aligned_x_y_o(stats, odds, stats_schema, odds_schema):
    """Test supervised extraction returns aligned X, Y, O following the naming grammar (T007)."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
    X, Y, O = dataloader.extract_train_data()

    # Return types: supervised always returns a three-tuple of frames
    assert isinstance(X, pd.DataFrame)
    assert isinstance(Y, pd.DataFrame)
    assert isinstance(O, pd.DataFrame)

    # X/Y/O share the same date index and rows
    assert isinstance(X.index, pd.DatetimeIndex)
    assert X.index.equals(Y.index)
    assert X.index.equals(O.index)
    assert len(X) == len(Y) == len(O)

    # Fixed features keep bare names; time-varying features are suffixed by the grammar
    assert {'league', 'home_team', 'away_team', 'home_latest_streak'}.issubset(X.columns)
    assert 'home_goals__inplay__30min' in X.columns

    # Y targets are evaluated at the postplay target moment
    assert list(Y.columns) == ['home_goals__postplay__0min', 'away_goals__postplay__0min']

    # Odds follow the provider-prefixed grammar
    assert all(col.startswith('bet365__') for col in O.columns)


def test_extract_train_data_inplay_target_excludes_later_snapshots(stats, odds, stats_schema, odds_schema):
    """Test an in-play 60 minute target leaks no post-target information (T008)."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
    X, Y, _ = dataloader.extract_train_data(
        target_event_status='inplay',
        target_event_time=pd.Timedelta('60min'),
    )
    # Features only include snapshots strictly before 60 minutes
    assert 'home_goals__inplay__30min' in X.columns
    assert all('60min' not in col and '90min' not in col for col in X.columns)
    # Y is evaluated at the 60 minute mark
    assert list(Y.columns) == ['home_goals__inplay__60min', 'away_goals__inplay__60min']


def test_extract_train_data_unsupervised_returns_x_none_o(stats, odds, stats_schema, odds_schema):
    """Test unsupervised extraction returns a uniform (X, None, O) three-tuple (T009)."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
    X, Y, O = dataloader.extract_train_data(learning_type='unsupervised')
    assert isinstance(X, pd.DataFrame)
    assert Y is None
    assert isinstance(O, pd.DataFrame)
    assert X.index.equals(O.index)


@pytest.mark.parametrize('learning_type', ['reinforcement', 'semi-supervised', 'other'])
def test_extract_train_data_rejects_unknown_learning_type(learning_type, stats, odds, stats_schema, odds_schema):
    """Test unknown learning types (including reinforcement) are rejected (T009)."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(ValueError, match='Invalid learning type'):
        dataloader.extract_train_data(learning_type=learning_type)


def test_extract_fixtures_data_before_train_data_fails(stats, odds, stats_schema, odds_schema):
    """Test extract_fixtures_data requires a prior extract_train_data call (T012)."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
    with pytest.raises(ValueError, match='extract_train_data'):
        dataloader.extract_fixtures_data()


def test_extract_fixtures_data_matches_train_columns(stats, odds, stats_schema, odds_schema):
    """Test fixtures data reproduces the training column layout with Y None (T012)."""
    targets = ['home_goals', 'away_goals']
    dataloader = _from_components(stats, odds, stats_schema, odds_schema, targets)
    X_train, _, O_train = dataloader.extract_train_data()
    X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
    assert Y_fix is None
    pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
    pd.testing.assert_index_equal(O_train.columns, O_fix.columns)
