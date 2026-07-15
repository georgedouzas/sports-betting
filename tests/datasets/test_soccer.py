"""Tests for the soccer dataloader (network mocked)."""

import pandas as pd
import pytest

from sportsbet.dataloaders import BaseDataLoader, DataLoader
from sportsbet.sources import BaseOddsSource, BaseStatsSource, FootballDataOdds, FootballDataStats
from tests.conftest import SnapshotsDataLoader

PROVIDERS = ['market_average', 'market_maximum']


@pytest.fixture
def loader(monkeypatch, long_snapshots):
    """A DataLoader whose network download is replaced by the long sample.

    The sample carries both the played matches and the upcoming one, so it stands in for both the training download and
    the fixtures download.
    """
    monkeypatch.setattr(DataLoader, '_snapshots', lambda self: long_snapshots)
    monkeypatch.setattr(DataLoader, '_fixtures_snapshots', lambda self: long_snapshots)
    return DataLoader(param_grid={'league': ['England']}, stats=FootballDataStats(), odds=FootballDataOdds())


class _Available:
    """A source publishing a fixed set of combinations and needing no catalogue."""

    name = 'available'

    def index_items(self, selection=None):
        return []

    def catalogue(self, payloads):
        return [
            {'league': 'England', 'division': 1, 'year': 2024},
            {'league': 'England', 'division': 2, 'year': 2024},
            {'league': 'Netherlands', 'division': 1, 'year': 2024},
        ]

    def required_items(self, params, schedule=None):
        return []

    def to_snapshots(self, payloads):
        return pd.DataFrame()


class _AvailableStats(_Available, BaseStatsSource):
    """The statistics of a source publishing a fixed set of combinations."""


class _AvailableOdds(_Available, BaseOddsSource):
    """The odds of a source publishing a fixed set of combinations."""


def test_a_source_is_asked_without_a_dataloader():
    """Test what is available is asked of the source, since a param_grid cannot be written before it is known."""
    params = _AvailableStats().available_params()
    assert {'league': 'England', 'division': 2, 'year': 2024} in params
    assert {'league': 'Netherlands', 'division': 2, 'year': 2024} not in params


def test_the_dataloader_has_no_public_discovery():
    """Test discovery is not a dataloader concern, so it exposes none."""
    assert not hasattr(DataLoader, 'get_all_params')


def test_get_odds_types_derived_from_data(loader):
    """Test odds types are derived from the odds ``provider`` column."""
    assert loader.get_odds_types() == PROVIDERS


def test_extract_train_data_maps_snapshots(loader):
    """Test the long snapshots map to aligned, moment-aware X, Y, O with market targets."""
    X, Y, O = loader.extract_train_data(odds_type='market_average')
    assert len(X) == len(Y) == len(O)
    assert X.index.equals(Y.index)
    assert X.index.equals(O.index)
    assert 'home_win__postplay__0min' in Y.columns
    assert {col.split('__')[0] for col in O.columns} == {'market_average'}
    assert {'home_points_avg', 'away_points_avg'}.issubset(X.columns)


def test_extract_train_data_odds_type_selects_provider(loader):
    """Test odds_type selects a single provider's odds."""
    _, _, O = loader.extract_train_data(odds_type='market_maximum')
    assert {col.split('__')[0] for col in O.columns} == {'market_maximum'}


def test_extract_train_data_no_odds(loader):
    """Test odds_type None returns no odds columns."""
    _, _, O = loader.extract_train_data(odds_type=None)
    assert O.shape[1] == 0


def test_extract_train_data_invalid_odds_type(loader):
    """Test an invalid odds type is rejected against the derived providers."""
    with pytest.raises(ValueError, match='Invalid odds type'):
        loader.extract_train_data(odds_type='bet365')


def test_extract_train_data_inplay_target(loader):
    """Test an in-play target resolves outcomes at the requested minute."""
    _, Y, _ = loader.extract_train_data(
        odds_type='market_average',
        target_event_status='inplay',
        target_event_time=pd.Timedelta('60min'),
    )
    assert 'home_win__inplay__60min' in Y.columns


def test_validate_snapshots_rejects_missing_columns(monkeypatch, long_snapshots):
    """Test the loader rejects snapshots missing required identity columns."""
    stats, odds = long_snapshots
    bad = (stats.drop(columns=['home_team']), odds)
    monkeypatch.setattr(DataLoader, '_snapshots', lambda self: bad)
    with pytest.raises(ValueError, match='missing the required columns'):
        DataLoader(stats=FootballDataStats(), odds=FootballDataOdds()).extract_train_data(odds_type='market_average')


def test_extract_fixtures_data_matches_columns(loader):
    """Test fixtures reproduce the training columns and carry the upcoming match."""
    X_train, _, O_train = loader.extract_train_data(odds_type='market_average')
    X_fix, Y_fix, O_fix = loader.extract_fixtures_data()
    assert Y_fix is None
    assert list(zip(X_fix['home_team'], X_fix['away_team'], strict=True)) == [('Arsenal', 'Chelsea')]
    pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
    pd.testing.assert_index_equal(O_train.columns, O_fix.columns)


def test_long_snapshots_are_consumed_without_downloading(long_snapshots):
    """Test a dataloader of snapshots provided directly needs no store and no network."""
    stats, odds = long_snapshots
    loader = SnapshotsDataLoader(stats, odds)
    assert loader.get_odds_types() == PROVIDERS
    _, Y, O = loader.extract_train_data(odds_type='market_average')
    assert 'home_win__postplay__0min' in Y.columns
    assert {col.split('__')[0] for col in O.columns} == {'market_average'}


def test_input_event_status_caps_features(long_snapshots):
    """Test the input horizon restricts the features to snapshots up to that moment."""
    stats, odds = long_snapshots
    loader = SnapshotsDataLoader(stats, odds)
    X_pre, *_ = loader.extract_train_data(
        odds_type='market_average',
        input_event_status='preplay',
        input_event_time=pd.Timedelta('0min'),
    )
    assert not [col for col in X_pre.columns if '__inplay__' in col]
    X_30, *_ = loader.extract_train_data(
        odds_type='market_average',
        input_event_status='inplay',
        input_event_time=pd.Timedelta('30min'),
    )
    assert any('__inplay__30min' in col for col in X_30.columns)
    assert not any('__inplay__60min' in col or '__inplay__90min' in col for col in X_30.columns)
    # The default keeps every in-play snapshot before the target.
    X_all, *_ = loader.extract_train_data(odds_type='market_average')
    assert any('__inplay__90min' in col for col in X_all.columns)


def test_base_dataloader_requires_snapshots():
    """Test the abstract base cannot be instantiated without a data source."""
    cls = BaseDataLoader
    with pytest.raises(TypeError, match='abstract'):
        cls()


def test_a_dataloader_will_not_choose_a_source_for_you():
    """Test a dataloader with no source says so, rather than reading a feed nobody asked for.

    Which feed the data came from decides what is in it, what it costs and whether anyone may redistribute it. A
    dataloader that picked one on your behalf would be answering that for you, quietly.
    """
    with pytest.raises(ValueError, match='does not choose where its data comes from'):
        DataLoader().extract_train_data()
