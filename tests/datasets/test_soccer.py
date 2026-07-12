"""Tests for the soccer dataloader (network mocked)."""

import pandas as pd
import pytest

from sportsbet.datasets import (
    BaseDataLoader,
    BaseOddsSource,
    BaseStatsSource,
    DummySoccerDataLoader,
    SoccerDataLoader,
    from_dataframe,
    from_snapshots,
)

PROVIDERS = ['market_average', 'market_maximum']


@pytest.fixture
def long_snapshots():
    """A long ``(stats, odds)`` sample reused from the offline dummy data."""
    return DummySoccerDataLoader(param_grid={'league': ['England']})._snapshots()


@pytest.fixture
def loader(monkeypatch, long_snapshots):
    """A SoccerDataLoader whose network download is replaced by the long sample."""
    monkeypatch.setattr(SoccerDataLoader, '_snapshots', lambda self: long_snapshots)
    return SoccerDataLoader(param_grid={'league': ['England']})


class _Available:
    """A source publishing a fixed set of combinations and needing no catalogue."""

    name = 'available'

    def index_items(self):
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


def _available_loader(param_grid=None):
    """Build a dataloader whose sources publish a fixed set of combinations."""
    return SoccerDataLoader(param_grid=param_grid, stats=_AvailableStats(), odds=_AvailableOdds())


def test_a_source_is_asked_without_a_dataloader():
    """Test what is available is asked of the source, since a param_grid cannot be written before it is known."""
    params = _AvailableStats().available_params()
    assert {'league': 'England', 'division': 2, 'year': 2024} in params
    assert {'league': 'Netherlands', 'division': 2, 'year': 2024} not in params


def test_the_dataloader_has_no_public_discovery():
    """Test discovery is not a dataloader concern, so it exposes none."""
    assert not hasattr(SoccerDataLoader, 'get_all_params')


class _FewerOdds(_AvailableOdds):
    """An odds source whose coverage starts later than the statistics."""

    def catalogue(self, payloads):
        return [{'league': 'England', 'division': 1, 'year': 2024}]


def test_selected_params_intersect_the_sources():
    """Test a season only one source publishes is never selected, since it cannot be modelled."""
    loader = SoccerDataLoader(stats=_AvailableStats(), odds=_FewerOdds())
    assert loader._selected_params() == [{'league': 'England', 'division': 1, 'year': 2024}]


def test_selected_params_filters_without_fabricating():
    """Test a partial param_grid filters the available combos instead of a cartesian product."""
    loader = _available_loader({'league': ['Netherlands']})
    assert loader._selected_params() == [{'league': 'Netherlands', 'division': 1, 'year': 2024}]


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
    monkeypatch.setattr(SoccerDataLoader, '_snapshots', lambda self: bad)
    with pytest.raises(ValueError, match='missing the required columns'):
        SoccerDataLoader().extract_train_data(odds_type='market_average')


def test_extract_fixtures_data_matches_columns(loader):
    """Test fixtures reproduce the training columns and carry the upcoming match."""
    X_train, _, O_train = loader.extract_train_data(odds_type='market_average')
    X_fix, Y_fix, O_fix = loader.extract_fixtures_data()
    assert Y_fix is None
    assert list(zip(X_fix['home_team'], X_fix['away_team'], strict=True)) == [('Arsenal', 'Chelsea')]
    pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
    pd.testing.assert_index_equal(O_train.columns, O_fix.columns)


def test_from_snapshots_consumes_long_data(long_snapshots):
    """Test canonical long snapshots are consumed without downloading."""
    stats, odds = long_snapshots
    loader = from_snapshots(stats, odds)
    assert loader.get_odds_types() == PROVIDERS
    _, Y, O = loader.extract_train_data(odds_type='market_average')
    assert 'home_win__postplay__0min' in Y.columns
    assert {col.split('__')[0] for col in O.columns} == {'market_average'}


def test_input_event_status_caps_features(long_snapshots):
    """Test the input horizon restricts the features to snapshots up to that moment."""
    stats, odds = long_snapshots
    loader = from_snapshots(stats, odds)
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


def test_from_dataframe_single_moment_splits_stats_and_odds():
    """Test a user's wide single-moment frame is split into long stats/odds at that moment."""
    wide = pd.DataFrame(
        [
            {
                'date': '2025-09-01',
                'league': 'England',
                'division': 1,
                'year': 2025,
                'home_team': 'Arsenal',
                'away_team': 'Chelsea',
                'home_points_avg': 2.0,
                'away_points_avg': 1.7,
                'market_average__home_win': 1.8,
                'market_average__draw': 3.4,
                'market_maximum__home_win': 1.9,
                'market_maximum__draw': 3.5,
            },
        ],
    )
    loader = from_dataframe(wide, event_status='preplay', event_time=pd.Timedelta('0min'))
    stats, odds = loader._snapshots()
    assert (stats['event_status'] == 'preplay').all()
    assert not [col for col in stats.columns if '__' in col]
    assert set(odds['provider']) == set(PROVIDERS)
    assert set(odds.loc[odds['provider'] == 'market_average', 'home_win']) == {1.8}
    assert loader.get_odds_types() == PROVIDERS


def test_base_dataloader_requires_snapshots():
    """Test the abstract base cannot be instantiated without a data source."""
    cls = BaseDataLoader
    with pytest.raises(TypeError, match='abstract'):
        cls()
