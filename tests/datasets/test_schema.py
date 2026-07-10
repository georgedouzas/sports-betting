"""Tests for schema validation."""

import pandas as pd
import pandera.pandas as pa
import pytest


def test_stastics_schema(stats, stats_schema):
    """Test statistics schema."""
    validated_stats = stats_schema.validate(stats)
    pd.testing.assert_frame_equal(stats, validated_stats)


def test_stats_schema_fails_on_unknown_event_status(stats, stats_schema):
    """Test statistics schema fails on unknown event status."""
    stats_wrong = stats.copy()
    stats_wrong.loc[0, 'event_status'] = 'halftime'
    with pytest.raises(pa.errors.SchemaError):
        stats_schema.validate(stats_wrong)


def test_stats_schema_fails_on_inplay_zero_event_time(stats, stats_schema):
    """Test statistics schema fails on inplay with zero event time."""
    stats_wrong = stats.copy()
    inplay_idx = stats_wrong.index[stats_wrong['event_status'] == 'inplay']
    idx = int(inplay_idx[0]) if len(inplay_idx) else 0
    stats_wrong.loc[idx, 'event_status'] = 'inplay'
    stats_wrong.loc[idx, 'event_time'] = pd.Timedelta(0)
    with pytest.raises(pa.errors.SchemaError):
        stats_schema.validate(stats_wrong)


def test_stats_schema_fails_on_postplay_nonzero_event_time(stats, stats_schema):
    """Test statistics schema fails on postplay with non-zero event time."""
    stats_wrong = stats.copy()
    post_idx = stats_wrong.index[stats_wrong['event_status'] == 'postplay']
    idx = int(post_idx[0]) if len(post_idx) else 0
    stats_wrong.loc[idx, 'event_status'] = 'postplay'
    stats_wrong.loc[idx, "event_time"] = pd.Timedelta(minutes=10)
    with pytest.raises(pa.errors.SchemaError):
        stats_schema.validate(stats_wrong)


def test_stats_schema_fails_on_duplicate_snapshot_key(stats, stats_schema):
    """Test statistics schema fails on duplicate snapshot key."""
    stats_wrong = stats.copy()
    bad = pd.concat([stats_wrong, stats_wrong.iloc[[0]]], ignore_index=True)
    with pytest.raises(pa.errors.SchemaError):
        stats_schema.validate(bad)


def test_stats_schema_fails_on_extra_column_due_to_strict(stats, stats_schema):
    """Test statistics schema fails on extra column due to strict mode."""
    stats_wrong = stats.copy()
    stats_wrong['unexpected'] = 123
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        stats_schema.validate(stats_wrong)


def test_stats_schema_fails_on_missing_required_column(stats, stats_schema):
    """Test statistics schema fails on missing required column."""
    stats_wrong = stats.copy().drop(columns=['home_team'])
    with pytest.raises(pa.errors.SchemaError):
        stats_schema.validate(stats_wrong)


def test_odds_schema(odds, odds_schema):
    """Test odds schema."""
    validated_odds = odds_schema.validate(odds)
    pd.testing.assert_frame_equal(odds, validated_odds)


def test_odds_schema_fails_on_unknown_event_status(odds, odds_schema):
    """Test statistics schema fails on unknown event status."""
    odds_wrong = odds.copy()
    odds_wrong.loc[0, 'event_status'] = 'halftime'
    with pytest.raises(pa.errors.SchemaError):
        odds_schema.validate(odds_wrong)


def test_odds_schema_fails_on_inplay_zero_event_time(odds, odds_schema):
    """Test statistics schema fails on inplay with zero event time."""
    odds_wrong = odds.copy()
    inplay_idx = odds_wrong.index[odds_wrong['event_status'] == 'inplay']
    idx = int(inplay_idx[0]) if len(inplay_idx) else 0
    odds_wrong.loc[idx, 'event_status'] = 'inplay'
    odds_wrong.loc[idx, 'event_time'] = pd.Timedelta(0)
    with pytest.raises(pa.errors.SchemaError):
        odds_schema.validate(odds_wrong)


def test_odds_schema_fails_on_postplay_nonzero_event_time(odds, odds_schema):
    """Test statistics schema fails on postplay with non-zero event time."""
    odds_wrong = odds.copy()
    post_idx = odds_wrong.index[odds_wrong['event_status'] == 'postplay']
    idx = int(post_idx[0]) if len(post_idx) else 0
    odds_wrong.loc[idx, 'event_status'] = 'postplay'
    odds_wrong.loc[idx, "event_time"] = pd.Timedelta(minutes=10)
    with pytest.raises(pa.errors.SchemaError):
        odds_schema.validate(odds_wrong)


def test_odds_schema_fails_on_duplicate_snapshot_key(odds, odds_schema):
    """Test statistics schema fails on duplicate snapshot key."""
    odds_wrong = odds.copy()
    bad = pd.concat([odds_wrong, odds_wrong.iloc[[0]]], ignore_index=True)
    with pytest.raises(pa.errors.SchemaError):
        odds_schema.validate(bad)


def test_odds_schema_fails_on_extra_column_due_to_strict(odds, odds_schema):
    """Test statistics schema fails on extra column due to strict mode."""
    odds_wrong = odds.copy()
    odds_wrong['unexpected'] = 123
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        odds_schema.validate(odds_wrong)


def test_odds_schema_fails_on_missing_required_column(odds, odds_schema):
    """Test statistics schema fails on missing required column."""
    odds_wrong = odds.copy().drop(columns=['home_team'])
    with pytest.raises(pa.errors.SchemaError):
        odds_schema.validate(odds_wrong)


def test_odds_schema_fails_on_postplay_odds_not_nan(odds, odds_schema):
    """Test odds schema fails on postplay odds not NaN."""
    odds_wrong = odds.copy()
    post_idx = odds_wrong.index[odds_wrong['event_status'] == 'postplay']
    idx = post_idx[0]
    for col in odds_wrong.columns:
        if col not in ('event_status', 'event_time', 'provider') and col not in odds_schema.snapshot_cols():
            odds_wrong.loc[idx, col] = 1.23
    with pytest.raises(pa.errors.SchemaError):
        odds_schema.validate(odds_wrong)
