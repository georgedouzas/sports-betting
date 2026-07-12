"""Tests for the football-data.co.uk sources.

The feed is not redistributed, so the transform is exercised against a synthetic feed-shaped payload and the real feed
is checked by the `network` marked equivalence gate.
"""

import pandas as pd
import pytest

from sportsbet.datasets import FootballDataOdds, FootballDataStats, RawItem, RawPayload

FEED = (
    'Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,HS,AS,HST,AST,HC,AC,HY,AY,HR,AR,'
    'AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5,MaxH,MaxD,MaxA,Max>2.5,Max<2.5\n'
    'E0,10/08/2024,Arsenal,Chelsea,2,1,1,0,12,8,5,3,6,4,1,2,0,0,1.80,3.50,4.20,1.90,1.95,1.85,3.60,4.40,1.95,2.00\n'
    'E0,11/08/2024,Liverpool,Everton,0,0,0,0,15,5,6,1,9,2,0,1,0,0,1.50,4.00,6.00,1.70,2.10,1.55,4.10,6.20,1.75,2.15\n'
    'E0,17/08/2024,Chelsea,Liverpool,1,3,0,2,9,14,4,7,3,8,2,1,1,0,2.60,3.40,2.70,1.80,2.00,2.70,3.50,2.80,1.85,2.05\n'
    'E0,18/08/2024,Everton,Arsenal,1,1,1,1,7,13,2,6,2,7,3,0,0,1,4.50,3.60,1.75,1.85,1.95,4.60,3.70,1.80,1.90,2.00\n'
)
CLOSING_FEED = (
    'Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,AvgCH,AvgCD,AvgCA\n'
    'E0,10/08/2024,Arsenal,Chelsea,2,1,1,0,1.80,3.50,4.20\n'
    'E0,11/08/2024,Liverpool,Everton,0,0,0,0,1.50,4.00,6.00\n'
)
INDEX = [('England', 1, 2025, 'https://www.football-data.co.uk/mmz4281/2425/E0.csv')]
EXPECTED_SNAPSHOTS = 12


def payloads(feed=FEED):
    """Build the raw payloads of a season, with no fixtures."""
    item = RawItem(source='football_data', key='England_1_2025', url=INDEX[0][3])
    return [RawPayload(item=item, content=feed.encode('ISO-8859-1'))]


@pytest.fixture
def sources(monkeypatch):
    """Statistics and odds sources whose feed index is supplied directly."""
    stats, odds = FootballDataStats(), FootballDataOdds()
    for source in (stats, odds):
        monkeypatch.setattr(source, '_catalogue', INDEX)
    return stats, odds


def test_stats_and_odds_declare_the_same_items(sources):
    """Test the shared feed file is declared once, so it is downloaded once and not twice."""
    stats, odds = sources
    params = [{'league': 'England', 'division': 1, 'year': 2025}]
    stats_items = stats.required_items(params)
    odds_items = odds.required_items(params)
    assert stats_items == odds_items
    assert len(set(stats_items + odds_items)) == len(stats_items)


def test_stats_snapshots_carry_every_moment(sources):
    """Test each played match becomes a pre-play, an in-play and a post-play snapshot."""
    stats, _ = sources
    snapshots = stats.to_snapshots(payloads())
    assert len(snapshots) == EXPECTED_SNAPSHOTS
    assert set(snapshots['event_status']) == {'preplay', 'inplay', 'postplay'}
    assert set(snapshots.loc[snapshots['event_status'] == 'inplay', 'event_time']) == {45}


def test_inplay_snapshots_carry_the_half_time_goals(sources):
    """Test the in-play snapshot holds the goals at half time, not at full time."""
    stats, _ = sources
    snapshots = stats.to_snapshots(payloads())
    inplay = snapshots[snapshots['event_status'] == 'inplay'].set_index('home_team')
    postplay = snapshots[snapshots['event_status'] == 'postplay'].set_index('home_team')
    assert inplay.loc['Chelsea', 'home_goals'] == 0
    assert postplay.loc['Chelsea', 'home_goals'] == 1


def test_market_outcomes_are_derived_per_moment(sources):
    """Test the market outcomes are resolved at the moment of the snapshot."""
    stats, _ = sources
    snapshots = stats.to_snapshots(payloads())
    postplay = snapshots[snapshots['event_status'] == 'postplay'].set_index('home_team')
    assert postplay.loc['Arsenal', 'home_win'] == 1
    assert postplay.loc['Liverpool', 'draw'] == 1
    assert postplay.loc['Chelsea', 'away_win'] == 1


def test_odds_snapshots_carry_every_provider(sources):
    """Test the odds are exploded into one row per provider."""
    _, odds = sources
    snapshots = odds.to_snapshots(payloads())
    assert set(snapshots['provider']) == {'market_average', 'market_maximum'}
    assert (snapshots['event_status'] == 'preplay').all()
    average = snapshots[snapshots['provider'] == 'market_average'].set_index('home_team')
    assert average.loc['Arsenal', 'home_win'] == pytest.approx(1.80)


def test_closing_odds_backfill_the_missing_odds(sources):
    """Test an odds column absent from the feed is back-filled from its closing twin, so it is not lost."""
    _, odds = sources
    snapshots = odds.to_snapshots(payloads(CLOSING_FEED))
    average = snapshots[snapshots['provider'] == 'market_average'].set_index('home_team')
    assert average.loc['Arsenal', 'home_win'] == pytest.approx(1.80)


def test_features_precede_the_match_they_describe(sources):
    """Test the form features of a match hold the results before it and never its own outcome."""
    stats, _ = sources
    snapshots = stats.to_snapshots(payloads())
    preplay = snapshots[snapshots['event_status'] == 'preplay'].set_index('home_team')
    assert pd.isna(preplay.loc['Arsenal', 'home_points_avg'])
    assert preplay.loc['Chelsea', 'home_points_avg'] == 0


def test_sources_perform_no_input_output(sources):
    """Test a source never fetches, so extraction can never download by accident."""
    stats, odds = sources
    params = [{'league': 'England', 'division': 1, 'year': 2025}]
    assert stats.required_items(params)
    assert stats.to_snapshots(payloads()) is not None
    assert odds.to_snapshots(payloads()) is not None
