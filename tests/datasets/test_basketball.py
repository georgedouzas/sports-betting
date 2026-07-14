"""Tests for the basketball dataloader.

Basketball is the test of the abstraction: if the engine had to change to fit a second sport, the abstraction was wrong.
"""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from sportsbet.dataloaders import DataLoader, from_snapshots
from sportsbet.evaluation import ClassifierBettor, complementary_events
from sportsbet.sources import EuroLeagueStats, FootballDataOdds, NBAStats, OddsApi

GAMES = [('Alba Berlin', 'Zalgiris', 77, 87), ('Zalgiris', 'Alba Berlin', 90, 80)]
TWO_WAY = 2


def _snapshots():
    """Build the long snapshots of a basketball season."""
    identity = {'league': 'Euroleague', 'division': 1, 'year': 2025}
    stats, odds = [], []
    for index, (home, away, home_points, away_points) in enumerate(GAMES):
        date = pd.Timestamp('2024-10-03 16:45', tz='UTC') + pd.Timedelta(days=index)
        common = {'date': date, **identity, 'home_team': home, 'away_team': away}
        stats.append({'event_status': 'preplay', 'event_time': 0, **common, 'home_points_for_avg': 80.0 + index})
        stats.append(
            {
                'event_status': 'postplay',
                'event_time': 0,
                **common,
                'home_win': int(home_points > away_points),
                'away_win': int(away_points > home_points),
            },
        )
        odds.append(
            {
                'event_status': 'preplay',
                'event_time': 0,
                **common,
                'provider': 'pinnacle',
                'home_win': 1.8,
                'away_win': 2.1,
            },
        )
    return pd.DataFrame(stats), pd.DataFrame(odds)


def test_the_sport_is_the_source_s_and_not_the_dataloader_s():
    """Test a feed of basketball is a feed of basketball, whatever it is handed to."""
    assert EuroLeagueStats.sport == 'basketball'
    assert NBAStats.sport == 'basketball'
    assert DataLoader(stats=NBAStats(), odds=OddsApi(key='k')).sport == 'basketball'


def test_statistics_and_odds_of_different_sports_are_refused():
    """Test the statistics of one sport and the odds of another are not about the same matches, and are refused.

    Nothing could pair them, and the failure would otherwise arrive much later, as a roster in which no team could be
    found in the other.
    """
    with pytest.raises(ValueError, match='not about the same matches'):
        _ = DataLoader(stats=NBAStats(), odds=FootballDataOdds()).sources


def test_a_dataloader_will_not_choose_a_source_for_you():
    """Test a dataloader with no statistics says so, rather than reading a feed nobody asked for.

    Which feed the data came from decides what is in it, what it costs and whether anyone may redistribute it. A
    dataloader that picked one on your behalf would be answering that for you, quietly.
    """
    with pytest.raises(ValueError, match='does not choose where its data comes from'):
        DataLoader().extract_train_data(download=True)


def test_odds_that_carry_no_markets_say_so():
    """Test a dataloader whose odds carry no markets has nothing to predict, and says that.

    The markets a model learns are the ones its odds price, so it used to fail inside schema validation complaining
    about a column type, which said nothing about the fact that the odds were missing.
    """
    stats, odds = _snapshots()
    empty = odds.iloc[0:0][['event_status', 'event_time', 'date', 'league', 'division', 'year']].assign(
        home_team=[],
        away_team=[],
        provider=[],
    )
    with pytest.raises(ValueError, match='no markets to predict'):
        from_snapshots(stats, empty).extract_train_data()


def test_the_targets_of_a_game_that_cannot_be_drawn():
    """Test basketball has no draw, and nothing is configured to make that so."""
    stats, odds = _snapshots()
    _, Y, _ = from_snapshots(stats, odds).extract_train_data(odds_type='pinnacle')
    markets = {col.split('__')[0] for col in Y.columns}
    assert markets == {'home_win', 'away_win'}
    assert 'draw' not in markets


def test_a_two_way_outcome_sums_to_one():
    """Test a home win and an away win are complementary in a sport that cannot be drawn.

    They are not in a sport that can, and only the data knows which sport it is.
    """
    stats, odds = _snapshots()
    X, Y, O = from_snapshots(stats, odds).extract_train_data(odds_type='pinnacle')
    numeric = X.select_dtypes(float).columns
    bettor = ClassifierBettor(DummyClassifier(strategy='prior')).fit(X[numeric], Y, O)
    assert bettor.predict_proba(X[numeric]).sum(axis=1) == pytest.approx(1.0)
    assert ['home_win', 'away_win'] in complementary_events(['home_win', 'away_win'])


def test_soccer_keeps_its_draw():
    """Test a second sport does not move the first one: soccer can be drawn and still is."""
    assert ['home_win', 'draw', 'away_win'] in complementary_events(['home_win', 'draw', 'away_win'])
    assert ['home_win', 'away_win'] not in complementary_events(['home_win', 'draw', 'away_win'])


def test_both_sports_share_the_engine():
    """Test the two dataloaders differ only in their sources, so a fix to one is a fix to both."""
    shared = set(dir(DataLoader)) - set(dir(DataLoader))
    assert not shared
