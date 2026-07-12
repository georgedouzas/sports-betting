"""Tests for the reconciliation of two sources.

A match whose odds are missing does not look like an error. It looks like a smaller dataset and a backtest that is
confidently wrong, so these tests are mostly about it failing loudly.
"""

import pandas as pd
import pytest

from sportsbet.datasets import ReconciliationReport, SoccerDataLoader, UnmatchedError, resolve
from sportsbet.datasets._resolver import ALIASES

KICKOFF = pd.Timestamp('2024-08-16 19:00', tz='UTC')
TOLERATED = 0.5


def _stats(*teams):
    """Build the statistics of the matches that exist."""
    return pd.DataFrame(
        [
            {
                'event_status': 'preplay',
                'event_time': pd.Timedelta(0),
                'date': KICKOFF + pd.Timedelta(days=index),
                'league': 'England',
                'division': 1,
                'year': 2025,
                'home_team': home,
                'away_team': away,
                'home_points_avg': 2.0,
            }
            for index, (home, away) in enumerate(teams)
        ],
    )


def _odds(*teams):
    """Build the odds of the matches a vendor priced."""
    return pd.DataFrame(
        [
            {
                'event_status': 'preplay',
                'event_time': pd.Timedelta(0),
                'date': KICKOFF + pd.Timedelta(days=index, minutes=3),
                'league': 'England',
                'division': 1,
                'year': 2025,
                'home_team': home,
                'away_team': away,
                'provider': 'pinnacle',
                'home_win': 1.8,
            }
            for index, (home, away) in enumerate(teams)
        ],
    )


def _resolve(stats, odds, aliases=None, max_unmatched_rate=0.0):
    """Reconcile the two, with the aliases the library knows."""
    return resolve(stats, odds, {**ALIASES, **(aliases or {})}, max_unmatched_rate)


def test_a_club_named_the_same_is_matched():
    """Test the names that already agree need no help."""
    stats = _stats(('Arsenal', 'Chelsea'))
    resolved, report = _resolve(stats, _odds(('Arsenal', 'Chelsea')))
    assert report.matched == 1
    assert report.unmatched_rate == 0.0
    assert list(resolved['home_team']) == ['Arsenal']


def test_a_club_named_differently_is_matched_by_a_known_alias():
    """Test the aliases the library knows bridge the names the two sources give the same club."""
    stats = _stats(('Man United', 'Nott\'m Forest'))
    resolved, report = _resolve(stats, _odds(('Manchester United', 'Nottingham Forest')))
    assert report.matched == 1
    assert list(resolved['home_team']) == ['Man United']
    assert list(resolved['away_team']) == ["Nott'm Forest"]


def test_the_odds_carry_the_identity_of_the_statistics():
    """Test the odds take the kick-off of the statistics, so the two align rather than nearly align."""
    stats = _stats(('Arsenal', 'Chelsea'))
    resolved, _ = _resolve(stats, _odds(('Arsenal', 'Chelsea')))
    assert resolved['date'].iloc[0] == KICKOFF


def test_only_the_noise_of_a_name_is_ignored():
    """Test accents, punctuation and club words are noise, so they are taken out before the names are compared."""
    stats = _stats(('Bayern Munich', 'Köln'))
    resolved, report = _resolve(stats, _odds(('FC Bayern Munich', 'Koln')))
    assert report.matched == 1
    assert list(resolved['home_team']) == ['Bayern Munich']


def test_an_unmatched_club_fails_loudly():
    """Test a match whose odds are missing raises, rather than becoming a quietly smaller dataset."""
    stats = _stats(('Arsenal', 'Chelsea'))
    with pytest.raises(UnmatchedError, match='unmatched'):
        _resolve(stats, _odds(('Arsenal FC Woolwich', 'Chelsea')))


def test_the_failure_says_which_names_it_could_not_place():
    """Test the failure names what it could not place, rather than leaving the user to guess."""
    stats = _stats(('Arsenal', 'Chelsea'))
    with pytest.raises(UnmatchedError) as failure:
        _resolve(stats, _odds(('Woolwich Arsenal Reserves', 'Chelsea')))
    assert 'Woolwich Arsenal Reserves' in failure.value.report.suggestions


def test_the_failure_offers_the_aliases_to_add():
    """Test the failure hands back the aliases to check and paste, so fixing it is mechanical."""
    stats = _stats(('Arsenal', 'Chelsea'))
    with pytest.raises(UnmatchedError) as failure:
        _resolve(stats, _odds(('Woolwich Arsenal Reserves', 'Chelsea')))
    assert 'Arsenal' in failure.value.report.aliases()
    assert 'aliases=' in str(failure.value)


def test_a_suggestion_is_never_applied_on_its_own():
    """Test a name that merely resembles another is not matched to it.

    A wrong alias attaches the odds of one club to another and says nothing about it, which is worse than not matching.
    """
    stats = _stats(('Manchester City', 'Chelsea'))
    with pytest.raises(UnmatchedError) as failure:
        _resolve(stats, _odds(('Manchester United', 'Chelsea')))
    assert failure.value.report.matched == 0


def test_a_given_alias_matches_what_the_library_does_not_know():
    """Test a name the library does not know is bridged by an alias the user gives."""
    stats = _stats(('Ath Bilbao', 'Chelsea'))
    resolved, report = _resolve(stats, _odds(('Athletic Bilbao', 'Chelsea')), {'Athletic Bilbao': 'Ath Bilbao'})
    assert report.matched == 1
    assert list(resolved['home_team']) == ['Ath Bilbao']


def test_the_tolerance_is_a_deliberate_choice():
    """Test a rate of missing odds can be tolerated, but only on purpose."""
    stats = _stats(('Arsenal', 'Chelsea'), ('Everton', 'Spurs'))
    _, report = _resolve(stats, _odds(('Arsenal', 'Chelsea')), max_unmatched_rate=TOLERATED)
    assert report.matched == 1
    assert report.unmatched_rate == TOLERATED
    assert len(report.unmatched_stats) == 1


def test_the_report_is_kept_on_the_dataloader(monkeypatch):
    """Test the reconciliation is inspectable after an extraction."""
    stats = _stats(('Man United', 'Chelsea'))
    odds = _odds(('Manchester United', 'Chelsea'))
    loader = SoccerDataLoader()
    monkeypatch.setattr(SoccerDataLoader, '_snapshots', lambda self: (stats, odds))
    assert isinstance(getattr(loader, 'reconciliation_', ReconciliationReport()), ReconciliationReport)


def test_the_same_source_is_never_reconciled():
    """Test the free feed needs no reconciliation, since its statistics and its odds come from the same row."""
    loader = SoccerDataLoader()
    stats_source, odds_source = loader.sources
    assert stats_source.name == odds_source.name
