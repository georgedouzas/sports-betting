"""Tests for the reconciliation of two sources.

A match whose odds are missing does not look like an error. It looks like a smaller dataset and a backtest that is
confidently wrong, so these tests are mostly about it failing loudly.
"""

import pandas as pd
import pytest

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import FootballDataOdds, FootballDataStats, UnmatchedError, resolve
from sportsbet.sources._resolver import similarity

KICKOFF = pd.Timestamp('2024-08-16 19:00', tz='UTC')
TOLERATED = 0.5
MATCHES = 10

FEED = [
    'Arsenal',
    'Aston Villa',
    'Bournemouth',
    'Brentford',
    'Brighton',
    'Chelsea',
    'Crystal Palace',
    'Everton',
    'Fulham',
    'Ipswich',
    'Leicester',
    'Liverpool',
    'Man City',
    'Man United',
    'Newcastle',
    "Nott'm Forest",
    'Southampton',
    'Tottenham',
    'West Ham',
    'Wolves',
]
VENDOR = [
    'Arsenal',
    'Aston Villa',
    'AFC Bournemouth',
    'Brentford',
    'Brighton and Hove Albion',
    'Chelsea',
    'Crystal Palace',
    'Everton',
    'Fulham',
    'Ipswich Town',
    'Leicester City',
    'Liverpool',
    'Manchester City',
    'Manchester United',
    'Newcastle United',
    'Nottingham Forest',
    'Southampton',
    'Tottenham Hotspur',
    'West Ham United',
    'Wolverhampton Wanderers',
]


def _frame(pairs, odds=False):
    """Build the snapshots of the matches of a season."""
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
                **({'provider': 'pinnacle', 'home_win': 1.8} if odds else {'home_points_avg': 2.0}),
            }
            for index, (home, away) in enumerate(pairs)
        ],
    )


def _season(names):
    """Pair every club of a roster with another, exactly once."""
    return [(names[index], names[index + MATCHES]) for index in range(MATCHES)]


def test_a_roster_is_paired_without_any_alias():
    """Test the clubs of a league are paired with no help, since the two sources hold the same roster."""
    odds = _frame(_season(VENDOR), odds=True)
    resolved, report = resolve(_frame(_season(FEED)), odds)
    assert report.matched == MATCHES
    assert report.unmatched_rate == 0.0
    paired = dict(zip(odds['home_team'], resolved['home_team'], strict=True))
    paired.update(zip(odds['away_team'], resolved['away_team'], strict=True))
    assert paired == dict(zip(VENDOR, FEED, strict=True))


def test_the_two_clubs_of_a_city_are_not_swapped():
    """Test the pairing that would be fatal is the one the roster makes safe: both names are present on both sides."""
    resolved, report = resolve(
        _frame([('Man United', 'Man City')]),
        _frame([('Manchester United', 'Manchester City')], odds=True),
    )
    assert report.matched == 1
    assert resolved['home_team'].iloc[0] == 'Man United'
    assert resolved['away_team'].iloc[0] == 'Man City'


def test_a_club_is_compared_by_the_letters_an_abbreviation_keeps():
    """Test names are compared by the prefixes of their words, which is what an abbreviation preserves.

    Comparing them as strings rates `Everton` against `Liverpool` above `Wolves` against `Wolverhampton Wanderers`,
    which is the mistake that attaches one club's odds to another.
    """
    assert similarity('wolves', 'wolverhampton wanderers') > similarity('everton', 'liverpool')
    assert similarity('everton', 'liverpool') == 0.0
    assert similarity('man united', 'manchester united') > similarity('man city', 'manchester united')


def test_only_the_noise_of_a_name_is_ignored():
    """Test accents, punctuation and club words are noise, so they are taken out before the names are compared."""
    resolved, report = resolve(
        _frame([('Bayern Munich', 'Köln')]),
        _frame([('FC Bayern Munich', 'Koln')], odds=True),
    )
    assert report.matched == 1
    assert resolved['home_team'].iloc[0] == 'Bayern Munich'


def test_the_odds_carry_the_identity_of_the_statistics():
    """Test the odds take the kick-off and the spelling of the statistics, so the two line up exactly."""
    resolved, _ = resolve(_frame([('Man United', 'Chelsea')]), _frame([('Manchester United', 'Chelsea')], odds=True))
    assert resolved['date'].iloc[0] == KICKOFF
    assert resolved['home_team'].iloc[0] == 'Man United'


def test_the_wrong_club_is_never_forced_onto_the_one_left_over():
    """Test the last name left is not paired just because it is the last one.

    A vendor that carries a club the statistics do not have would otherwise have its odds attached to the wrong match.
    """
    with pytest.raises(UnmatchedError) as failure:
        resolve(_frame([('Sheffield United', 'Arsenal')]), _frame([('Sheffield Wednesday', 'Arsenal')], odds=True))
    assert failure.value.report.matched == 0


def test_a_club_the_odds_do_not_carry_fails_loudly():
    """Test a match whose odds are missing raises, rather than becoming a quietly smaller dataset."""
    with pytest.raises(UnmatchedError, match='unmatched'):
        resolve(
            _frame([('Arsenal', 'Chelsea'), ('Everton', 'Fulham')]),
            _frame([('Arsenal', 'Chelsea')], odds=True),
        )


def test_unrelated_names_are_never_paired():
    """Test two names with nothing in common are reported rather than paired."""
    with pytest.raises(UnmatchedError) as failure:
        resolve(_frame([('Everton', 'Arsenal')]), _frame([('Liverpool', 'Arsenal')], odds=True))
    assert 'Liverpool' in failure.value.report.suggestions


def test_an_abbreviation_is_paired_by_the_letters_it_keeps():
    """Test a club abbreviated by shortening its words is paired, since that is what an abbreviation does."""
    resolved, report = resolve(
        _frame([('Ath Bilbao', 'Arsenal')]),
        _frame([('Athletic Club Bilbao', 'Arsenal')], odds=True),
    )
    assert report.matched == 1
    assert resolved['home_team'].iloc[0] == 'Ath Bilbao'


def test_the_failure_offers_the_aliases_to_check_and_paste():
    """Test the failure hands back the names as they are written, so fixing it is mechanical."""
    with pytest.raises(UnmatchedError) as failure:
        resolve(_frame([('Sheffield United', 'Arsenal')]), _frame([('Sheffield Wednesday', 'Arsenal')], odds=True))
    assert "'Sheffield Wednesday'" in failure.value.report.aliases()
    assert 'aliases=' in str(failure.value)


def test_a_given_alias_bridges_what_the_pairing_leaves_over():
    """Test a name the pairing cannot place is bridged by an alias the user gives."""
    resolved, report = resolve(
        _frame([('Sheffield United', 'Arsenal')]),
        _frame([('Sheffield Wednesday', 'Arsenal')], odds=True),
        {'Sheffield Wednesday': 'Sheffield United'},
    )
    assert report.matched == 1
    assert resolved['home_team'].iloc[0] == 'Sheffield United'


def test_the_tolerance_is_a_deliberate_choice():
    """Test a rate of missing odds can be tolerated, but only on purpose."""
    _, report = resolve(
        _frame([('Arsenal', 'Chelsea'), ('Everton', 'Fulham')]),
        _frame([('Arsenal', 'Chelsea')], odds=True),
        max_unmatched_rate=TOLERATED,
    )
    assert report.matched == 1
    assert report.unmatched_rate == TOLERATED
    assert len(report.unmatched_stats) == 1


def test_the_free_feed_is_never_reconciled():
    """Test the free feed needs no reconciliation, since its statistics and its odds come from the same row."""
    stats_source, odds_source = DataLoader(stats=FootballDataStats(), odds=FootballDataOdds()).sources
    assert stats_source.name == odds_source.name


def test_a_club_from_another_competition_is_not_a_problem_to_fix():
    """Test a club that belongs to nobody here is dropped quietly rather than reported as a missing alias.

    An odds source lists more than one competition under a league, so a cup tie against a club from another division
    arrives alongside the league games. Telling the user to write an alias for it would be telling them to fix something
    that is not broken.
    """
    stats = _frame([('Arsenal', 'Chelsea'), ('Everton', 'Fulham')])
    odds = _frame([('Arsenal', 'Chelsea'), ('Everton', 'Fulham'), ('Coventry City', 'Hull City')], odds=True)
    resolved, report = resolve(stats, odds)
    assert report.matched == len(stats)
    assert report.unmatched_rate == 0.0
    assert not report.suggestions
    assert set(resolved['home_team']) == {'Arsenal', 'Everton'}
