"""Tests for the reconciliation of two sources."""

import pandas as pd

from sportsbet.core import MATCH_COLS
from sportsbet.sources import (
    build_roster,
    count_common_prefix,
    measure_names_similarity,
    normalize_identity,
    normalize_team_name,
    pair_rosters,
    resolve_odds,
)


def test_normalize_team_name_removes_noise():
    """Test normalizing a team name takes out the differences that carry no meaning."""
    assert normalize_team_name('Manchester United') == 'manchester united'
    assert normalize_team_name('Manchester United FC') == 'manchester united'
    assert normalize_team_name('Manchester United F.C.') == 'manchester united'


def test_count_common_prefix_counts_leading_chars():
    """Test counting the common prefix takes the leading characters two strings share."""
    assert count_common_prefix('Manchester United', 'Manchester') == len('Manchester')
    assert count_common_prefix('Manchester United FC', 'Manchester United') == len('Manchester United')
    assert count_common_prefix('Manchester United F.C.', 'Man United') == len('Man')
    assert count_common_prefix('Arsenal', 'Man Utd') == 0


def test_measure_names_similarity_scores_shared_tokens():
    """Test measuring the similarity of two names scores them by the tokens they share."""
    assert measure_names_similarity('Manchester United', 'Manchester') == 1.0
    assert measure_names_similarity('Arsenal', 'Man Utd') == 0.0
    assert measure_names_similarity('Manchester United', 'Man United') == 1.0
    assert measure_names_similarity('everton', 'liverpool') == 0.0
    assert measure_names_similarity('wolves', 'wolverhampton wanderers') > measure_names_similarity(
        'everton',
        'liverpool',
    )
    assert measure_names_similarity('man united', 'manchester united') > measure_names_similarity(
        'man city',
        'manchester united',
    )


def test_build_roster_keeps_written_name(stats):
    """Test building a roster maps each normalized name back to the way it is written."""
    roster = build_roster(stats)
    assert roster['man united'] == 'Man United'


def test_pair_rosters_leaves_none_unpaired(stats, odds):
    """Test pairing rosters matches the identical names and pairs the rest by what they resemble."""
    stats_names = {normalize_team_name(name) for name in [*stats['home_team'], *stats['away_team']]}
    odds_names = {normalize_team_name(name) for name in [*odds['home_team'], *odds['away_team']]}
    matched, unpaired_odds, unpaired_stats = pair_rosters(stats_names, odds_names)
    assert len(matched) == len(stats_names) == len(odds_names)
    assert not unpaired_odds
    assert not unpaired_stats


def test_normalize_identity_normalizes_team_names(stats):
    """Test normalizing the identity carries the match columns with the team names normalized."""
    identity = normalize_identity(stats)
    assert list(identity.columns) == MATCH_COLS
    assert identity['home_team'].tolist() == [normalize_team_name(name) for name in stats['home_team']]


def test_resolve_odds_carries_stats_spelling(stats, odds):
    """Test resolving the odds carries the identity of the statistics, so the two feeds line up on one spelling."""
    resolved = resolve_odds(stats, odds)
    assert len(resolved) == len(odds)
    assert 'Manchester United' not in set(resolved['home_team'])
    assert set(resolved['home_team']) <= set(stats['home_team'])


def test_resolve_odds_drops_unpairable_club(stats, odds):
    """Test resolving the odds drops a club the pairing cannot place rather than attaching a wrong one."""
    unpairable = odds.assign(home_team=odds['home_team'].replace('Manchester United', 'Utd of Manchester'))
    resolved = resolve_odds(stats, unpairable)
    assert 'Man United' not in set(resolved['home_team'])


def test_resolve_odds_bridges_unpairable_club_with_alias(stats, odds):
    """Test resolving the odds bridges a club the pairing cannot place with an alias the user gives."""
    unpairable = odds.assign(home_team=odds['home_team'].replace('Manchester United', 'Utd of Manchester'))
    resolved = resolve_odds(stats, unpairable, {'Utd of Manchester': 'Man United'})
    assert 'Man United' in set(resolved['home_team'])


def _build_meetings(odds_dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the statistics of two meetings of one pairing in a season, and odds on the given days."""
    identity = {'league': 'Euroleague', 'division': 1, 'year': 2026, 'event_status': 'preplay', 'event_time': 0}
    teams = {'home_team': 'Olympiacos', 'away_team': 'Real Madrid'}
    kickoffs = ['2025-10-16 19:00', '2026-03-05 19:00']
    stats = pd.DataFrame([{'date': pd.Timestamp(day, tz='UTC'), **identity, **teams} for day in kickoffs])
    odds = pd.DataFrame(
        [
            {'date': pd.Timestamp(day, tz='UTC'), **identity, **teams, 'provider': 'acme', 'home_win': price}
            for price, day in enumerate(odds_dates, start=2)
        ],
    )
    return stats, odds


def test_resolve_odds_keeps_each_meeting_of_a_pairing_on_its_own_date():
    """Test a pairing that meets twice in a season gives each game its own odds."""
    stats, odds = _build_meetings(['2025-10-16 18:00', '2026-03-05 20:30'])
    resolved = resolve_odds(stats, odds)
    dates = resolved.set_index('home_win')['date']
    assert dates[2] == pd.Timestamp('2025-10-16 19:00', tz='UTC')
    assert dates[3] == pd.Timestamp('2026-03-05 19:00', tz='UTC')


def test_resolve_odds_drops_odds_far_from_any_kickoff():
    """Test odds whose pairing has no kick-off within 36 hours belong to no known game and are dropped."""
    stats, odds = _build_meetings(['2025-10-19 19:00'])
    assert resolve_odds(stats, odds).empty


def test_resolve_odds_pairs_the_clippers_with_no_alias_from_the_user():
    """Test the Odds API's `Los Angeles Clippers` joins ESPN's `LA Clippers`, which the pairing alone cannot place."""
    identity = {'league': 'NBA', 'division': 1, 'year': 2027, 'event_status': 'preplay', 'event_time': 0}
    date = pd.Timestamp('2026-11-15 03:30', tz='UTC')
    stats = pd.DataFrame([{'date': date, **identity, 'home_team': 'LA Clippers', 'away_team': 'Los Angeles Lakers'}])
    odds = stats.assign(home_team='Los Angeles Clippers', provider='acme', home_win=1.9)
    resolved = resolve_odds(stats, odds)
    assert resolved['home_team'].tolist() == ['LA Clippers']
