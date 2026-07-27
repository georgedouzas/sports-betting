"""Tests for the reconciliation of two sources."""

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
