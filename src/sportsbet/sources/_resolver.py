"""Pair the odds of one feed to the matches of another when the two name their clubs differently."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from ..core import ALIASES, GROUPS_COLS, MATCH_COLS, TEAMS_COLS

NOISE = {'fc', 'afc', 'cf', 'sc', 'ac', 'as', 'ss', 'us', 'if', 'bk', 'club', 'the'}
MIN_PREFIX = 3
MIN_SIMILARITY = 0.6
MIN_MARGIN = 0.15


def normalize_team_name(name: str) -> str:
    """Return a team name with the differences that carry no meaning taken out."""
    text = unicodedata.normalize('NFKD', str(name))
    text = ''.join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r'[^a-z0-9 ]', '', text.lower())
    tokens = [token for token in text.split() if token not in NOISE]
    return ' '.join(tokens)


def count_common_prefix(one: str, other: str) -> int:
    """Return how many characters two strings begin with in common."""
    common = 0
    for character, candidate in zip(one, other, strict=False):
        if character != candidate:
            break
        common += 1
    return common


def measure_names_similarity(one: str, other: str) -> float:
    """Return how alike two names are, by the tokens they begin with in common."""
    tokens, others = sorted([one.split(), other.split()], key=len)
    if not tokens or not others:
        return 0.0
    total = 0.0
    for token in tokens:
        scores = [
            count_common_prefix(token, candidate) / min(len(token), len(candidate))
            for candidate in others
            if count_common_prefix(token, candidate) >= MIN_PREFIX
        ]
        total += max(scores, default=0.0)
    return total / len(tokens)


def pair_rosters(
    normalized_stats_names: set[str],
    normalized_odds_names: set[str],
) -> tuple[dict[str, str], set[str], set[str]]:
    """Return the odds names paired to the stats names, and the names left unpaired on each side."""
    matched = {name: name for name in normalized_odds_names & normalized_stats_names}
    unpaired_odds = normalized_odds_names - set(matched)
    unpaired_stats = normalized_stats_names - set(matched.values())
    candidates = []
    for name in unpaired_odds:
        scores = sorted(((measure_names_similarity(name, other), other) for other in unpaired_stats), reverse=True)
        if not scores:
            break
        best, runner_up = scores[0], (scores[1] if len(scores) > 1 else (0.0, ''))
        candidates.append((best[0], best[0] - runner_up[0], name, best[1]))
    for score, margin, name, other in sorted(candidates, reverse=True):
        if name not in unpaired_odds or other not in unpaired_stats:
            continue
        alone = len(unpaired_odds) == 1 and len(unpaired_stats) == 1
        if score >= MIN_SIMILARITY and (margin >= MIN_MARGIN or alone):
            matched[name] = other
            unpaired_odds.discard(name)
            unpaired_stats.discard(other)
    return matched, unpaired_odds, unpaired_stats


def build_roster(data: pd.DataFrame) -> dict[str, str]:
    """Return the clubs of a frame, keyed by their normalized name and valued by how they are written."""
    return {normalize_team_name(name): name for col in TEAMS_COLS for name in data[col]}


def map_odds_names(stats: pd.DataFrame, odds: pd.DataFrame, aliases: dict[str, str]) -> dict:
    """Return, per league and season, the odds names mapped to the statistics names."""
    given = {normalize_team_name(name): normalize_team_name(alias) for name, alias in aliases.items()}
    mapping: dict = {}
    for key, odds_group in odds.groupby(GROUPS_COLS):
        stats_group = stats[(stats[GROUPS_COLS] == pd.Series(key, index=GROUPS_COLS)).all(axis=1)]
        if stats_group.empty:
            continue
        stats_names = set(build_roster(stats_group))
        odds_names = {given.get(name, name) for name in build_roster(odds_group)}
        paired, _, _ = pair_rosters(stats_names, odds_names)
        mapping[key] = {**given, **paired}
    return mapping


def normalize_identity(data: pd.DataFrame, mapping: dict | None = None) -> pd.DataFrame:
    """Return the match columns with the team names normalized, and remapped when a mapping is given."""
    identity = data[MATCH_COLS].copy()
    for col in TEAMS_COLS:
        identity[col] = identity[col].map(normalize_team_name)
    if mapping is None:
        return identity
    keys = list(zip(*[data[col] for col in GROUPS_COLS], strict=True))
    for col in TEAMS_COLS:
        identity[col] = [mapping.get(key, {}).get(name, name) for key, name in zip(keys, identity[col], strict=True)]
    return identity


def resolve_odds(
    stats: pd.DataFrame,
    odds: pd.DataFrame,
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return the odds with the identity of the matches they belong to.

    Args:
        stats:
            The long statistics snapshots, which say which matches exist.

        odds:
            The long odds snapshots.

        aliases:
            The team names of the odds source, mapped to the names of the statistics source, for the clubs the pairing
            leaves over.

    Returns:
        odds:
            The odds carrying the identity of the statistics.

    Examples:
        >>> import pandas as pd
        >>> from sportsbet.sources import resolve_odds
        >>> identity = {'league': 'England', 'division': 1, 'year': 2025}
        >>> moment = {'event_status': 'preplay', 'event_time': pd.Timedelta(0)}
        >>> stats = pd.DataFrame([
        ...     {'date': pd.Timestamp('2025-08-16', tz='UTC'), **identity, **moment,
        ...      'home_team': 'Man United', 'away_team': 'Arsenal'},
        ... ])
        >>> odds = pd.DataFrame([
        ...     {'date': pd.Timestamp('2025-08-16', tz='UTC'), **identity, **moment,
        ...      'home_team': 'Manchester United', 'away_team': 'Arsenal',
        ...      'provider': 'acme', 'home_win': 1.8},
        ... ])
        >>> paired = resolve_odds(stats, odds)
        >>> # The odds carry the identity of the statistics, so `Manchester United` becomes `Man United`.
        >>> paired[['home_team', 'away_team', 'home_win']].to_dict('records')
        [{'home_team': 'Man United', 'away_team': 'Arsenal', 'home_win': 1.8}]
        >>> # A club the pairing cannot place is named, and can be given as an alias.
        >>> paired = resolve_odds(stats, odds.assign(home_team='Utd of Manchester'),
        ...                  aliases={'Utd of Manchester': 'Man United'})
        >>> len(paired)
        1
    """
    mapping = map_odds_names(stats, odds, {**ALIASES, **(aliases or {})})
    matches = stats[[*MATCH_COLS, 'date']].drop_duplicates(subset=MATCH_COLS)
    canonical = normalize_identity(matches).assign(
        date_=matches['date'].to_numpy(),
        home_team_=matches['home_team'].to_numpy(),
        away_team_=matches['away_team'].to_numpy(),
    )
    resolved = odds.drop(columns=['date']).assign(**{col: normalize_identity(odds, mapping)[col] for col in MATCH_COLS})
    resolved = resolved.merge(canonical, on=MATCH_COLS, how='left')

    found = resolved['date_'].notna()
    resolved = resolved[found.to_numpy()].copy()
    resolved['date'] = resolved.pop('date_')
    resolved['home_team'] = resolved.pop('home_team_')
    resolved['away_team'] = resolved.pop('away_team_')
    resolved = resolved[odds.columns]
    return resolved
