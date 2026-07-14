"""Reconciles the matches of one source with the matches of another."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Self

import pandas as pd

MATCH_COLS = ['league', 'division', 'year', 'home_team', 'away_team']
GROUP_COLS = ['league', 'division', 'year']
TEAM_COLS = ['home_team', 'away_team']
NOISE = {'fc', 'afc', 'cf', 'sc', 'ac', 'as', 'ss', 'us', 'if', 'bk', 'club', 'the'}
SUGGESTIONS = 3
MIN_PREFIX = 3
MIN_SIMILARITY = 0.6
MIN_MARGIN = 0.15

ALIASES: dict[str, str] = {
    'Olimpia Milano': 'EA7 Emporio Armani Milan',
}
"""The clubs the pairing cannot place, and only those.

A club is normally paired without help, because the two sources hold the same roster and a name is only ever compared
with the twenty or so clubs of its own league and season. What lands here is a club the two sources call by genuinely
different names, sharing no word between them.

`Olimpia Milano` is the historic name of the club; `EA7 Emporio Armani Milan` is its sponsor's. They have nothing in
common but the city, so nothing can pair them and nothing should try.
"""


def normalize(name: str) -> str:
    """Return a team name with the differences that carry no meaning taken out.

    Accents, punctuation, casing and the club words that every league sprinkles differently are noise. What is left is
    compared exactly: a name that still differs is a different name, and it is reported rather than guessed at.

    Args:
        name:
            The team name.

    Returns:
        normalized:
            The name without its noise.
    """
    text = unicodedata.normalize('NFKD', str(name))
    text = ''.join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r'[^a-z0-9 ]', ' ', text.lower())
    tokens = [token for token in text.split() if token not in NOISE]
    return ' '.join(tokens)


class UnmatchedError(Exception):
    """Too many matches of one source could not be found in the other.

    A match whose odds are missing is not an error that shows itself: it is a slightly smaller dataset and a backtest
    that is confidently wrong. So it is raised rather than tolerated.

    Args:
        report:
            What was reconciled and what was not.

    Examples:
        >>> import pandas as pd
        >>> from sportsbet.sources import resolve
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
        >>> from sportsbet.sources import UnmatchedError
        >>> # A club the pairing cannot place leaves its match without odds, so it is raised rather than dropped.
        >>> try:
        ...     resolve(stats, odds.assign(home_team='Real Madrid'))
        ... except UnmatchedError as error:
        ...     error.report.matched
        ...     error.report.suggestions
        0
        {'Real Madrid': ['Man United']}
        >>> # A suggestion is never applied on its own: a wrong alias attaches one club's odds to another silently.
    """

    def __init__(self: Self, report: ReconciliationReport) -> None:
        self.report = report
        super().__init__(str(report))


@dataclass
class ReconciliationReport:
    """How well the matches of one source were found in the other.

    Attributes:
        matched:
            The matches found in both sources.

        unmatched_stats:
            The matches whose odds were not found. These are the dangerous ones: they would silently become a dataset
            with holes in it.

        unmatched_odds:
            The odds whose match was not found.

        unmatched_rate:
            The proportion of matches whose odds were not found.

        suggestions:
            For every team name that was not found, the names it most resembles. A suggestion is never applied on its
            own, since a wrong alias attaches the odds of one club to another and says nothing about it.

    Examples:
        >>> import pandas as pd
        >>> from sportsbet.sources import resolve
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
        >>> _, report = resolve(stats, odds)
        >>> report.matched, report.unmatched_rate
        (1, 0.0)
        >>> report.unmatched_stats.empty
        True
    """

    matched: int = 0
    unmatched_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    unmatched_odds: pd.DataFrame = field(default_factory=pd.DataFrame)
    unmatched_rate: float = 0.0
    suggestions: dict[str, list[str]] = field(default_factory=dict)

    def aliases(self: Self) -> str:
        """Return the aliases to add, ready to be pasted and corrected.

        Returns:
            aliases:
                The suggested aliases as source code.
        """
        if not self.suggestions:
            return '{}'
        lines = [
            f'    {name!r}: {names[0]!r},' + (f'  # or {names[1:]}' if len(names) > 1 else '')
            for name, names in sorted(self.suggestions.items())
        ]
        return '{\n' + '\n'.join(lines) + '\n}'

    def __str__(self: Self) -> str:
        """Render the report."""
        total = self.matched + len(self.unmatched_stats)
        lines = [f'Matched {self.matched} of {total} matches ({self.unmatched_rate:.1%} unmatched).']
        if self.suggestions:
            lines.append(f'These team names were not found: {sorted(self.suggestions)}.')
            lines.append(f'Check them and pass them as `aliases`:\naliases={self.aliases()}')
        return ' '.join(lines[:2]) + ('\n' + lines[2] if len(lines) > 2 else '')  # noqa: PLR2004


def _prefix(one: str, other: str) -> int:
    """Return how many characters two tokens begin with in common."""
    common = 0
    for character, candidate in zip(one, other, strict=False):
        if character != candidate:
            break
        common += 1
    return common


def similarity(one: str, other: str) -> float:
    """Return how alike two names are, by the tokens they begin with in common.

    A club is abbreviated by shortening its words, not by changing their letters: `Man United` for `Manchester United`,
    `Wolves` for `Wolverhampton Wanderers`. So tokens are compared by the prefix they share, which is what an
    abbreviation preserves.

    Comparing the names as strings does not survive this. It rates `Everton` against `Liverpool` more alike than
    `Wolves` against `Wolverhampton Wanderers`, which is exactly the mistake that attaches one club's odds to another.

    Args:
        one:
            A normalized team name.

        other:
            A normalized team name.

    Returns:
        similarity:
            How alike they are, from 0 to 1.
    """
    tokens, others = sorted([one.split(), other.split()], key=len)
    if not tokens or not others:
        return 0.0
    total = 0.0
    for token in tokens:
        scores = [
            _prefix(token, candidate) / min(len(token), len(candidate))
            for candidate in others
            if _prefix(token, candidate) >= MIN_PREFIX
        ]
        total += max(scores, default=0.0)
    return total / len(tokens)


def _pair_rosters(stats_names: set[str], odds_names: set[str]) -> tuple[dict[str, str], set[str], set[str]]:
    """Pair the roster of one source with the roster of the other, for a single league and season.

    The two rosters are the same clubs, so this is a pairing of about twenty names against twenty, not a search through
    every club there is. That is what makes it safe: the names that could be confused with each other are all present,
    so they match themselves before anything is inferred, and what is left over has nothing to be confused with.

    A name is paired only when it is clearly the best of the roster and clearly better than the next best. Anything
    ambiguous is left unpaired and reported, since a wrong pairing says nothing about itself.
    """
    matched = {name: name for name in odds_names & stats_names}
    unpaired_odds = odds_names - set(matched)
    unpaired_stats = stats_names - set(matched.values())

    candidates = []
    for name in unpaired_odds:
        scores = sorted(((similarity(name, other), other) for other in unpaired_stats), reverse=True)
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


def _roster(data: pd.DataFrame) -> dict[str, str]:
    """Return the clubs of a league and season, normalized and as they are written."""
    return {normalize(name): name for col in TEAM_COLS for name in data[col]}


def _mapping(stats: pd.DataFrame, odds: pd.DataFrame, aliases: dict[str, str]) -> tuple[dict, dict[str, list[str]]]:
    """Return, per league and season, the names of the odds mapped to the names of the statistics.

    What could not be paired is reported the way it is written rather than the way it is compared, so an alias can be
    checked and pasted as it stands.

    A name is only reported when there is a club left for it to be. An odds source lists more than one competition under
    a league, so a cup tie against a club from another division arrives alongside the league games. It belongs to nobody
    here, and telling the user to write an alias for it would be telling them to fix something that is not broken.
    """
    given = {normalize(name): normalize(alias) for name, alias in aliases.items()}
    mapping: dict = {}
    unmatched: dict[str, list[str]] = {}
    for key, odds_group in odds.groupby(GROUP_COLS):
        stats_group = stats[(stats[GROUP_COLS] == pd.Series(key, index=GROUP_COLS)).all(axis=1)]
        if stats_group.empty:
            continue
        stats_roster = _roster(stats_group)
        odds_roster = {given.get(name, name): written for name, written in _roster(odds_group).items()}
        paired, unpaired_odds, unpaired_stats = _pair_rosters(set(stats_roster), set(odds_roster))
        mapping[key] = {**given, **paired}
        for name in unpaired_odds:
            close = sorted(unpaired_stats, key=lambda other: -similarity(name, other))[:SUGGESTIONS]
            if close:
                unmatched[odds_roster[name]] = [stats_roster[other] for other in close]
    return mapping, unmatched


def _identity(data: pd.DataFrame, mapping: dict | None = None) -> pd.DataFrame:
    """Return the match columns with the team names normalized, and mapped when a mapping is given."""
    identity = data[MATCH_COLS].copy()
    for col in TEAM_COLS:
        identity[col] = identity[col].map(normalize)
    if mapping is None:
        return identity
    keys = list(zip(*[data[col] for col in GROUP_COLS], strict=True))
    for col in TEAM_COLS:
        identity[col] = [mapping.get(key, {}).get(name, name) for key, name in zip(keys, identity[col], strict=True)]
    return identity


def resolve(
    stats: pd.DataFrame,
    odds: pd.DataFrame,
    aliases: dict[str, str] | None = None,
    max_unmatched_rate: float = 0.0,
) -> tuple[pd.DataFrame, ReconciliationReport]:
    """Return the odds with the identity of the matches they belong to, and how well they were reconciled.

    Two sources name the same club differently, so the odds carry the identity of the statistics rather than their own.
    The statistics say which matches exist; the odds say what they were priced at.

    The names are paired within a league and a season, where the two sources hold the same twenty clubs. That is what
    makes pairing them safe rather than a guess: every name that could be confused with another is present on both
    sides, so it matches itself before anything is inferred.

    A match whose odds are not found is the failure that matters. It does not look like an error, it looks like a
    smaller dataset, and it produces a backtest that is confidently wrong. So it is counted and, past the tolerance,
    raised.

    Args:
        stats:
            The long statistics snapshots, which say which matches exist.

        odds:
            The long odds snapshots.

        aliases:
            The team names of the odds source, mapped to the names of the statistics source, for the clubs the pairing
            leaves over.

        max_unmatched_rate:
            The proportion of matches that may go without odds. The default `0.0` allows none.

    Returns:
        (odds, report):
            The odds carrying the identity of the statistics, and what was reconciled.

    Raises:
        UnmatchedError:
            If more matches went without odds than the tolerance allows.

    Examples:
        >>> import pandas as pd
        >>> from sportsbet.sources import resolve
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
        >>> paired, report = resolve(stats, odds)
        >>> # The odds carry the identity of the statistics, so `Manchester United` becomes `Man United`.
        >>> paired[['home_team', 'away_team', 'home_win']].to_dict('records')
        [{'home_team': 'Man United', 'away_team': 'Arsenal', 'home_win': 1.8}]
        >>> report.matched, report.unmatched_rate
        (1, 0.0)
        >>> # A club the pairing cannot place is named, and can be given as an alias.
        >>> paired, report = resolve(stats, odds.assign(home_team='Utd of Manchester'),
        ...                          aliases={'Utd of Manchester': 'Man United'})
        >>> report.matched
        1
    """
    mapping, suggestions = _mapping(stats, odds, {**ALIASES, **(aliases or {})})
    matches = stats[[*MATCH_COLS, 'date']].drop_duplicates(subset=MATCH_COLS)
    canonical = _identity(matches).assign(
        date_=matches['date'].to_numpy(),
        home_team_=matches['home_team'].to_numpy(),
        away_team_=matches['away_team'].to_numpy(),
    )
    resolved = odds.drop(columns=['date']).assign(**{col: _identity(odds, mapping)[col] for col in MATCH_COLS})
    resolved = resolved.merge(canonical, on=MATCH_COLS, how='left')

    found = resolved['date_'].notna()
    unmatched_odds = odds[~found.to_numpy()]
    resolved = resolved[found.to_numpy()].copy()
    resolved['date'] = resolved.pop('date_')
    resolved['home_team'] = resolved.pop('home_team_')
    resolved['away_team'] = resolved.pop('away_team_')
    resolved = resolved[odds.columns]

    priced = set(map(tuple, resolved[MATCH_COLS].drop_duplicates().to_numpy()))
    unmatched_stats = matches[[tuple(row) not in priced for row in matches[MATCH_COLS].to_numpy()]]
    total = len(matches)
    report = ReconciliationReport(
        matched=total - len(unmatched_stats),
        unmatched_stats=unmatched_stats,
        unmatched_odds=unmatched_odds,
        unmatched_rate=len(unmatched_stats) / total if total else 0.0,
        suggestions=suggestions,
    )
    if report.unmatched_rate > max_unmatched_rate:
        raise UnmatchedError(report)
    return resolved, report
