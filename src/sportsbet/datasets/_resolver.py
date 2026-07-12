"""Reconciles the matches of one source with the matches of another."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Self

import pandas as pd

MATCH_COLS = ['league', 'division', 'year', 'home_team', 'away_team']
TEAM_COLS = ['home_team', 'away_team']
NOISE = {'fc', 'afc', 'cf', 'sc', 'ac', 'as', 'ss', 'us', 'if', 'bk', 'club', 'the'}
SUGGESTIONS = 3

ALIASES: dict[str, str] = {
    'Manchester United': 'Man United',
    'Manchester City': 'Man City',
    'Nottingham Forest': "Nott'm Forest",
    'Wolverhampton Wanderers': 'Wolves',
    'Tottenham Hotspur': 'Tottenham',
    'Brighton and Hove Albion': 'Brighton',
    'West Ham United': 'West Ham',
    'Newcastle United': 'Newcastle',
    'Leicester City': 'Leicester',
    'Ipswich Town': 'Ipswich',
    'Leeds United': 'Leeds',
    'Sheffield United': 'Sheffield United',
    'Luton Town': 'Luton',
    'Norwich City': 'Norwich',
}


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


def _identity(data: pd.DataFrame, aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """Return the match columns with the team names normalized and aliased."""
    mapping = {normalize(name): normalize(alias) for name, alias in (aliases or {}).items()}
    identity = data[MATCH_COLS].copy()
    for col in TEAM_COLS:
        normalized = identity[col].map(normalize)
        identity[col] = normalized.map(lambda name: mapping.get(name, name))
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

    A match whose odds are not found is the failure that matters. It does not look like an error, it looks like a
    smaller dataset, and it produces a backtest that is confidently wrong. So it is counted and, past the tolerance,
    raised.

    Args:
        stats:
            The long statistics snapshots, which say which matches exist.

        odds:
            The long odds snapshots.

        aliases:
            The team names of the odds source, mapped to the names of the statistics source.

        max_unmatched_rate:
            The proportion of matches that may go without odds. The default `0.0` allows none.

    Returns:
        (odds, report):
            The odds carrying the identity of the statistics, and what was reconciled.

    Raises:
        UnmatchedError:
            If more matches went without odds than the tolerance allows.
    """
    matches = stats[[*MATCH_COLS, 'date']].drop_duplicates(subset=MATCH_COLS)
    canonical = _identity(matches).assign(
        date_=matches['date'].to_numpy(),
        home_team_=matches['home_team'].to_numpy(),
        away_team_=matches['away_team'].to_numpy(),
    )
    resolved = odds.drop(columns=['date']).assign(**{col: _identity(odds, aliases)[col] for col in MATCH_COLS})
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
        suggestions=_suggestions(unmatched_odds, matches, aliases),
    )
    if report.unmatched_rate > max_unmatched_rate:
        raise UnmatchedError(report)
    return resolved, report


def _suggestions(
    unmatched_odds: pd.DataFrame,
    matches: pd.DataFrame,
    aliases: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Return, for every team name that was not found, the names it most resembles.

    A name an alias already bridges is not reported, even when its match failed for the sake of the other team: the user
    would be sent to write an alias that already exists.

    They are a starting point for an alias, never an alias. A wrong one attaches the odds of one club to another and
    says nothing about it, which is worse than not matching at all.
    """
    if unmatched_odds.empty:
        return {}
    mapping = {normalize(name): normalize(alias) for name, alias in (aliases or {}).items()}
    known = sorted({name for col in TEAM_COLS for name in matches[col]})
    normalized = {normalize(name): name for name in known}
    suggestions = {}
    for col in TEAM_COLS:
        for name in unmatched_odds[col].unique():
            aliased = mapping.get(normalize(name), normalize(name))
            if aliased in normalized:
                continue
            close = get_close_matches(aliased, list(normalized), n=SUGGESTIONS, cutoff=0.4)
            suggestions[name] = [normalized[match] for match in close]
    return suggestions
