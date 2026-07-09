"""Dataloader for offline soccer sample data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import ClassVar, Self

import pandas as pd

from .. import ParamGrid
from ._soccer._dataloader import IDENTITY_COLS, PROVIDERS, SoccerDataLoader
from ._soccer._utils import MARKETS, market_outcomes

# One entry per sample match: identity, pre-match streaks, the full-time score, and whether
# it is an upcoming fixture. In-play snapshots are generated from the score by `_timeline`.
_MATCHES: list[dict] = [
    {
        'date': '2024-08-16',
        'league': 'England',
        'home_team': 'Man United',
        'away_team': 'Fulham',
        'home_streak': 2,
        'away_streak': -1,
        'score': (1, 0),
        'fixture': False,
    },
    {
        'date': '2024-08-24',
        'league': 'England',
        'home_team': 'Newcastle',
        'away_team': 'Tottenham',
        'home_streak': 1,
        'away_streak': 2,
        'score': (2, 1),
        'fixture': False,
    },
    {
        'date': '2024-09-01',
        'league': 'England',
        'home_team': 'Brighton',
        'away_team': 'Everton',
        'home_streak': 0,
        'away_streak': -3,
        'score': (0, 0),
        'fixture': False,
    },
    {
        'date': '2024-09-14',
        'league': 'England',
        'home_team': 'Chelsea',
        'away_team': 'West Ham',
        'home_streak': 3,
        'away_streak': 1,
        'score': (3, 1),
        'fixture': False,
    },
    {
        'date': '2024-10-05',
        'league': 'England',
        'home_team': 'Liverpool',
        'away_team': 'Crystal Palace',
        'home_streak': 4,
        'away_streak': 1,
        'score': (2, 1),
        'fixture': False,
    },
    {
        'date': '2024-11-02',
        'league': 'England',
        'home_team': 'Aston Villa',
        'away_team': 'Wolves',
        'home_streak': -1,
        'away_streak': 2,
        'score': (1, 1),
        'fixture': False,
    },
    {
        'date': '2025-01-18',
        'league': 'England',
        'home_team': 'Man City',
        'away_team': 'Brentford',
        'home_streak': 5,
        'away_streak': -2,
        'score': (4, 0),
        'fixture': False,
    },
    {
        'date': '2025-03-08',
        'league': 'England',
        'home_team': 'Nottingham',
        'away_team': 'Bournemouth',
        'home_streak': 2,
        'away_streak': 2,
        'score': (2, 2),
        'fixture': False,
    },
    {
        'date': '2025-05-25',
        'league': 'Spain',
        'home_team': 'Barcelona',
        'away_team': 'Real Madrid',
        'home_streak': 3,
        'away_streak': 3,
        'score': (3, 2),
        'fixture': False,
    },
    {
        'date': '2025-09-01',
        'league': 'England',
        'home_team': 'Arsenal',
        'away_team': 'Chelsea',
        'home_streak': 1,
        'away_streak': -2,
        'score': None,
        'fixture': True,
    },
]

# Base pre-match decimal odds per market and per-provider multiplier.
_BASE_ODDS: dict[str, float] = {
    'home_win': 1.80,
    'draw': 3.40,
    'away_win': 4.20,
    'over_2.5': 1.90,
    'under_2.5': 1.95,
    'over_3.5': 3.10,
    'under_3.5': 1.35,
}
_PROVIDER_FACTOR: dict[str, float] = {'bet365': 1.00, 'market_average': 0.98, 'market_maximum': 1.06}


def _timeline(match: dict) -> list[tuple[str, int, int, int]]:
    """Generate ``(event_status, minutes, home_goals, away_goals)`` snapshots for a match."""
    if match['fixture']:
        return [('preplay', 0, 0, 0), ('inplay', 30, 0, 0)]
    home_goals, away_goals = match['score']
    return [
        ('preplay', 0, 0, 0),
        ('inplay', 30, 0, 0),
        ('inplay', 60, (home_goals + 1) // 2, (away_goals + 1) // 2),
        ('inplay', 90, home_goals, away_goals),
        ('postplay', 0, home_goals, away_goals),
    ]


class DummySoccerDataLoader(SoccerDataLoader):
    """Dataloader for offline soccer sample data.

    The data are bundled in-play sample snapshots that require no downloading, so
    they familiarize the user with the dataloader interface and drive the
    documentation examples and doctests offline. It shares the interface of
    [`SoccerDataLoader`][sportsbet.datasets.SoccerDataLoader].

    Args:
        param_grid:
            Selects the sample data to include, mirroring scikit-learn's
            `ParameterGrid`. The default `None` selects all sample data.

    Examples:
        >>> from sportsbet.datasets import DummySoccerDataLoader
        >>> import pandas as pd
        >>> loader = DummySoccerDataLoader(param_grid={'league': ['England']})
        >>> X_train, Y_train, O_train = loader.extract_train_data(odds_type='bet365')
        >>> X_fix, Y_fix, O_fix = loader.extract_fixtures_data()
        >>> Y_fix is None
        True
        >>> pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
        >>> pd.testing.assert_index_equal(O_train.columns, O_fix.columns)
    """

    _PARAM_GRID: ClassVar[ParamGrid] = {
        'league': ['England', 'Spain'],
        'division': [1],
        'year': [2025],
    }

    @classmethod
    def _param_grid_all(cls: type[Self]) -> ParamGrid:
        return cls._PARAM_GRID

    def _snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        selected_leagues = {params['league'] for params in self._selected_params() if 'league' in params}
        stats_records: list[dict] = []
        odds_records: list[dict] = []
        for match in _MATCHES:
            if selected_leagues and match['league'] not in selected_leagues:
                continue
            identity = {
                'date': match['date'],
                'league': match['league'],
                'division': 1,
                'year': 2025,
                'home_team': match['home_team'],
                'away_team': match['away_team'],
            }
            for status, minutes, home_goals, away_goals in _timeline(match):
                markets = market_outcomes(pd.Series([home_goals]), pd.Series([away_goals])).iloc[0].to_dict()
                stats_records.append(
                    {
                        **identity,
                        'event_status': status,
                        'event_time': pd.Timedelta(minutes=minutes),
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'home_latest_streak': match['home_streak'],
                        'away_latest_streak': match['away_streak'],
                        **markets,
                    },
                )
                if status == 'postplay':
                    continue
                for provider in PROVIDERS:
                    factor = _PROVIDER_FACTOR[provider] * (1 + 0.004 * minutes)
                    odds_records.append(
                        {
                            **identity,
                            'event_status': status,
                            'event_time': pd.Timedelta(minutes=minutes),
                            'provider': provider,
                            **{market: round(base * factor, 2) for market, base in _BASE_ODDS.items()},
                        },
                    )
        stats = pd.DataFrame(stats_records)[
            [
                'event_status',
                'event_time',
                *IDENTITY_COLS,
                'home_goals',
                'away_goals',
                'home_latest_streak',
                'away_latest_streak',
                *MARKETS,
            ]
        ]
        odds = pd.DataFrame(odds_records)[['event_status', 'event_time', *IDENTITY_COLS, 'provider', *MARKETS]]
        return self._finalize(stats), self._finalize(odds)
