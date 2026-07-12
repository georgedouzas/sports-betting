"""Dataloader for offline soccer sample data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import ClassVar, Self

import pandas as pd
from sklearn.model_selection import ParameterGrid

from ... import ParamGrid
from .._base._dataloader import BaseDataLoader
from .._base._schema import IDENTITY_COLS
from .._sources._base import BaseStatsSource, RawItem, RawPayload
from .._utils import market_outcomes

_PARAM_GRID: ParamGrid = {'league': ['England', 'Spain'], 'division': [1], 'year': [2025]}


class _DummySoccerStats(BaseStatsSource):
    """Publishes the parameters of the bundled sample data.

    The data is bundled rather than downloaded, so there is nothing to fetch and nothing to transform.
    """

    name: ClassVar[str] = 'dummy_soccer'

    def index_items(self: Self) -> list[RawItem]:
        """Return no items, since nothing is downloaded."""
        return []

    def catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the parameters of the bundled data."""
        return list(ParameterGrid(_PARAM_GRID))

    def required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return no items, since nothing is downloaded."""
        return []

    def to_snapshots(self: Self, payloads: list[RawPayload]) -> pd.DataFrame:
        """Never called: the dataloader carries the bundled snapshots itself."""
        msg = 'The bundled sample data is not built from payloads.'
        raise NotImplementedError(msg)


_MARKETS = ['home_win', 'draw', 'away_win', 'over_2.5', 'under_2.5']
_PROVIDERS = ['market_average', 'market_maximum']
_FEATURES = ['home_points_avg', 'away_points_avg']
_MATCHES: list[dict] = [
    {
        'date': '2024-08-16',
        'league': 'England',
        'home': 'Man United',
        'away': 'Fulham',
        'home_points_avg': 2.1,
        'away_points_avg': 1.2,
        'score': (1, 0),
        'fixture': False,
    },
    {
        'date': '2024-08-24',
        'league': 'England',
        'home': 'Newcastle',
        'away': 'Tottenham',
        'home_points_avg': 1.5,
        'away_points_avg': 1.8,
        'score': (1, 2),
        'fixture': False,
    },
    {
        'date': '2024-09-01',
        'league': 'England',
        'home': 'Brighton',
        'away': 'Everton',
        'home_points_avg': 1.3,
        'away_points_avg': 0.9,
        'score': (0, 0),
        'fixture': False,
    },
    {
        'date': '2024-09-14',
        'league': 'England',
        'home': 'Chelsea',
        'away': 'West Ham',
        'home_points_avg': 1.9,
        'away_points_avg': 1.4,
        'score': (3, 1),
        'fixture': False,
    },
    {
        'date': '2024-10-05',
        'league': 'England',
        'home': 'Liverpool',
        'away': 'Crystal Palace',
        'home_points_avg': 2.4,
        'away_points_avg': 1.1,
        'score': (2, 1),
        'fixture': False,
    },
    {
        'date': '2024-11-02',
        'league': 'England',
        'home': 'Aston Villa',
        'away': 'Wolves',
        'home_points_avg': 1.6,
        'away_points_avg': 1.0,
        'score': (1, 1),
        'fixture': False,
    },
    {
        'date': '2025-01-18',
        'league': 'England',
        'home': 'Man City',
        'away': 'Brentford',
        'home_points_avg': 2.6,
        'away_points_avg': 1.3,
        'score': (4, 0),
        'fixture': False,
    },
    {
        'date': '2025-03-08',
        'league': 'England',
        'home': 'Nottingham',
        'away': 'Bournemouth',
        'home_points_avg': 1.4,
        'away_points_avg': 1.5,
        'score': (2, 2),
        'fixture': False,
    },
    {
        'date': '2025-05-25',
        'league': 'Spain',
        'home': 'Barcelona',
        'away': 'Real Madrid',
        'home_points_avg': 2.3,
        'away_points_avg': 2.2,
        'score': (3, 2),
        'fixture': False,
    },
    {
        'date': '2025-09-01',
        'league': 'England',
        'home': 'Arsenal',
        'away': 'Chelsea',
        'home_points_avg': 2.0,
        'away_points_avg': 1.7,
        'score': None,
        'fixture': True,
    },
]

# Base pre-match decimal odds per market and a per-provider multiplier.
_BASE_ODDS = {'home_win': 1.80, 'draw': 3.40, 'away_win': 4.20, 'over_2.5': 1.90, 'under_2.5': 1.95}
_PROVIDER_FACTOR = {'market_average': 0.98, 'market_maximum': 1.06}


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


class DummySoccerDataLoader(BaseDataLoader):
    """Dataloader for offline soccer sample data.

    The data are bundled in-play sample snapshots that require no downloading, so
    they familiarize the user with the dataloader interface and drive the
    documentation examples and doctests offline. It shares the interface of
    [`BaseDataLoader`][sportsbet.datasets.BaseDataLoader].

    Args:
        param_grid:
            Selects the sample data to include, mirroring scikit-learn's
            `ParameterGrid`. The default `None` selects all sample data.

    Examples:
        >>> from sportsbet.datasets import DummySoccerDataLoader
        >>> import pandas as pd
        >>> loader = DummySoccerDataLoader(param_grid={'league': ['England']})
        >>> X_train, Y_train, O_train = loader.extract_train_data(odds_type='market_average')
        >>> X_fix, Y_fix, O_fix = loader.extract_fixtures_data()
        >>> Y_fix is None
        True
        >>> pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
        >>> pd.testing.assert_index_equal(O_train.columns, O_fix.columns)
    """

    @property
    def sources(self: Self) -> tuple[_DummySoccerStats]:
        """The source publishing the parameters of the bundled data."""
        return (_DummySoccerStats(),)

    def _all_params(self: Self) -> list[dict]:
        return list(ParameterGrid(_PARAM_GRID))

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
                'home_team': match['home'],
                'away_team': match['away'],
            }
            for status, minutes, home_goals, away_goals in _timeline(match):
                markets = market_outcomes(pd.Series([home_goals]), pd.Series([away_goals]), _MARKETS).iloc[0].to_dict()
                features = (
                    {'home_points_avg': match['home_points_avg'], 'away_points_avg': match['away_points_avg']}
                    if status == 'preplay'
                    else dict.fromkeys(_FEATURES)
                )
                stats_records.append(
                    {
                        **identity,
                        'event_status': status,
                        'event_time': pd.Timedelta(minutes=minutes),
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        **markets,
                        **features,
                    },
                )
                if status == 'postplay':
                    continue
                for provider in _PROVIDERS:
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
            ['event_status', 'event_time', *IDENTITY_COLS, 'home_goals', 'away_goals', *_MARKETS, *_FEATURES]
        ]
        odds = pd.DataFrame(odds_records)[['event_status', 'event_time', *IDENTITY_COLS, 'provider', *_MARKETS]]
        return stats, odds
