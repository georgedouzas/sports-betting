"""Includes utilities for soccer data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import ClassVar, Self

from ... import OutputsMapping
from .._base import BaseDataLoader


class BaseSoccerDataLoader(BaseDataLoader):
    """The base class for soccer dataloaders.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    OVER_UNDER_SCORES: ClassVar[dict[str, float]] = {
        '2.5': 2.5,
        '3.5': 3.5,
        '4.5': 4.5,
        '5.5': 5.5,
        '6.5': 6.5,
        '7.5': 7.5,
        '8.5': 8.5,
        '9.5': 9.5,
    }

    @property
    def _outputs_mapping(self: Self) -> OutputsMapping:
        return {
            ('home_team_goals', 'away_team_goals'): {
                'home_win': lambda data: data['home_team_goals'] > data['away_team_goals'],
                'away_win': lambda data: data['away_team_goals'] > data['home_team_goals'],
                'draw': lambda data: data['home_team_goals'] == data['away_team_goals'],
                'over_2.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['2.5'],
                'under_2.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['2.5'],
                'over_3.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['3.5'],
                'under_3.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['3.5'],
                'over_4.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['4.5'],
                'under_4.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['4.5'],
                'over_5.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['5.5'],
                'under_5.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['5.5'],
                'over_6.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['6.5'],
                'under_6.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['6.5'],
                'over_7.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['7.5'],
                'under_7.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['7.5'],
                'over_8.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['8.5'],
                'under_8.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['8.5'],
                'over_9.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                > self.OVER_UNDER_SCORES['9.5'],
                'under_9.5': lambda data: data['home_team_goals'] + data['away_team_goals']
                < self.OVER_UNDER_SCORES['9.5'],
            },
        }

    @property
    def _required_cols(self: Self) -> list[str]:
        return ['league', 'division', 'year', 'home_team', 'away_team']

    @property
    def _stages(self: Self) -> list[str]:
        return [
            '0 min',
            *[f'{minute!s} min' for minute in range(1, 46)],
            'half_time',
            *[f'{minute!s} min' for minute in range(45, 90)],
            'full_time',
        ]
