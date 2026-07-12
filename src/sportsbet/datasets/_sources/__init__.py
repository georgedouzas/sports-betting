"""It provides the data sources of the dataloaders."""

from __future__ import annotations

from ._base import BaseOddsSource, BaseSource, BaseStatsSource, RawItem, RawPayload
from ._football_data import FootballDataOdds, FootballDataStats

__all__: list[str] = [
    'BaseOddsSource',
    'BaseSource',
    'BaseStatsSource',
    'FootballDataOdds',
    'FootballDataStats',
    'RawItem',
    'RawPayload',
]
