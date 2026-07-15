"""It provides the sources the data comes from."""

from __future__ import annotations

from ._base import (
    BaseOddsSource,
    BaseSource,
    BaseStatsSource,
    RawItem,
    RawPayload,
)
from ._euroleague import EuroLeagueStats
from ._football_data import FootballDataOdds, FootballDataStats
from ._nba import NBAStats
from ._odds_api import OddsApi
from ._resolver import ReconciliationReport, UnmatchedError, resolve
from ._sample import SampleSoccerOdds, SampleSoccerStats
from ._schema import (
    BaseOddsSchema,
    BaseStatsSchema,
    optional_col,
    required_col,
)
from ._utils import market_outcomes

__all__: list[str] = [
    'BaseOddsSchema',
    'BaseOddsSource',
    'BaseSource',
    'BaseStatsSchema',
    'BaseStatsSource',
    'EuroLeagueStats',
    'FootballDataOdds',
    'FootballDataStats',
    'NBAStats',
    'OddsApi',
    'RawItem',
    'RawPayload',
    'ReconciliationReport',
    'SampleSoccerOdds',
    'SampleSoccerStats',
    'UnmatchedError',
    'market_outcomes',
    'optional_col',
    'required_col',
    'resolve',
]
