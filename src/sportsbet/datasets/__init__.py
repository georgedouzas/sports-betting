"""It provides the tools to extract sports betting data."""

from __future__ import annotations

from ._base._dataloader import BaseDataLoader, load_dataloader
from ._base._factory import from_dataframe, from_snapshots
from ._base._schema import (
    BaseOddsSchema,
    BaseStatsSchema,
    optional_col,
    required_col,
)
from ._basketball._dataloader import BasketballDataLoader
from ._resolver import ReconciliationReport, UnmatchedError, resolve
from ._soccer._dataloader import SoccerDataLoader
from ._soccer._dummy import DummySoccerDataLoader
from ._sources._base import (
    BaseOddsSource,
    BaseSource,
    BaseStatsSource,
    RawItem,
    RawPayload,
)
from ._sources._euroleague import EuroLeagueStats
from ._sources._football_data import FootballDataOdds, FootballDataStats
from ._sources._nba import NBAStats
from ._sources._odds_api import OddsApi
from ._store import BaseStore, LocalStore, NotPreparedError, PreparationReport
from ._utils import market_outcomes

__all__: list[str] = [
    'BaseDataLoader',
    'BaseOddsSchema',
    'BaseOddsSource',
    'BaseSource',
    'BaseStatsSchema',
    'BaseStatsSource',
    'BaseStore',
    'BasketballDataLoader',
    'DummySoccerDataLoader',
    'EuroLeagueStats',
    'FootballDataOdds',
    'FootballDataStats',
    'LocalStore',
    'NBAStats',
    'NotPreparedError',
    'OddsApi',
    'PreparationReport',
    'RawItem',
    'RawPayload',
    'ReconciliationReport',
    'SoccerDataLoader',
    'UnmatchedError',
    'from_dataframe',
    'from_snapshots',
    'load_dataloader',
    'market_outcomes',
    'optional_col',
    'required_col',
    'resolve',
]
