"""Read the sources the data comes from."""

from __future__ import annotations

from ..core import MATCH_COLS
from ._base import (
    BaseOddsSource,
    BaseSource,
    BaseStatsSource,
    RawItem,
    RawPayload,
    fetch_payloads,
    read_csv_content,
)
from ._odds._football_data import FootballDataOdds
from ._odds._odds_api import OddsApi
from ._odds._sample import SampleSoccerOdds
from ._resolver import (
    build_roster,
    count_common_prefix,
    measure_names_similarity,
    normalize_identity,
    normalize_team_name,
    pair_rosters,
    resolve_odds,
)
from ._schema import (
    BaseOddsSchema,
    BaseStatsSchema,
    optional_col,
    required_col,
)
from ._stats._euroleague import EuroLeagueStats
from ._stats._football_data import FootballDataStats
from ._stats._nba import NBAStats
from ._stats._sample import SampleSoccerStats
from ._utils import derive_market_outcomes

__all__: list[str] = [
    'MATCH_COLS',
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
    'SampleSoccerOdds',
    'SampleSoccerStats',
    'build_roster',
    'count_common_prefix',
    'derive_market_outcomes',
    'fetch_payloads',
    'measure_names_similarity',
    'normalize_identity',
    'normalize_team_name',
    'optional_col',
    'pair_rosters',
    'read_csv_content',
    'required_col',
    'resolve_odds',
]
