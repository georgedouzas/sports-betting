"""Concrete soccer statistics and odds schemas."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import Annotated

import pandas as pd

from .._base._schema import BaseOddsSchema, BaseStatsSchema, optional_col, required_col

IN_MATCH = ['inplay', 'postplay']
TRADING = ['preplay', 'inplay']


class SoccerStatsSchema(BaseStatsSchema):
    """Statistics schema for soccer snapshots."""

    date: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = required_col()
    league: str = required_col()
    division: int = required_col()
    year: int = required_col()
    home_team: str = required_col()
    away_team: str = required_col()
    # In-match observations
    home_goals: int = optional_col(IN_MATCH, fixed=False)
    away_goals: int = optional_col(IN_MATCH, fixed=False)
    # Pre-match, time-invariant features
    home_latest_streak: int = optional_col(['preplay'], fixed=True)
    away_latest_streak: int = optional_col(['preplay'], fixed=True)
    # Target-source market outcomes (derived from goals at each snapshot)
    home_win: int = optional_col(IN_MATCH, fixed=False)
    draw: int = optional_col(IN_MATCH, fixed=False)
    away_win: int = optional_col(IN_MATCH, fixed=False)
    over_2_5: int = optional_col(IN_MATCH, fixed=False, alias='over_2.5')
    under_2_5: int = optional_col(IN_MATCH, fixed=False, alias='under_2.5')
    over_3_5: int = optional_col(IN_MATCH, fixed=False, alias='over_3.5')
    under_3_5: int = optional_col(IN_MATCH, fixed=False, alias='under_3.5')


class SoccerOddsSchema(BaseOddsSchema):
    """Odds schema for soccer snapshots."""

    date: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = required_col()
    league: str = required_col()
    division: int = required_col()
    year: int = required_col()
    home_team: str = required_col()
    away_team: str = required_col()
    provider: str = optional_col(['preplay'], fixed=True)
    home_win: float = optional_col(TRADING, fixed=False)
    draw: float = optional_col(TRADING, fixed=False)
    away_win: float = optional_col(TRADING, fixed=False)
    over_2_5: float = optional_col(TRADING, fixed=False, alias='over_2.5')
    under_2_5: float = optional_col(TRADING, fixed=False, alias='under_2.5')
    over_3_5: float = optional_col(TRADING, fixed=False, alias='over_3.5')
    under_3_5: float = optional_col(TRADING, fixed=False, alias='under_3.5')
