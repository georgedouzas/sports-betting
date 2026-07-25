"""Configuration for the pytest test suite."""

from importlib.resources import files
from typing import Annotated

import pandas as pd
import pytest

from sportsbet.sources import (
    BaseOddsSchema,
    BaseStatsSchema,
    optional_col,
    required_col,
)

SAMPLES_PATH = files('tests') / 'sources' / 'samples'


@pytest.fixture
def stats() -> pd.DataFrame:
    """Load statistics data."""
    stats_file = SAMPLES_PATH / 'stats.csv'
    data = pd.read_csv(stats_file, parse_dates=['date'])
    data['event_time'] = pd.to_timedelta(data['event_time'], unit='m').astype('timedelta64[ns]')
    data['date'] = pd.to_datetime(data['date'], utc=True).astype('datetime64[ns, UTC]')
    return data


@pytest.fixture
def odds() -> pd.DataFrame:
    """Load odds data."""
    odds_file = SAMPLES_PATH / 'odds.csv'
    data = pd.read_csv(odds_file, parse_dates=['date'])
    data['event_time'] = pd.to_timedelta(data['event_time'], unit='m').astype('timedelta64[ns]')
    data['date'] = pd.to_datetime(data['date'], utc=True).astype('datetime64[ns, UTC]')
    return data


@pytest.fixture
def stats_schema() -> BaseStatsSchema:
    """Load statistics schema."""

    class StatsSchema(BaseStatsSchema):
        """Statistics schema."""

        date: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = required_col()
        league: str = required_col()
        division: int = required_col()
        year: int = required_col()
        home_team: str = required_col()
        away_team: str = required_col()
        home_goals: int = optional_col(['inplay'], False)
        away_goals: int = optional_col(['inplay'], False)
        home_latest_streak: int = optional_col(['preplay'], True)
        away_latest_streak: int = optional_col(['preplay'], True)

    return StatsSchema


@pytest.fixture
def odds_schema() -> BaseOddsSchema:
    """Load odds schema."""

    class OddsSchema(BaseOddsSchema):
        """Odds schema."""

        date: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = required_col()
        league: str = required_col()
        division: int = required_col()
        year: int = required_col()
        home_team: str = required_col()
        away_team: str = required_col()
        provider: str = optional_col(['preplay'], True)
        home_win: float = optional_col(['preplay', 'inplay'], False)
        away_win: float = optional_col(['preplay', 'inplay'], False)

    return OddsSchema
