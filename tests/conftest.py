"""Configuration for the pytest test suite."""

import socket
from importlib.resources import files
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytest
from click.testing import CliRunner

from sportsbet.datasets import (
    BaseOddsSchema,
    BaseStatsSchema,
    optional_col,
    required_col,
)

CONFIG = """
from sklearn.model_selection import TimeSeriesSplit
from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation import OddsComparisonBettor
DATALOADER = DummySoccerDataLoader(param_grid={'league': ['England', 'Spain']})
DROP_NA_THRES = 0.0
ODDS_TYPE = 'market_average'
TARGET_EVENT_STATUS = 'postplay'
TARGET_EVENT_TIME = None
BETTOR = OddsComparisonBettor(alpha=0.03)
CV = TimeSeriesSplit(2)
"""


@pytest.fixture(autouse=True, scope='session')
def pandas_terminal_width() -> None:
    """Set options to display data."""
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that reaches the network, unless it is marked `network`.

    Extraction must never fetch, so the only way to know it does not is to make fetching impossible and watch the suite
    stay green.
    """
    if request.node.get_closest_marker('network'):
        return

    def guard(*args: object, **kwargs: object) -> None:
        msg = 'The test suite must not reach the network. Use a recorded payload or mark the test `network`.'
        raise RuntimeError(msg)

    monkeypatch.setattr(socket.socket, 'connect', guard)
    monkeypatch.setattr(socket.socket, 'connect_ex', guard)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def cli_config_path(tmp_path: Path) -> Path:
    """Create configuration file."""
    with Path.open(tmp_path / 'config.py', 'wt') as config_file:
        config_file.write(CONFIG)
    return tmp_path / 'config.py'


@pytest.fixture
def stats() -> pd.DataFrame:
    """Load statistics data."""
    stats_file = files('tests') / 'samples' / 'stats.csv'
    data = pd.read_csv(stats_file, parse_dates=['date'])
    data['event_time'] = pd.to_timedelta(data['event_time'], unit='m').astype('timedelta64[ns]')
    data['date'] = pd.to_datetime(data['date'], utc=True).astype('datetime64[ns, UTC]')
    return data


@pytest.fixture
def odds() -> pd.DataFrame:
    """Load odds data."""
    odds_file = files('tests') / 'samples' / 'odds.csv'
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
        season: int = required_col()
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
        season: int = required_col()
        home_team: str = required_col()
        away_team: str = required_col()
        provider: str = optional_col(['preplay'], True)
        home_win: float = optional_col(['preplay', 'inplay'], False)
        away_win: float = optional_col(['preplay', 'inplay'], False)

    return OddsSchema
