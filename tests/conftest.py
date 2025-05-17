"""Configuration for the pytest test suite."""

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

CONFIG = """
from sklearn.model_selection import TimeSeriesSplit
from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation import OddsComparisonBettor
DATALOADER_CLASS = DummySoccerDataLoader
PARAM_GRID = {'league': ['England', 'Greece']}
DROP_NA_THRES = 0.8
ODDS_TYPE = 'interwetten'
BETTOR = OddsComparisonBettor(alpha=0.03)
CV = TimeSeriesSplit(2)
"""


@pytest.fixture(autouse=True, scope='session')
def pandas_terminal_width() -> None:
    """Set options to display data."""
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)


@pytest.fixture
def cli_config_path(tmp_path: Path) -> Path:
    """Create configuration file."""
    with Path.open(tmp_path / 'config.py', 'wt') as config_file:
        config_file.write(CONFIG)
    return tmp_path / 'config.py'


@pytest.fixture
def cli_runner() -> CliRunner:
    """CLI runner for tests."""
    return CliRunner()
