"""Configuration for the pytest test suite."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(autouse=True, scope='session')
def pandas_terminal_width() -> None:  # noqa: PT004
    """Set options to display data."""
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)


@pytest.fixture()
def cli_config_path(tmp_path: Path) -> Path:
    """Create configuration file."""
    with Path.open(tmp_path / 'config.py', 'wt') as config_file:
        config_file.write(
            """
from sklearn.model_selection import TimeSeriesSplit
from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation import OddsComparisonBettor
CONFIG = {
    'data': {
    'dataloader': DummySoccerDataLoader,
        'param_grid': {
            'league': ['England', 'Greece'],
        },
        'drop_na_thres': 0.8,
        'odds_type': 'interwetten',
    },
    'betting': {'bettor': OddsComparisonBettor, 'alpha': 0.03, 'tscv': TimeSeriesSplit(2)},
}
""",
        )
    return tmp_path / 'config.py'
