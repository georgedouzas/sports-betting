"""Configuration for the pytest test suite."""

import pandas as pd
import pytest


@pytest.fixture(autouse=True, scope='session')
def pandas_terminal_width() -> None:  # noqa: PT004
    """Set options to display data."""
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)
