import pandas
import pytest

@pytest.fixture(autouse=True, scope='session')
def pandas_terminal_width():
    pandas.set_option('display.width', 1000)
    pandas.set_option('display.max_columns', 1000)