"""Test the command line reaches what the Python API reaches.

The command line drifted behind the Python API once already, and nothing noticed, because nothing was watching. These
tests watch: a parameter added to the API and not to the command line fails here.
"""

import inspect

import pytest

from sportsbet.cli import main
from sportsbet.dataloaders import BaseDataLoader, DataLoader
from sportsbet.evaluation import OddsComparisonBettor, backtest
from sportsbet.sources import OddsApi

UNREACHABLE = {'self', 'param_grid', 'key', 'classifier', 'X', 'Y', 'O', 'bettor'}


def _options(command):
    """Return what a command can be told, by the names the library gives those things."""
    found = main
    for name in command:
        found = found.commands[name]
    return {param.name for param in found.params}


def _parameters(callable_):
    """Return the parameters of an API callable that a command should be able to reach."""
    return {name for name in inspect.signature(callable_).parameters if name not in UNREACHABLE}


@pytest.mark.parametrize(
    ('command', 'api', 'renamed'),
    [
        (['data', 'training'], BaseDataLoader.extract_train_data, {}),
        (['model', 'backtest'], backtest, {}),
        (['model', 'backtest'], OddsComparisonBettor.__init__, {'odds_types': 'model_odds_types'}),
        (['data', 'training'], OddsApi.__init__, {name: f'odds_{name}' for name in ('markets', 'regions', 'moments')}),
        (['data', 'training'], DataLoader.__init__, {}),
    ],
)
def test_a_command_reaches_what_the_api_reaches(command, api, renamed):
    """Test every parameter of the Python API can be given to the command line."""
    options = _options(command)
    wanted = {renamed.get(name, name) for name in _parameters(api)}
    assert not wanted - options, f'`{" ".join(command)}` cannot reach {sorted(wanted - options)}'


def test_the_sources_can_be_configured():
    """Test a data source is chosen and configured from the command line, as it is from Python.

    A source is where the data comes from and how it is bought, so a command line that cannot configure one cannot reach
    most of the library.
    """
    options = _options(['data', 'training'])
    assert {'stats', 'odds', 'odds_key_env', 'odds_markets', 'odds_regions', 'odds_moments'} <= options
    assert {'aliases', 'max_unmatched_rate'} <= options
