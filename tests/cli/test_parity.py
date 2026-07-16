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
        (['dataloader', 'train', 'extract'], BaseDataLoader.extract_train_data, {}),
        (['dataloader', 'exploration', 'extract'], BaseDataLoader.extract_exploration_data, {}),
        (['evaluation', 'backtest'], backtest, {}),
        (['evaluation', 'backtest'], OddsComparisonBettor.__init__, {'odds_types': 'model_odds_types'}),
        (
            ['dataloader', 'train', 'extract'],
            OddsApi.__init__,
            {name: f'odds_{name}' for name in ('markets', 'regions', 'moments')},
        ),
        (['dataloader', 'train', 'extract'], DataLoader.__init__, {}),
    ],
)
def test_a_command_reaches_what_the_api_reaches(command, api, renamed):
    """Test every parameter of the Python API can be given to the command line."""
    options = _options(command)
    wanted = {renamed.get(name, name) for name in _parameters(api)}
    assert not wanted - options, f'`{" ".join(command)}` cannot reach {sorted(wanted - options)}'


def test_execution_reaches_what_the_api_reaches():
    """Test every execution capability is reachable from the command line."""
    for command in (['venue'], ['markets'], ['balance'], ['quote'], ['place'], ['status'], ['cancel']):
        assert 'venue_ref' in _options(['execution', *command])


def test_the_site_path_reaches_what_the_api_reaches():
    """Test the browser primitives are on the command line as well as in the tools."""
    for command in (['read'], ['act'], ['fix']):
        assert 'venue_ref' in _options(['execution', 'page', *command])
    assert {'click_ref', 'type_ref', 'select_ref'} <= _options(['execution', 'page', 'act'])
    assert {'match', 'locators'} <= _options(['execution', 'page', 'fix'])


def test_cancelling_names_the_bet_by_what_makes_it_that_bet():
    """Test a bet is cancelled by its identity, which is the venue, the match, the market and the selection."""
    assert {'match', 'market', 'selection'} <= _options(['execution', 'cancel'])


def test_placing_takes_both_confirmations_and_both_ceilings():
    """Test the command line can refuse and can be confirmed.

    Nothing is staked without both figures, so a command line missing either could not place at all, and one missing a
    ceiling could not refuse.
    """
    options = _options(['execution', 'place'])
    assert {'confirm_stake', 'confirm_exposure'} <= options
    assert {'max_stake', 'max_exposure', 'kill'} <= options


def test_placing_has_no_dry_run_flag():
    """Test a dry run is the absence of a confirmation rather than a flag.

    A flag has a default, and a default that places is a default that spends. There is nothing here to set wrongly.
    """
    assert 'dry_run' not in _options(['execution', 'place'])


def test_no_command_takes_a_secret():
    """Test a secret is never a command line option, since an option is a shell history entry."""
    forbidden = {'key', 'password', 'secret', 'token', 'app_key', 'username', 'credential'}
    for group in ('venue', 'markets', 'balance', 'quote', 'place', 'status', 'cancel'):
        options = _options(['execution', group])
        assert not options & forbidden, f'`execution {group}` can be handed {sorted(options & forbidden)}'


def test_the_sources_can_be_configured():
    """Test a data source is chosen and configured from the command line, as it is from Python.

    A source is where the data comes from and how it is bought, so a command line that cannot configure one cannot reach
    most of the library.
    """
    options = _options(['dataloader', 'train', 'extract'])
    assert {'stats', 'odds', 'odds_key_env', 'odds_markets', 'odds_regions', 'odds_moments'} <= options
    assert {'aliases', 'max_unmatched_rate'} <= options
