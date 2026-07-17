"""Test the tools reach what the commands reach.

The command line and the tools drifted apart once and nothing noticed, because nothing was watching. The command line
gained saved artifacts and the event arguments that decide the moment a model bets at, and the tools kept neither, so an
agent could not choose the moment and re-downloaded the seasons on every call. With a metered odds feed that is money.

This watches. A command that gains an option and a tool that does not gain it fails here.
"""

import asyncio

from sportsbet.cli._cli import main
from sportsbet.mcp import server

RENAMED = {
    'dataloader_path': 'dataloader',
    'bettor_path': 'bettor',
    'data_path': 'output',
    'quote_path': 'quote',
    'venue_ref': 'venue',
}
UNREACHABLE = {'output'}
SPELLED = {
    'execution_read_status': {'quote'},
}
PAIRS = [
    (['dataloader', 'params'], 'available_params'),
    (['dataloader', 'odds-types'], 'odds_types'),
    (['dataloader', 'train', 'extract'], 'extract_train_data'),
    (['dataloader', 'exploration', 'extract'], 'extract_exploration_data'),
    (['dataloader', 'fixtures', 'extract'], 'extract_fixtures_data'),
    (['evaluation', 'backtest'], 'backtest'),
    (['evaluation', 'fit'], 'fit'),
    (['evaluation', 'bet'], 'bet'),
    (['execution', 'quote'], 'execution_quote'),
    (['execution', 'place'], 'execution_place'),
    (['execution', 'balance'], 'execution_read_balance'),
    (['execution', 'status'], 'execution_read_status'),
    (['execution', 'cancel'], 'execution_cancel'),
    (['execution', 'venue'], 'execution_venue_info'),
    (['execution', 'run'], 'execution_run'),
]


def _tools():
    """Return every tool and what it can be told."""
    return {tool.name: set(tool.inputSchema['properties']) for tool in asyncio.run(server.list_tools())}


def _command(path):
    """Return what a command can be told."""
    found = main
    for name in path:
        found = found.commands[name]
    return {RENAMED.get(param.name, param.name) for param in found.params}


def test_every_command_has_a_tool():
    """Test an agent can find every capability the command line has."""
    tools = _tools()
    for path, tool in PAIRS:
        assert tool in tools, f'`{" ".join(path)}` has no tool'


def test_the_data_and_model_tools_reach_what_the_commands_reach():
    """Test a tool can be told everything the command it mirrors can be told."""
    tools = _tools()
    for path, tool in PAIRS:
        wanted = _command(path) - UNREACHABLE - SPELLED.get(tool, set())
        assert not wanted - tools[tool], f'`{tool}` cannot be told {sorted(wanted - tools[tool])}'


def test_an_agent_can_choose_the_moment_it_bets_at():
    """Test the event arguments are reachable, since they are what decide the moment a model bets at.

    A tool set without them can only bet the one way, and the moment is most of what a betting model is about.
    """
    tools = _tools()
    moment = {'target_event_status', 'target_event_time', 'input_event_status', 'input_event_time', 'drop_na_thres'}
    assert moment <= tools['extract_train_data']
    assert moment <= tools['extract_exploration_data']


def test_an_agent_downloads_once_and_reuses_it():
    """Test the seasons are bought once.

    Everything after the extract reads what it wrote. A tool set that rebuilt from the selection each time would buy the
    same seasons again on every call, and a metered feed charges for each of them.
    """
    tools = _tools()
    assert 'output' in tools['extract_train_data']
    for reader in ('extract_fixtures_data', 'backtest', 'fit', 'bet'):
        assert 'dataloader' in tools[reader], f'`{reader}` does not read the saved dataloader'
    assert 'stats' not in tools['backtest']
    assert 'stats' not in tools['bet']


def test_an_agent_fits_once_and_reuses_it():
    """Test a model is fitted once and read back, rather than fitted again for every bet."""
    tools = _tools()
    assert 'output' in tools['fit']
    assert 'bettor' in tools['bet']
    assert 'model' not in tools['bet']


def test_no_tool_takes_a_secret():
    """Test a secret is never a tool argument, since a tool argument is written into a transcript."""
    forbidden = {'key', 'password', 'secret', 'token', 'app_key', 'username', 'credential'}
    for name, properties in _tools().items():
        assert not properties & forbidden, f'`{name}` can be handed {sorted(properties & forbidden)}'
