"""Tests for the server that lets an assistant drive the library.

The surface it replaces had no tests at all, which is most of why it was replaced. Nothing here touches the network.
"""

import asyncio

import pandas as pd

from sportsbet.mcp import server

SELECTION = {'stats': 'football-data', 'odds': 'football-data', 'leagues': ['England']}
TOOLS = [
    'available_params',
    'extract_train_data',
    'extract_fixtures_data',
    'backtest',
    'bet',
]


def _call(name, **arguments):
    """Call a tool the way an assistant would, and return what it answered."""
    _, structured = asyncio.run(server.call_tool(name, arguments))
    return structured.get('result', structured) if isinstance(structured, dict) else structured


def test_the_library_is_reachable_through_the_tools():
    """Test an assistant can find every capability the command line has."""
    tools = asyncio.run(server.list_tools())
    assert [tool.name for tool in tools] == TOOLS


def test_a_tool_is_told_what_to_do_and_reads_no_file():
    """Test a tool takes what it needs in its arguments, as a command does."""
    tools = asyncio.run(server.list_tools())
    for tool in tools:
        properties = tool.inputSchema['properties']
        assert 'stats' in properties
        assert 'config_path' not in properties


def test_a_key_is_never_an_argument():
    """Test what a tool is told is the name of the variable holding the key, never the key.

    A key passed as an argument is a key written into a transcript.
    """
    tools = asyncio.run(server.list_tools())
    for tool in tools:
        properties = tool.inputSchema['properties']
        assert 'odds_key_env' in properties
        assert 'key' not in properties


def test_a_tool_can_reach_a_source_that_fetches(monkeypatch):
    """Test the library can open its own event loop while a tool is being answered.

    A tool is answered inside an event loop, and the library opens one of its own to fetch, and a loop cannot be opened
    inside a loop.
    """
    opened = []

    class _Dataloader:
        def extract_train_data(self, **rest):
            asyncio.run(asyncio.sleep(0))
            opened.append(True)
            return pd.DataFrame([{'a': 1}]), pd.DataFrame([{'b': 1}]), pd.DataFrame([{'c': 1}])

    monkeypatch.setattr('sportsbet.mcp._server.build_dataloader', lambda **selection: _Dataloader())
    _call('extract_train_data', **SELECTION)
    assert opened


def test_an_assistant_can_go_from_nothing_to_value_bets(offline_dataloader):
    """Test the whole journey works without a line of Python written by the user."""
    assert _call('available_params', **SELECTION)

    train = _call('extract_train_data', **SELECTION, odds_type='market_average')
    assert train['X']
    assert train['Y']
    assert train['O']

    assert 'X' in _call('extract_fixtures_data', **SELECTION, odds_type='market_average')
    assert _call('backtest', **SELECTION, odds_type='market_average', model='odds-comparison', cv=2)
    assert isinstance(_call('bet', **SELECTION, odds_type='market_average', model='odds-comparison'), list)


def test_the_tools_reach_what_the_commands_reach():
    """Test an assistant can say everything a command can say."""
    tools = {tool.name: set(tool.inputSchema['properties']) for tool in asyncio.run(server.list_tools())}
    strategy = {'model', 'alpha', 'betting_markets', 'init_cash', 'stake', 'model_odds_types'}
    assert strategy <= tools['backtest']
    assert strategy <= tools['bet']
    assert 'cv' in tools['backtest']


def test_a_strategy_reaches_the_model(offline_dataloader):
    """Test what a tool is told about the model is what the model is given."""
    common = {'odds_type': 'market_average', 'model': 'odds-comparison', 'cv': 2}
    one = _call('backtest', **SELECTION, **common, betting_markets=['home_win'])
    every = _call('backtest', **SELECTION, **common)
    assert one[0]['Number of bets'] < every[0]['Number of bets']
