"""Tests for the server that lets an assistant drive the library.

The surface it replaces had no tests at all, which is most of why it was replaced. Nothing here touches the network.
"""

import asyncio

import pytest

from sportsbet.mcp import server

SELECTION = {'stats': 'football-data', 'odds': 'football-data', 'leagues': ['England']}
TOOLS = [
    'available_params',
    'estimate_preparation',
    'prepare',
    'extract_train_data',
    'extract_fixtures_data',
    'backtest',
    'bet',
]
COST = 17722
METERED = {'to_fetch': 898, 'held': 0, 'cost': {'odds_api': COST}, 'total_cost': COST}


def _call(name, **arguments):
    """Call a tool the way an assistant would, and return what it answered."""
    _, structured = asyncio.run(server.call_tool(name, arguments))
    return structured.get('result', structured) if isinstance(structured, dict) else structured


def test_the_library_is_reachable_through_the_tools():
    """Test an assistant can find every capability the command line has."""
    tools = asyncio.run(server.list_tools())
    assert [tool.name for tool in tools] == TOOLS


def test_a_tool_is_told_what_to_do_and_reads_no_file():
    """Test a tool takes what it needs in its arguments, as a command does.

    There is nothing to write down first, and nothing left behind to fall out of date.
    """
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


def test_a_free_preparation_needs_no_confirmation(offline_dataloader):
    """Test a preparation that costs nothing is not made to ask permission to cost nothing.

    The rule exists to stop a surprise on a bill, not to add ceremony to a free download.
    """
    _call('prepare', **SELECTION)


def test_the_price_is_free_to_ask_for(offline_dataloader):
    """Test what a preparation would cost can be learned without paying for it."""
    assert _call('estimate_preparation', **SELECTION)['total_cost'] == 0


def test_a_preparation_that_costs_money_is_refused_until_the_price_is_known(monkeypatch, offline_dataloader):
    """Test an assistant cannot spend by accident.

    A season of odds costs thousands of credits. An assistant trying to be helpful could buy them in a single call, and
    whoever is paying would find out afterwards. So the refusal lives in the code, not in a sentence in a docstring
    asking the model to be careful.
    """
    monkeypatch.setattr('sportsbet.mcp._server._report', lambda selection: (None, METERED))
    with pytest.raises(Exception, match=str(COST)):
        _call('prepare', **SELECTION)


def test_a_preparation_is_refused_when_the_price_confirmed_is_the_wrong_one(monkeypatch, offline_dataloader):
    """Test confirming some other number does not buy the data, and the real cost is named."""
    monkeypatch.setattr('sportsbet.mcp._server._report', lambda selection: (None, METERED))
    with pytest.raises(Exception, match=str(COST)):
        _call('prepare', **SELECTION, confirm_cost=100)


def test_a_preparation_runs_once_the_price_is_confirmed(monkeypatch, offline_dataloader):
    """Test the data is bought when whoever is paying has been told what it costs and has said so."""
    prepared = []

    class _Dataloader:
        def prepare(self):
            prepared.append(True)
            return type('Report', (), {'to_fetch': [], 'held': [], 'estimated_cost': {'odds_api': COST}})()

    monkeypatch.setattr('sportsbet.mcp._server._report', lambda selection: (_Dataloader(), METERED))
    _call('prepare', **SELECTION, confirm_cost=COST)
    assert prepared


def test_a_tool_can_reach_a_source_that_fetches(monkeypatch):
    """Test the library can open its own event loop while a tool is being answered.

    A tool is answered inside an event loop, and the library opens one of its own to fetch, and a loop cannot be opened
    inside a loop. So every source that actually downloads would fail here, while a source that downloads nothing would
    pass, which is how this went unnoticed until a real feed was pointed at it.
    """

    class _Dataloader:
        def prepare(self, dry_run=False, refresh=False):
            asyncio.run(asyncio.sleep(0))
            return type('Report', (), {'to_fetch': [], 'held': [], 'estimated_cost': {}, 'unavailable': []})()

    monkeypatch.setattr('sportsbet.mcp._server.build_dataloader', lambda **selection: _Dataloader())
    assert _call('prepare', **SELECTION) == {'fetched': 0, 'held': 0, 'cost': {}}


def test_an_assistant_can_go_from_nothing_to_value_bets(offline_dataloader):
    """Test the whole journey works without a line of Python written by the user."""
    _call('prepare', **SELECTION)

    assert _call('available_params', **SELECTION)

    train = _call('extract_train_data', **SELECTION, odds_type='market_average')
    assert train['X']
    assert train['Y']
    assert train['O']

    assert 'X' in _call('extract_fixtures_data', **SELECTION, odds_type='market_average')
    assert _call('backtest', **SELECTION, odds_type='market_average', model='odds-comparison', cv=2)
    assert isinstance(_call('bet', **SELECTION, odds_type='market_average', model='odds-comparison'), list)


def test_the_tools_reach_what_the_commands_reach():
    """Test an assistant can say everything a command can say.

    The two surfaces are told the same things, so a betting strategy that can be described to one can be described to
    the other. The tools could not name the markets to bet on, the cash or the stake, so they quietly bet every market
    with the defaults and answered a different question from the one the command answers.
    """
    tools = {tool.name: set(tool.inputSchema['properties']) for tool in asyncio.run(server.list_tools())}
    strategy = {'model', 'alpha', 'betting_markets', 'init_cash', 'stake', 'model_odds_types'}
    assert strategy <= tools['backtest']
    assert strategy <= tools['bet']
    assert 'cv' in tools['backtest']


def test_a_strategy_reaches_the_model(offline_dataloader):
    """Test what a tool is told about the model is what the model is given."""
    results = _call(
        'backtest',
        **SELECTION,
        odds_type='market_average',
        model='odds-comparison',
        betting_markets=['home_win'],
        cv=2,
    )
    every_market = _call('backtest', **SELECTION, odds_type='market_average', model='odds-comparison', cv=2)
    assert results[0]['Number of bets'] < every_market[0]['Number of bets']
