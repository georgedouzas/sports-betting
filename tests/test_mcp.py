"""Tests for the server that lets an assistant drive the library.

The surface it replaces had no tests at all, which is most of why it was replaced. Nothing here touches the network.
"""

import asyncio

import pytest

from sportsbet.mcp import server

FREE_CONFIG = """
from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation import OddsComparisonBettor
DATALOADER = DummySoccerDataLoader(param_grid={'league': ['England']})
BETTOR = OddsComparisonBettor(alpha=0.03)
"""

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


@pytest.fixture
def free_config(tmp_path):
    """A configuration whose sources charge nothing."""
    path = tmp_path / 'free.py'
    path.write_text(FREE_CONFIG)
    return str(path)


def _call(name, **arguments):
    """Call a tool the way an assistant would, and return what it answered."""
    _, structured = asyncio.run(server.call_tool(name, arguments))
    return structured.get('result', structured) if isinstance(structured, dict) else structured


def test_the_library_is_reachable_through_the_tools():
    """Test an assistant can find every capability the command line has."""
    tools = asyncio.run(server.list_tools())
    assert [tool.name for tool in tools] == TOOLS


def test_every_tool_is_handed_a_configuration():
    """Test a tool is told which configuration to use, rather than being handed a dataloader in its arguments.

    It is the same configuration the command line reads, so the two surfaces cannot come to mean different things, and a
    key stays out of a tool call, where it would be written into a transcript.
    """
    tools = asyncio.run(server.list_tools())
    for tool in tools:
        assert 'config_path' in tool.inputSchema['properties']
        assert 'key' not in tool.inputSchema['properties']


def test_a_free_preparation_needs_no_confirmation(free_config):
    """Test a preparation that costs nothing is not made to ask permission to cost nothing.

    The rule exists to stop a surprise on a bill, not to add ceremony to a free download.
    """
    _call('prepare', config_path=free_config)


def test_the_price_is_free_to_ask_for(free_config):
    """Test what a preparation would cost can be learned without paying for it."""
    estimate = _call('estimate_preparation', config_path=free_config)
    assert estimate['total_cost'] == 0


def test_a_preparation_that_costs_money_is_refused_until_the_price_is_known(monkeypatch, free_config):
    """Test an assistant cannot spend by accident.

    A season of odds costs thousands of credits. An assistant trying to be helpful could buy them in a single call, and
    whoever is paying would find out afterwards. So the refusal lives in the code, not in a sentence in a docstring
    asking the model to be careful.
    """
    monkeypatch.setattr(
        'sportsbet.mcp._server._report',
        lambda config_path: (None, {'to_fetch': 898, 'held': 0, 'cost': {'odds_api': COST}, 'total_cost': COST}),
    )
    with pytest.raises(Exception, match=str(COST)):
        _call('prepare', config_path=free_config)


def test_a_preparation_is_refused_when_the_price_confirmed_is_the_wrong_one(monkeypatch, free_config):
    """Test confirming some other number does not buy the data, and the real cost is named."""
    monkeypatch.setattr(
        'sportsbet.mcp._server._report',
        lambda config_path: (None, {'to_fetch': 898, 'held': 0, 'cost': {'odds_api': COST}, 'total_cost': COST}),
    )
    with pytest.raises(Exception, match=str(COST)):
        _call('prepare', config_path=free_config, confirm_cost=100)


def test_a_preparation_runs_once_the_price_is_confirmed(monkeypatch, free_config):
    """Test the data is bought when whoever is paying has been told what it costs and said so."""
    prepared = []

    class _Dataloader:
        def prepare(self):
            prepared.append(True)
            return type('Report', (), {'to_fetch': [], 'held': [], 'estimated_cost': {'odds_api': COST}})()

    monkeypatch.setattr(
        'sportsbet.mcp._server._report',
        lambda config_path: (
            _Dataloader(),
            {'to_fetch': 898, 'held': 0, 'cost': {'odds_api': COST}, 'total_cost': COST},
        ),
    )
    _call('prepare', config_path=free_config, confirm_cost=COST)
    assert prepared


def test_an_assistant_can_go_from_nothing_to_value_bets(free_config):
    """Test the whole journey works without a line of Python written by the user."""
    _call('prepare', config_path=free_config)

    params = _call('available_params', config_path=free_config)
    assert params

    train = _call('extract_train_data', config_path=free_config, odds_type='market_average')
    assert train['X']
    assert train['Y']
    assert train['O']

    fixtures = _call('extract_fixtures_data', config_path=free_config)
    assert 'X' in fixtures

    results = _call('backtest', config_path=free_config, odds_type='market_average')
    assert results

    bets = _call('bet', config_path=free_config, odds_type='market_average')
    assert isinstance(bets, list)


def test_a_tool_can_reach_a_source_that_fetches(monkeypatch, free_config):
    """Test the library can open its own event loop while a tool is being answered.

    A tool is answered inside an event loop, and the library opens one of its own to fetch, and a loop cannot be opened
    inside a loop. So every source that actually downloads would fail here, while the dummy one that downloads nothing
    would pass, which is how this went unnoticed until a real feed was pointed at it.
    """

    class _Dataloader:
        def prepare(self, dry_run=False, refresh=False):
            asyncio.run(asyncio.sleep(0))
            return type('Report', (), {'to_fetch': [], 'held': [], 'estimated_cost': {}, 'unavailable': []})()

    monkeypatch.setattr('sportsbet.mcp._server.read_dataloader', lambda mod: _Dataloader())
    assert _call('prepare', config_path=free_config) == {'fetched': 0, 'held': 0, 'cost': {}}
