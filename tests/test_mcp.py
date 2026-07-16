"""Tests for the server that lets an assistant drive the library.

The surface it replaces had no tests at all, which is most of why it was replaced. Nothing here touches the network.
"""

import asyncio

import pandas as pd

from sportsbet.mcp import server

SELECTION = {'stats': 'football-data', 'odds': 'football-data', 'leagues': ['England']}
TOOLS = [
    'available_params',
    'extract_exploration_data',
    'extract_train_data',
    'extract_fixtures_data',
    'backtest',
    'bet',
    'execution_venue_info',
    'execution_authenticate',
    'execution_read_balance',
    'execution_list_markets',
    'execution_quote',
    'execution_place',
    'execution_read_status',
    'execution_cancel',
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
        assert 'config_path' not in properties
        wanted = 'venue' if tool.name.startswith('execution_') else 'stats'
        assert wanted in properties


def test_a_key_is_never_an_argument():
    """Test what a tool is told is the name of the variable holding a secret, never the secret.

    A secret passed as an argument is a secret written into a transcript. A data tool names the variable itself, and an
    execution tool names a venue that names its own, so neither has anywhere to put one.
    """
    forbidden = {'key', 'password', 'secret', 'token', 'app_key', 'username', 'credential'}
    tools = asyncio.run(server.list_tools())
    for tool in tools:
        properties = set(tool.inputSchema['properties'])
        assert not properties & forbidden, f'`{tool.name}` can be handed {sorted(properties & forbidden)}'
        if not tool.name.startswith('execution_'):
            assert 'odds_key_env' in properties


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


VENUE_FILE = """
from tests.conftest import FakeVenue

VENUE = FakeVenue(
    prices={('Arsenal vs Chelsea', 'home_win', 'Arsenal'): 2.10},
)
"""
STAKE = 10.0
INTENTS = [
    {'match': 'Arsenal vs Chelsea', 'market': 'home_win', 'selection': 'Arsenal', 'stake': STAKE, 'min_price': 2.0},
]


def _venue_ref(tmp_path):
    """Write a venue an agent can name, and return the reference."""
    path = tmp_path / 'venue.py'
    path.write_text(VENUE_FILE)
    return f'{path}:VENUE'


def test_an_agent_that_stakes_nothing_gets_the_quote(tmp_path):
    """Test a tool call with no confirmation stakes nothing.

    An agent is the caller most likely to skim, which is why the rule is in the code rather than in the description.
    """
    venue = _venue_ref(tmp_path)
    quote = _call('execution_quote', venue=venue, intents=INTENTS)
    assert quote['total_stake'] == STAKE
    receipts = _call('execution_place', venue=venue, quote=quote)
    assert [receipt['status'] for receipt in receipts] == ['dry_run']
    assert sum(receipt['stake'] for receipt in receipts) == 0.0


def test_an_agent_passing_the_wrong_figures_is_told_the_real_ones(tmp_path):
    """Test a tool call with figures that do not match stakes nothing and states the real figures."""
    venue = _venue_ref(tmp_path)
    quote = _call('execution_quote', venue=venue, intents=INTENTS)
    receipts = _call('execution_place', venue=venue, quote=quote, confirm_stake=999.0, confirm_exposure=999.0)
    assert [receipt['status'] for receipt in receipts] == ['refused_unconfirmed']
    assert sum(receipt['stake'] for receipt in receipts) == 0.0
    assert str(STAKE) in receipts[0]['detail']


def test_an_agent_passing_the_quoted_figures_places_the_bets(tmp_path):
    """Test a tool call with the quoted figures places the bets."""
    venue = _venue_ref(tmp_path)
    quote = _call('execution_quote', venue=venue, intents=INTENTS)
    receipts = _call(
        'execution_place',
        venue=venue,
        quote=quote,
        confirm_stake=quote['total_stake'],
        confirm_exposure=quote['total_exposure'],
    )
    assert [receipt['status'] for receipt in receipts] == ['matched_full']
    assert sum(receipt['stake'] for receipt in receipts) == STAKE


def test_the_notes_come_back_exactly_as_they_were_written(tmp_path):
    """Test the site knowledge reaches the agent and the library never reads it."""
    path = tmp_path / 'venue.py'
    path.write_text(
        "from tests.conftest import FakeVenue\n"
        "VENUE = FakeVenue()\n"
        "VENUE.notes = 'The slip opens on the right. Confirm is two steps.'\n"
        "VENUE.url = 'https://example.invalid/stoixima'\n",
    )
    info = _call('execution_venue_info', venue=f'{path}:VENUE')
    assert info['notes'] == 'The slip opens on the right. Confirm is two steps.'
    assert info['url'] == 'https://example.invalid/stoixima'
