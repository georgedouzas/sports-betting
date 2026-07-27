"""Tests for the server that lets an assistant drive the library.

The surface it replaces had no tests at all, which is most of why it was replaced. Nothing here touches the network.
"""

import asyncio

import pandas as pd
import pytest

from sportsbet.execution import BetIdentity, PlacementIntent, PlacementReceipt, PlacementStatus, build_receipts_frame
from sportsbet.mcp import server
from tests.conftest import FakeVenue

SELECTION = {'stats': 'football-data', 'odds': 'football-data', 'leagues': ['England']}
STAKE = 10.0
TOOLS = [
    'available_params',
    'odds_types',
    'extract_train_data',
    'extract_exploration_data',
    'extract_fixtures_data',
    'backtest',
    'fit',
    'bet',
    'execution_venue_info',
    'execution_authenticate',
    'execution_read_balance',
    'execution_list_markets',
    'execution_read_status',
    'execution_cancel',
    'execution_run',
    'browser_navigate',
    'browser_snapshot',
    'browser_click',
    'browser_type',
    'browser_select',
    'browser_fix',
]


def _call(name, **arguments):
    """Call a tool the way an assistant would, and return what it answered."""
    _, structured = asyncio.run(server.call_tool(name, arguments))
    return structured.get('result', structured) if isinstance(structured, dict) else structured


def test_the_library_is_reachable_through_the_tools():
    """Test an assistant can find every capability the command line has."""
    tools = asyncio.run(server.list_tools())
    assert [tool.name for tool in tools] == TOOLS


SELECTS = {'available_params', 'odds_types', 'extract_train_data', 'extract_exploration_data'}
READS = {'extract_fixtures_data', 'backtest', 'fit', 'bet'}


def test_a_tool_is_told_what_to_do_in_its_arguments():
    """Test a tool takes what it needs in its arguments, as a command does.

    A tool that selects is told the selection. A tool that reads what an extract wrote is told where that file is, the
    way a command is, so the seasons are downloaded once instead of once per call.
    """
    tools = asyncio.run(server.list_tools())
    for tool in tools:
        properties = tool.inputSchema['properties']
        assert 'config_path' not in properties
        if tool.name in SELECTS:
            assert 'stats' in properties
        elif tool.name in READS:
            assert 'dataloader' in properties
        else:
            assert 'venue' in properties


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
        if tool.name in SELECTS:
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


def test_an_assistant_can_go_from_nothing_to_value_bets(offline_dataloader, tmp_path):
    """Test the whole journey works without a line of Python written by the user.

    It is also the download-once story: the extract writes the dataloader, and everything after it reads that file, so
    a metered odds feed is bought once rather than once per call.
    """
    assert _call('available_params', **SELECTION)
    assert _call('odds_types', **SELECTION)

    saved = str(tmp_path / 'dataloader.pkl')
    train = _call('extract_train_data', **SELECTION, odds_type='market_average', output=saved)
    assert train['X']
    assert train['Y']
    assert train['O']
    assert train['output'] == saved

    assert 'X' in _call('extract_fixtures_data', dataloader=saved)
    assert _call('backtest', dataloader=saved, model='OddsComparisonBettor()', cv=2)

    model = str(tmp_path / 'model.pkl')
    fitted = _call('fit', dataloader=saved, output=model, model='OddsComparisonBettor()')
    assert fitted['output'] == model
    assert fitted['betting_markets']

    assert isinstance(_call('bet', dataloader=saved, bettor=model), list)


def test_the_seasons_are_downloaded_once(offline_dataloader, tmp_path):
    """Test nothing after the extract goes back to the source.

    An agent that backtests, fits and bets used to buy the same seasons three times. With a metered feed that is money,
    and the agent doing the obvious thing is what spent it.
    """
    saved = str(tmp_path / 'dataloader.pkl')
    _call('extract_train_data', **SELECTION, odds_type='market_average', output=saved)

    built = []
    monkey = pytest.MonkeyPatch()
    monkey.setattr('sportsbet.mcp._server.build_dataloader', lambda **rest: built.append(True))
    try:
        _call('backtest', dataloader=saved, model='OddsComparisonBettor()', cv=2)
        model = str(tmp_path / 'model.pkl')
        _call('fit', dataloader=saved, output=model, model='OddsComparisonBettor()')
        _call('bet', dataloader=saved, bettor=model)
        _call('extract_fixtures_data', dataloader=saved)
    finally:
        monkey.undo()
    assert built == []


def test_the_tools_reach_what_the_commands_reach():
    """Test an assistant can say everything a command can say."""
    tools = {tool.name: set(tool.inputSchema['properties']) for tool in asyncio.run(server.list_tools())}
    assert 'model' in tools['backtest']
    assert 'model' in tools['fit']
    assert 'cv' in tools['backtest']


def test_the_model_expression_reaches_the_model(offline_dataloader, tmp_path):
    """Test what a tool is told about the model is what the model is given."""
    saved = str(tmp_path / 'dataloader.pkl')
    _call('extract_train_data', **SELECTION, odds_type='market_average', output=saved)
    one = _call('backtest', dataloader=saved, model='OddsComparisonBettor(betting_markets=["home_win"])', cv=2)
    every = _call('backtest', dataloader=saved, model='OddsComparisonBettor()', cv=2)
    assert one[0]['Number of bets'] < every[0]['Number of bets']


def test_the_notes_of_a_website_reach_the_agent(tmp_path):
    """Test the site knowledge of a browser session reaches the agent and the library never reads it.

    The tool has to answer for a browser session, since that is the venue kind whose notes exist. A plain venue would
    have hidden that the tool refused one.
    """
    path = tmp_path / 'venue.py'
    path.write_text(
        "from sportsbet.execution import BrowserSession\n"
        "VENUE = BrowserSession(\n"
        "    key='novibet',\n"
        "    url='https://example.invalid/stoixima',\n"
        "    notes='The slip opens on the right. Confirm is two steps.',\n"
        ")\n",
    )
    info = _call('execution_venue_info', venue=f'{path}:VENUE')
    assert info['key'] == 'novibet'
    assert info['notes'] == 'The slip opens on the right. Confirm is two steps.'
    assert info['url'] == 'https://example.invalid/stoixima'


def test_execution_run_dry_run_returns_the_bet_it_would_make(monkeypatch):
    """A dry run through the tool returns the one receipt the unit would place, staking nothing."""

    async def fake_execute_event(event, bettor, loader, session, *, stake, urls, live, poll):
        receipt = PlacementReceipt(
            identity=BetIdentity('stub', event, 'home_win', 'Arsenal'),
            status=PlacementStatus.DRY_RUN,
            value_bet=f'{event}|home_win',
        )
        return build_receipts_frame([receipt])

    monkeypatch.setattr('sportsbet.mcp._server.build_venue', lambda ref: object())
    monkeypatch.setattr('sportsbet.mcp._server.load_dataloader', lambda path: object())
    monkeypatch.setattr('sportsbet.mcp._server.load_bettor', lambda path: object())
    monkeypatch.setattr('sportsbet.mcp._server.execute_event', fake_execute_event)
    records = _call(
        'execution_run',
        venue='venue.py:VENUE',
        dataloader='loader.pkl',
        bettor='model.pkl',
        event='Arsenal vs Chelsea',
        stake=10.0,
        urls=['https://book.invalid/ac'],
        live=False,
    )
    assert len(records) == 1
    assert records[0]['selection'] == 'Arsenal'
    assert records[0]['status'] == 'dry_run'


def test_execution_read_status_returns_what_the_venue_holds_for_one_bet(monkeypatch):
    """Reading status names one bet by its identity and returns what the venue holds for it."""
    venue = FakeVenue(prices={('Arsenal vs Chelsea', 'home_win', 'Arsenal'): 2.0})
    asyncio.run(
        venue.place(
            PlacementIntent(
                identity=BetIdentity('fake', 'Arsenal vs Chelsea', 'home_win', 'Arsenal'),
                stake=STAKE,
                min_price=2.0,
                value_bet='Arsenal vs Chelsea|home_win',
            ),
        ),
    )
    monkeypatch.setattr('sportsbet.mcp._server.build_venue', lambda ref: venue)
    records = _call(
        'execution_read_status',
        venue='venue.py:VENUE',
        match='Arsenal vs Chelsea',
        market='home_win',
        selection='Arsenal',
    )
    assert len(records) == 1
    assert records[0]['stake'] == STAKE
