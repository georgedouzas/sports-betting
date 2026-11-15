"""Test the single-event execution command drives the runner offline."""

import asyncio

import pandas as pd
import pytest

from sportsbet.cli import main
from sportsbet.execution import (
    BaseVenue,
    BetIdentity,
    EventRunRecord,
    PlacementIntent,
    PlacementReceipt,
    PlacementStatus,
    RunOutcome,
    build_receipts_frame,
    build_steps_frame,
)
from tests.conftest import FakeVenue

STAKE = 10.0


class MinimalVenue(BaseVenue):
    """A venue that implements only what a run needs."""

    key = 'minimal'

    async def authenticate(self):
        """Accept the login."""

    async def read_price(self, identity):
        """Offer nothing."""

    async def place(self, intent):
        """Place nothing."""


def _dry_run_record(event):
    """Return the record of a dry run that found a bet."""
    receipt = PlacementReceipt(
        identity=BetIdentity('stub', event, 'home_win', 'Arsenal'),
        status=PlacementStatus.DRY_RUN,
        stake=STAKE,
        price=2.2,
        value_bet=f'{event}|home_win',
        detail='Dry run, so nothing was staked.',
    )
    now = pd.Timestamp('2026-11-15 14:55', tz='UTC')
    steps = build_steps_frame([(now, 'decision', 'Backs Arsenal on home_win at 2.0 or better.')])
    return EventRunRecord(event, now, RunOutcome.DRY_RUN, steps, build_receipts_frame([receipt]))


def _invoke_run(cli_runner, tmp_path, *extra):
    """Invoke the run command with saved inputs and the given extra options."""
    dataloader = tmp_path / 'loader.pkl'
    dataloader.write_text('loader')
    bettor = tmp_path / 'model.pkl'
    bettor.write_text('model')
    options = ['--venue', 'venue.py:VENUE', '-d', str(dataloader), '-b', str(bettor), '--event', 'Arsenal vs Chelsea']
    return cli_runner.invoke(main, ['execution', 'run', *options, '--stake', str(STAKE), *extra])


@pytest.fixture
def recorded_run(monkeypatch):
    """Replace the run with one that records what it was told and returns a dry run record."""
    calls: dict[str, object] = {}

    async def fake_execute_event(event, bettor, loader, venue, *, stake, ledger_path, lead, poll, live):
        calls.update(event=event, stake=stake, ledger_path=ledger_path, lead=lead, poll=poll, live=live)
        return _dry_run_record(event)

    monkeypatch.setattr('sportsbet.cli._execution._build_venue', lambda ref: object())
    monkeypatch.setattr('sportsbet.cli._execution.load_dataloader', lambda path: object())
    monkeypatch.setattr('sportsbet.cli._execution.load_bettor', lambda path: object())
    monkeypatch.setattr('sportsbet.cli._execution.execute_event', fake_execute_event)
    return calls


def test_run_dry_run_prints_the_steps_and_receipt(cli_runner, recorded_run, tmp_path):
    """Test a dry run is the default, takes the default lead and poll, and prints the steps and the receipt."""
    result = _invoke_run(cli_runner, tmp_path)
    assert result.exit_code == 0, result.output
    assert recorded_run['live'] is False
    assert (recorded_run['lead'], recorded_run['poll']) == (pd.Timedelta('5min'), pd.Timedelta('5min'))
    assert recorded_run['ledger_path'] == 'sports-betting-ledger.jsonl'
    assert 'Steps' in result.output
    assert 'Arsenal' in result.output


def test_run_output_writes_the_steps_and_receipt(cli_runner, recorded_run, tmp_path):
    """Test `--output` writes the steps and the receipt under `sports-betting-data`."""
    result = _invoke_run(cli_runner, tmp_path, '--output', str(tmp_path / 'out'), '--lead', '10min')
    written = tmp_path / 'out' / 'sports-betting-data'
    assert result.exit_code == 0, result.output
    assert recorded_run['lead'] == pd.Timedelta('10min')
    assert pd.read_csv(written / 'run_steps.csv')['step'].tolist() == ['decision']
    assert pd.read_csv(written / 'receipts.csv')['status'].tolist() == ['dry_run']


def test_balance_unsupported_exits_with_one(cli_runner, monkeypatch):
    """Test a venue that does not read balances exits with status 1 and says so, rather than a traceback."""
    monkeypatch.setattr('sportsbet.cli._execution.build_venue', lambda ref: MinimalVenue())
    result = cli_runner.invoke(main, ['execution', 'balance', '--venue', 'venue.py:VENUE'])
    assert result.exit_code == 1, result.output
    assert 'does not read its balance' in result.output
    assert 'Traceback' not in result.output


def test_page_act_without_an_action_refused_before_the_browser(cli_runner, monkeypatch):
    """Test `page act` with nothing to do is refused before any browser is opened."""
    monkeypatch.setattr('sportsbet.cli._execution.build_venue', lambda ref: pytest.fail('a browser was opened'))
    result = cli_runner.invoke(main, ['execution', 'page', 'act', '--venue', 'venue.py:VENUE', '--url', 'http://x'])
    assert result.exit_code == 2  # noqa: PLR2004
    assert 'Name what to do' in result.output


def test_status_shows_what_the_venue_holds_for_one_bet(cli_runner, monkeypatch):
    """Status names one bet by its identity and prints what the venue holds for it."""
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
    monkeypatch.setattr('sportsbet.cli._execution.build_venue', lambda ref: venue)

    result = cli_runner.invoke(
        main,
        [
            'execution',
            'status',
            '--venue',
            'venue.py:VENUE',
            '--match',
            'Arsenal vs Chelsea',
            '--market',
            'home_win',
            '--selection',
            'Arsenal',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'What the venue holds' in result.output
