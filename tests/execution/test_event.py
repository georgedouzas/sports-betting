"""Test running one preplay event from start to finish against the reference API venue."""

import asyncio
import json

import aiohttp
import pandas as pd
import pytest

from sportsbet.core import ExecutionError
from sportsbet.execution import PlacementStatus, RunOutcome, execute_event
from tests.execution.conftest import BOOK_KEY, EVENT, SOURCE_PRICE, STAKE, ReferenceVenue, StubBettor, run_event

VENUE_PRICE = 2.2
MOMENT_PRICE = 2.1
POLLS_IN_TWELVE_MINUTES = 3


def _run(loader, venue, clock, wait, tmp_path, **settings):
    """Run the event and return its record."""
    return run_event(loader, venue, clock, wait, tmp_path / 'ledger.jsonl', **settings)


def test_execute_event_unknown_event_execution_error(scripted, start, tmp_path):
    """Test a run for an event that is not a fixture raises an error naming it and contacts no venue."""
    loader = scripted()
    clock, _ = start(loader)
    venue = ReferenceVenue()
    with pytest.raises(ExecutionError, match='Liverpool vs Everton'):
        asyncio.run(execute_event('Liverpool vs Everton', StubBettor(), loader, venue, stake=STAKE, clock=clock))
    assert venue.logins == 0


def test_execute_event_other_fixtures_are_ignored(scripted, start, tmp_path):
    """Test a run that names one of several fixtures decides and places only on that match."""
    loader = scripted()
    other = loader.X.assign(home_team='Liverpool', away_team='Everton')
    loader.X, loader.O = pd.concat([other, loader.X]), pd.concat([loader.O, loader.O])
    clock, wait = start(loader)
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    record = _run(loader, venue, clock, wait, tmp_path, live=True)
    assert [intent.identity.match for intent in venue.placements] == [EVENT]
    assert record.outcome is RunOutcome.PLACED


def test_execute_event_authentication_fails_venue_failed(scripted, start, tmp_path):
    """Test a venue that refuses the login stops the run before waiting, and nothing is placed."""
    loader = scripted()
    clock, wait = start(loader, before_moment='10min')
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE}, blocked=True)
    record = _run(loader, venue, clock, wait, tmp_path, live=True)
    assert record.outcome is RunOutcome.VENUE_FAILED
    assert wait.waits == []
    assert venue.placements == []
    assert record.receipts.empty


def test_execute_event_moment_is_kickoff_minus_five_minutes(scripted, start, tmp_path):
    """Test the betting moment is the kickoff minus the default five minute lead."""
    loader = scripted()
    clock, wait = start(loader)
    record = _run(loader, ReferenceVenue({BOOK_KEY: VENUE_PRICE}), clock, wait, tmp_path)
    assert record.moment == loader.X.index[0] - pd.Timedelta('5min')


def test_execute_event_inplay_model_execution_error(scripted, start, tmp_path):
    """Test a dataloader fitted for an in-play moment is refused before any venue is contacted."""
    loader = scripted()
    loader.target_event_status_ = 'inplay'
    clock, wait = start(loader)
    venue = ReferenceVenue()
    with pytest.raises(ExecutionError, match='In-play execution is not supported'):
        _run(loader, venue, clock, wait, tmp_path)
    assert venue.logins == 0


def test_execute_event_negative_lead_execution_error(scripted, start, tmp_path):
    """Test a negative lead time is refused before any venue is contacted."""
    loader = scripted()
    clock, wait = start(loader)
    venue = ReferenceVenue()
    with pytest.raises(ExecutionError, match='must not be negative'):
        _run(loader, venue, clock, wait, tmp_path, lead=pd.Timedelta('-1min'))
    assert venue.logins == 0


def test_execute_event_started_too_late_moment_passed(scripted, start, tmp_path):
    """Test a run that starts after its betting moment places nothing and says the moment passed."""
    loader = scripted()
    clock, wait = start(loader, before_moment='-1min')
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    record = _run(loader, venue, clock, wait, tmp_path, live=True)
    assert record.outcome is RunOutcome.MOMENT_PASSED
    assert venue.logins == 0


def test_execute_event_polls_every_five_minutes_then_the_rest(scripted, start, tmp_path):
    """Test twelve minutes to the moment with a five minute poll waits five, five and then the two left."""
    loader = scripted()
    clock, wait = start(loader, before_moment='12min')
    record = _run(loader, ReferenceVenue({BOOK_KEY: VENUE_PRICE}), clock, wait, tmp_path)
    polls = record.steps[record.steps['step'] == 'poll']
    assert wait.waits == [300.0, 300.0, 120.0]
    assert len(polls) == POLLS_IN_TWELVE_MINUTES
    assert all(EVENT in detail and 'preplay' in detail and 'to the moment' in detail for detail in polls['detail'])


def test_execute_event_each_poll_shows_its_own_price(scripted, start, tmp_path):
    """Test a source price that changes between polls is shown as each poll read it."""
    loader = scripted({'home_win': 2.0}, {'home_win': 2.0}, {'home_win': 2.05}, {'home_win': 2.1})
    clock, wait = start(loader, before_moment='7min')
    record = _run(loader, ReferenceVenue({BOOK_KEY: VENUE_PRICE}), clock, wait, tmp_path)
    polls = record.steps.loc[record.steps['step'] == 'poll', 'detail'].tolist()
    assert polls[0].endswith('home_win 2.0.')
    assert polls[1].endswith('home_win 2.05.')


def test_execute_event_event_leaves_the_fixtures_event_unlisted(scripted, start, tmp_path):
    """Test a poll that finds the event gone stops the run, and nothing is placed."""
    loader = scripted({'home_win': SOURCE_PRICE}, 'unlisted')
    clock, wait = start(loader, before_moment='7min')
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    record = _run(loader, venue, clock, wait, tmp_path, live=True)
    assert record.outcome is RunOutcome.EVENT_UNLISTED
    assert venue.placements == []


def test_execute_event_value_bet_backs_the_selection(scripted, start, tmp_path):
    """Test a backed `home_win` on `Arsenal vs Chelsea` backs Arsenal with the configured stake."""
    loader = scripted()
    clock, wait = start(loader)
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    _run(loader, venue, clock, wait, tmp_path, live=True)
    (intent,) = venue.placements
    assert (intent.identity.selection, intent.identity.market, intent.stake) == ('Arsenal', 'home_win', STAKE)


def test_execute_event_decides_on_the_data_read_at_the_moment(scripted, start, tmp_path):
    """Test the bettor is applied to the odds read at the moment, not those read at the start."""
    loader = scripted({'home_win': 1.5}, {'home_win': 1.5}, {'home_win': MOMENT_PRICE})
    clock, wait = start(loader, before_moment='1min')
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    _run(loader, venue, clock, wait, tmp_path, live=True)
    assert venue.placements[0].min_price == MOMENT_PRICE


def test_execute_event_source_fails_at_the_moment_data_failed(scripted, start, tmp_path):
    """Test a read that fails at the moment places nothing and records why."""
    failure = aiohttp.ClientConnectionError('the source is down')
    loader = scripted({'home_win': SOURCE_PRICE}, {'home_win': SOURCE_PRICE}, failure)
    clock, wait = start(loader, before_moment='1min')
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    record = _run(loader, venue, clock, wait, tmp_path, live=True)
    assert record.outcome is RunOutcome.DATA_FAILED
    assert 'the source is down' in record.steps['detail'].iloc[-2]
    assert venue.placements == []


def test_execute_event_no_value_holds_no_receipt(scripted, start, tmp_path):
    """Test a bettor that backs nothing places nothing, says so, and the record holds no receipt."""
    loader = scripted()
    clock, wait = start(loader)
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    record = _run(loader, venue, clock, wait, tmp_path, bettor=StubBettor(value=False), live=True)
    assert record.outcome is RunOutcome.NO_VALUE
    assert record.receipts.empty
    assert 'No value bet' in ' '.join(record.steps['detail'])


def test_execute_event_dry_run_records_the_venue_price(scripted, start, tmp_path):
    """Test a dry run with value records one dry-run receipt at the venue's price and the stake, staking nothing."""
    loader = scripted()
    clock, wait = start(loader)
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    record = _run(loader, venue, clock, wait, tmp_path)
    (receipt,) = record.receipts.to_dict('records')
    assert (receipt['status'], receipt['price'], receipt['stake']) == ('dry_run', VENUE_PRICE, STAKE)
    assert record.outcome is RunOutcome.DRY_RUN
    assert venue.placements == []


def test_execute_event_dry_run_worse_venue_price_refused_price(scripted, start, tmp_path):
    """Test a dry run that finds the venue below the minimum records a refused-price receipt with both prices."""
    loader = scripted()
    clock, wait = start(loader)
    record = _run(loader, ReferenceVenue({BOOK_KEY: 1.9}), clock, wait, tmp_path)
    assert record.receipts.iloc[0]['status'] == PlacementStatus.REFUSED_PRICE.value
    assert '1.9' in record.receipts.iloc[0]['detail']
    assert str(SOURCE_PRICE) in record.receipts.iloc[0]['detail']


def test_execute_event_selection_not_offered_rejected(scripted, start, tmp_path):
    """Test a venue that does not offer the selection gives a rejected receipt saying so."""
    loader = scripted()
    clock, wait = start(loader)
    record = _run(loader, ReferenceVenue(), clock, wait, tmp_path, live=True)
    assert record.receipts.iloc[0]['status'] == PlacementStatus.REJECTED.value
    assert 'does not offer' in record.receipts.iloc[0]['detail']


def test_execute_event_armed_run_places_once_and_records_every_step(scripted, start, tmp_path):
    """Test an armed run logs the intended bet, places it once, and its record lists every step in order."""
    loader = scripted()
    clock, wait = start(loader, before_moment='6min')
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    record = _run(loader, venue, clock, wait, tmp_path, live=True)
    assert len(venue.placements) == 1
    assert record.outcome is RunOutcome.PLACED
    assert record.receipts.iloc[0]['venue_bet_id'] == 'REF-1'
    steps = record.steps['step'].tolist()
    wanted = ['start', 'venue_check', 'poll', 'poll', 'decision', 'venue_check', 'placement', 'receipt', 'outcome']
    assert [step for step in steps if step in wanted] == wanted
    placement = record.steps.loc[record.steps['step'] == 'placement', 'detail'].iloc[0]
    assert all(part in placement for part in ('Arsenal', 'home_win', str(STAKE), 'reference', str(SOURCE_PRICE)))


def test_execute_event_rerun_after_placed_already_placed(scripted, start, tmp_path):
    """Test a second armed run for a bet the ledger holds as placed places nothing and says it is already placed."""
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    for _ in range(2):
        loader = scripted()
        clock, wait = start(loader)
        record = _run(loader, venue, clock, wait, tmp_path, live=True)
    assert len(venue.placements) == 1
    assert record.outcome is RunOutcome.ALREADY_PLACED
    assert record.receipts.iloc[0]['status'] == PlacementStatus.ALREADY_PLACED.value


def test_execute_event_rerun_after_crash_refused_unconfirmed(scripted, start, tmp_path):
    """Test a rerun that finds its bet attempted with no receipt places nothing and says to check the venue."""
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    loader = scripted()
    clock, wait = start(loader)
    first = _run(loader, venue, clock, wait, tmp_path, live=True)
    ledger = tmp_path / 'ledger.jsonl'
    attempted = [line for line in ledger.read_text().splitlines() if json.loads(line)['status'] == 'attempted']
    ledger.write_text(attempted[0] + '\n')
    venue.placements.clear()
    loader = scripted()
    clock, wait = start(loader)
    record = _run(loader, venue, clock, wait, tmp_path, live=True)
    assert first.outcome is RunOutcome.PLACED
    assert venue.placements == []
    assert record.receipts.iloc[0]['status'] == PlacementStatus.REFUSED_UNCONFIRMED.value
    assert 'Check the venue' in record.receipts.iloc[0]['detail']


def test_execute_event_dry_runs_do_not_block_a_live_run(scripted, start, tmp_path):
    """Test a ledger that holds only dry runs for a bet lets an armed run place it."""
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    for live in (False, False, True):
        loader = scripted()
        clock, wait = start(loader)
        record = _run(loader, venue, clock, wait, tmp_path, live=live)
    assert record.outcome is RunOutcome.PLACED
    assert len(venue.placements) == 1


def test_execute_event_releases_the_venue_before_waiting(scripted, start, tmp_path):
    """Test the venue opened for the check is released before the wait, and again when the run ends."""
    loader = scripted()
    clock, wait = start(loader, before_moment='6min')
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    closes_while_waiting = []

    async def watching_wait(seconds):
        closes_while_waiting.append(venue.closes)
        await wait(seconds)

    _run(loader, venue, clock, watching_wait, tmp_path, live=True)
    assert closes_while_waiting[0] == 1
    assert venue.closes == 2  # noqa: PLR2004


def test_execute_event_error_while_placing_releases_the_venue(scripted, start, tmp_path):
    """Test an error raised by placing releases the venue before it propagates."""
    loader = scripted()
    clock, wait = start(loader)
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})

    async def failing_place(intent):
        msg = 'The connection dropped.'
        raise TimeoutError(msg)

    venue.place = failing_place
    with pytest.raises(TimeoutError):
        _run(loader, venue, clock, wait, tmp_path, live=True)
    assert venue.closes == 2  # noqa: PLR2004
