"""Test the ledger a run keeps, through the file it writes and the runs that read it."""

import json

import pytest

from sportsbet.execution import PlacementStatus, RunOutcome
from tests.execution.conftest import BOOK_KEY, STAKE, ReferenceVenue, identity_of, run_event

VENUE_PRICE = 2.2


def _write_line(path, status):
    """Write a ledger line holding the event's home win bet with a status."""
    line = {
        'ref': identity_of().ref_,
        'status': status,
        'written_at': '2026-11-15T14:55:00+00:00',
        'match': identity_of().match,
        'market': 'home_win',
        'selection': 'Arsenal',
        'stake': STAKE,
        'price': None,
        'venue_bet_id': None,
    }
    with path.open('a') as ledger:
        ledger.write(json.dumps(line) + '\n')


def _rerun(scripted, start, ledger):
    """Run the event armed against a venue that offers its price, and return the record and the venue."""
    loader = scripted()
    clock, wait = start(loader)
    venue = ReferenceVenue({BOOK_KEY: VENUE_PRICE})
    return run_event(loader, venue, clock, wait, ledger, live=True), venue


def test_execute_event_ledger_records_attempt_then_receipt(scripted, start, tmp_path):
    """Test an armed run writes the attempt before placing and the receipt after, and no secret."""
    ledger = tmp_path / 'ledger.jsonl'
    _rerun(scripted, start, ledger)
    statuses = [json.loads(line)['status'] for line in ledger.read_text().splitlines()]
    assert statuses == ['attempted', 'matched_full']


@pytest.mark.parametrize(
    ('held', 'outcome', 'status'),
    [
        ('matched_full', RunOutcome.ALREADY_PLACED, PlacementStatus.ALREADY_PLACED),
        ('attempted', RunOutcome.REFUSED, PlacementStatus.REFUSED_UNCONFIRMED),
        ('refused_unconfirmed', RunOutcome.REFUSED, PlacementStatus.REFUSED_UNCONFIRMED),
    ],
)
def test_execute_event_ledger_held_bet_is_not_placed_again(scripted, start, tmp_path, held, outcome, status):
    """Test a bet the ledger holds as placed, attempted or unconfirmed is not placed again."""
    ledger = tmp_path / 'ledger.jsonl'
    _write_line(ledger, held)
    record, venue = _rerun(scripted, start, ledger)
    assert venue.placements == []
    assert record.outcome is outcome
    assert record.receipts.iloc[0]['status'] == status.value


def test_execute_event_ledger_dry_run_lets_the_bet_go_on(scripted, start, tmp_path):
    """Test a bet the ledger holds only as a dry run is placed."""
    ledger = tmp_path / 'ledger.jsonl'
    _write_line(ledger, 'dry_run')
    record, venue = _rerun(scripted, start, ledger)
    assert record.outcome is RunOutcome.PLACED
    assert len(venue.placements) == 1


def test_execute_event_ledger_torn_last_line_is_skipped(scripted, start, tmp_path, caplog):
    """Test a last line a crash left torn is skipped with a log line, and the lines before it still count."""
    ledger = tmp_path / 'ledger.jsonl'
    _write_line(ledger, 'attempted')
    with ledger.open('a') as written:
        written.write('{"ref": "torn')
    with caplog.at_level('INFO', logger='sportsbet.execution'):
        record, venue = _rerun(scripted, start, ledger)
    assert 'torn' in caplog.text
    assert record.receipts.iloc[0]['status'] == PlacementStatus.REFUSED_UNCONFIRMED.value
    assert venue.placements == []
