"""Test one run behaves the same with an API venue and a browser venue."""

import pytest

from sportsbet.execution import RunOutcome
from tests.execution.conftest import BOOK_KEY, ReferenceVenue, run_event

SITE_PRICE = 2.5
VARIABLE = 'SPORTSBET_TEST_TOKEN'


@pytest.mark.parametrize(('live', 'outcome'), [(False, RunOutcome.DRY_RUN), (True, RunOutcome.PLACED)])
def test_execute_event_same_run_two_venues(scripted, start, book, tmp_path, live, outcome):
    """Test the same run against the reference API venue and the served browser venue goes through the same steps."""
    records = []
    for venue in (ReferenceVenue({BOOK_KEY: SITE_PRICE}), book):
        loader = scripted()
        clock, wait = start(loader, before_moment='6min')
        records.append(run_event(loader, venue, clock, wait, tmp_path / f'{venue.key}.jsonl', live=live))
    api, browser = records
    assert api.steps['step'].tolist() == browser.steps['step'].tolist()
    assert api.outcome is browser.outcome is outcome
    assert list(api.receipts.columns) == list(browser.receipts.columns)
    assert api.receipts.iloc[0]['price'] == browser.receipts.iloc[0]['price'] == SITE_PRICE


def test_execute_event_secret_stays_out_of_the_ledger_and_record(scripted, start, tmp_path, monkeypatch):
    """Test an armed run with a venue reading a token from a named variable writes the token nowhere."""
    sentinel = 'sentinel-value-5f3a'
    monkeypatch.setenv(VARIABLE, sentinel)
    loader = scripted()
    clock, wait = start(loader)
    venue = ReferenceVenue({BOOK_KEY: SITE_PRICE}, token_env=VARIABLE)
    ledger = tmp_path / 'ledger.jsonl'
    record = run_event(loader, venue, clock, wait, ledger, live=True)
    assert record.outcome is RunOutcome.PLACED
    assert sentinel not in ledger.read_text()
    assert sentinel not in record.steps.to_csv()
    assert sentinel not in record.receipts.to_csv()
