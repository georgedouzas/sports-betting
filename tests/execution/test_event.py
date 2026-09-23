"""Test watching one event and placing the model's bet at its moment."""

import asyncio
import logging

import pandas as pd

from sportsbet.execution import PlacementStatus, execute_event
from tests.execution.conftest import EVENT, RecordingPlacer, StubBettor, StubSession

URL = 'https://book.invalid/arsenal-chelsea'
STAKE = 10.0


def _kickoff(loader):
    """Return the kickoff of the one fixture the loader carries."""
    X_fix, _, _ = loader.extract_fixtures_data()
    return X_fix.index[0]


def test_execute_event_places_one_bet_when_armed(event_dataloader, clock_and_wait):
    """An armed run against a matching URL places one receipt for the model's selection at the configured stake."""
    loader = event_dataloader()
    clock, wait = clock_and_wait(_kickoff(loader) - pd.Timedelta('1min'))
    placer = RecordingPlacer()
    receipts = asyncio.run(
        execute_event(
            EVENT,
            StubBettor(),
            loader,
            StubSession(),
            stake=STAKE,
            urls=[URL],
            live=True,
            placer=placer,
            poll=pd.Timedelta('5min'),
            clock=clock,
            wait=wait,
        ),
    )
    assert len(placer.calls) == 1
    assert len(receipts) == 1
    row = receipts.iloc[0]
    assert row['selection'] == 'Arsenal'
    assert row['market'] == 'home_win'
    assert row['stake'] == STAKE
    assert row['status'] == PlacementStatus.MATCHED_FULL.value


def test_execute_event_places_nothing_without_value(event_dataloader, clock_and_wait):
    """A model that finds no value places nothing and returns an empty frame."""
    loader = event_dataloader()
    clock, wait = clock_and_wait(_kickoff(loader) - pd.Timedelta('1min'))
    placer = RecordingPlacer()
    receipts = asyncio.run(
        execute_event(
            EVENT,
            StubBettor(value=False),
            loader,
            StubSession(),
            stake=STAKE,
            urls=[URL],
            live=True,
            placer=placer,
            poll=pd.Timedelta('5min'),
            clock=clock,
            wait=wait,
        ),
    )
    assert receipts.empty
    assert placer.calls == []


def test_execute_event_dry_run_stakes_nothing(event_dataloader, clock_and_wait):
    """A run that is not armed records a dry-run receipt, stakes nothing, and never calls the placer."""
    loader = event_dataloader()
    clock, wait = clock_and_wait(_kickoff(loader) - pd.Timedelta('1min'))
    placer = RecordingPlacer()
    receipts = asyncio.run(
        execute_event(
            EVENT,
            StubBettor(),
            loader,
            StubSession(),
            stake=STAKE,
            urls=[URL],
            live=False,
            placer=placer,
            poll=pd.Timedelta('5min'),
            clock=clock,
            wait=wait,
        ),
    )
    assert len(receipts) == 1
    assert receipts.iloc[0]['status'] == PlacementStatus.DRY_RUN.value
    assert receipts.iloc[0]['stake'] == 0.0
    assert placer.calls == []


def test_execute_event_stops_when_moment_passed(event_dataloader, clock_and_wait):
    """A clock past the betting moment places nothing and returns an empty frame."""
    loader = event_dataloader()
    clock, wait = clock_and_wait(_kickoff(loader) + pd.Timedelta('1min'))
    placer = RecordingPlacer()
    receipts = asyncio.run(
        execute_event(
            EVENT,
            StubBettor(),
            loader,
            StubSession(),
            stake=STAKE,
            urls=[URL],
            live=True,
            placer=placer,
            clock=clock,
            wait=wait,
        ),
    )
    assert receipts.empty
    assert placer.calls == []


def test_execute_event_stops_when_no_url_matches(event_dataloader, clock_and_wait):
    """URLs that do not carry the event place nothing and never authenticate."""
    loader = event_dataloader()
    clock, wait = clock_and_wait(_kickoff(loader) - pd.Timedelta('1min'))
    session = StubSession(page_event=None)
    placer = RecordingPlacer()
    receipts = asyncio.run(
        execute_event(
            EVENT,
            StubBettor(),
            loader,
            session,
            stake=STAKE,
            urls=[URL],
            live=True,
            placer=placer,
            clock=clock,
            wait=wait,
        ),
    )
    assert receipts.empty
    assert placer.calls == []
    assert session.authenticated is False


def test_execute_event_logs_status_and_decision(event_dataloader, clock_and_wait, caplog):
    """The log shows the event's status and the pre-place line before money moves."""
    caplog.set_level(logging.INFO, logger='sportsbet.execution')
    loader = event_dataloader()
    clock, wait = clock_and_wait(_kickoff(loader) - pd.Timedelta('1min'))
    asyncio.run(
        execute_event(
            EVENT,
            StubBettor(),
            loader,
            StubSession(),
            stake=STAKE,
            urls=[URL],
            live=False,
            poll=pd.Timedelta('5min'),
            clock=clock,
            wait=wait,
        ),
    )
    assert 'preplay' in caplog.text
    assert 'About to stake' in caplog.text


def test_execute_event_places_at_most_one_bet(event_dataloader, clock_and_wait):
    """Only one placement is made over the whole run, whatever the price does afterward."""
    loader = event_dataloader()
    clock, wait = clock_and_wait(_kickoff(loader) - pd.Timedelta('2min'))
    placer = RecordingPlacer()
    receipts = asyncio.run(
        execute_event(
            EVENT,
            StubBettor(),
            loader,
            StubSession(),
            stake=STAKE,
            urls=[URL],
            live=True,
            placer=placer,
            poll=pd.Timedelta('30s'),
            clock=clock,
            wait=wait,
        ),
    )
    assert len(placer.calls) == 1
    assert len(receipts) == 1
