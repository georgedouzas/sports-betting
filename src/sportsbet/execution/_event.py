"""Watch one betting event and place the model's bet at its moment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiohttp
import pandas as pd

from ..core import ExecutionError
from ..dataloaders import BaseDataLoader
from ..evaluation import BaseBettor
from ._base import (
    BaseVenue,
    EventRunRecord,
    PlacementIntent,
    PlacementReceipt,
    PlacementStatus,
    RunOutcome,
    build_receipts_frame,
    build_steps_frame,
)
from ._decision import _read_source_price, decide_bet
from ._ledger import PLACED_STATUSES, UNSETTLED_STATUSES, _Ledger
from ._schedule import DEFAULT_LEAD, find_betting_moment

_logger = logging.getLogger('sportsbet.execution')

DEFAULT_POLL = pd.Timedelta('5min')
DEFAULT_LEDGER = 'sports-betting-ledger.jsonl'
READ_ERRORS = (aiohttp.ClientError, OSError)

_Clock = Callable[[], pd.Timestamp]
_Wait = Callable[[float], Awaitable[None]]


def _read_now() -> pd.Timestamp:
    """Return the current time."""
    return pd.Timestamp.now(tz='UTC')


class _Steps:
    """The steps of a run, each logged as it is recorded."""

    def __init__(self: _Steps, read_now: _Clock) -> None:
        self.read_now = read_now
        self.recorded: list[tuple[pd.Timestamp, str, str]] = []

    def add(self: _Steps, step: str, detail: str) -> None:
        """Record a step and log it."""
        self.recorded.append((self.read_now(), step, detail))
        _logger.info('%s: %s', step, detail)


def _read_event(dataloader: BaseDataLoader, event: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Return the event's one-row features, odds, and kickoff, read afresh, or raise if it is not a fixture."""
    X_fix, _, O_fix = dataloader.extract_fixtures_data()
    if not X_fix.empty and O_fix is not None:
        labels = X_fix['home_team'].astype(str) + ' vs ' + X_fix['away_team'].astype(str)
        mask = (labels == event).to_numpy()
        if mask.any():
            return X_fix[mask], O_fix[mask], X_fix.index[mask][0]
    msg = f'`{event}` is not among the fixtures, so there is nothing to act on.'
    raise ExecutionError(msg)


def _describe_prices(odds: pd.DataFrame, markets: list[str]) -> str:
    """Return the source's latest price of each betting market, as a phrase."""
    prices = [(market, _read_source_price(odds, market)) for market in markets]
    shown = [f'{market} {price}' for market, price in prices if price is not None]
    return ', '.join(shown) if shown else 'no prices'


def _build_receipt(
    intent: PlacementIntent,
    status: PlacementStatus,
    detail: str,
    *,
    stake: float = 0.0,
    price: float | None = None,
) -> PlacementReceipt:
    """Return a receipt the run wrote itself, for a bet the venue was not asked to place."""
    return PlacementReceipt(
        identity=intent.identity,
        status=status,
        stake=stake,
        price=price,
        value_bet=intent.value_bet,
        detail=detail,
    )


def _check_ledger(ledger: _Ledger, intent: PlacementIntent) -> PlacementReceipt | None:
    """Return the receipt that stops a rerun, when the ledger already holds the bet."""
    held = ledger.read_status(intent.identity.ref_)
    if held in PLACED_STATUSES:
        detail = f'The ledger holds this bet as {held}, so it was not placed again.'
        return _build_receipt(intent, PlacementStatus.ALREADY_PLACED, detail)
    if held in UNSETTLED_STATUSES:
        detail = (
            f'The ledger holds this bet as {held} with no confirmation. Check the venue for the bet before running '
            'again, since it may have gone on.'
        )
        return _build_receipt(intent, PlacementStatus.REFUSED_UNCONFIRMED, detail)
    return None


async def _act_at_venue(
    venue: BaseVenue,
    intent: PlacementIntent,
    ledger: _Ledger,
    steps: _Steps,
    *,
    live: bool,
) -> PlacementReceipt:
    """Read the venue's price, then record the dry run or place the bet once."""
    price = await venue.read_price(intent.identity)
    if price is None:
        steps.add('price', f'{venue.key} does not offer {intent.identity.selection} on {intent.identity.market}.')
        detail = 'The venue does not offer this selection, so nothing was placed.'
        return _build_receipt(intent, PlacementStatus.REJECTED, detail)
    steps.add('price', f'{venue.key} offers {price}, and the minimum is {intent.min_price}.')
    if price < intent.min_price:
        detail = f'The venue offers {price}, below the minimum of {intent.min_price}, so nothing was placed.'
        return _build_receipt(intent, PlacementStatus.REFUSED_PRICE, detail, price=price)
    if not live:
        detail = 'Dry run, so nothing was staked.'
        receipt = _build_receipt(intent, PlacementStatus.DRY_RUN, detail, stake=intent.stake, price=price)
        ledger.record_receipt(receipt, steps.read_now())
        return receipt
    identity = intent.identity
    steps.add(
        'placement',
        f'About to stake {intent.stake} on {identity.selection} ({identity.market}) in {identity.match} at '
        f'{venue.key}, price {intent.min_price} or better.',
    )
    ledger.record_attempt(intent, steps.read_now())
    receipt = await venue.place(intent)
    ledger.record_receipt(receipt, steps.read_now())
    return receipt


async def _check_venue(venue: BaseVenue, steps: _Steps, when: str) -> bool:
    """Return whether the venue accepted the login, recording the answer."""
    try:
        await venue.authenticate()
    except ExecutionError as error:
        steps.add('venue_check', f'{venue.key} failed {when}: {error}')
        return False
    steps.add('venue_check', f'{venue.key} accepted the login {when}.')
    return True


async def _wait_for_moment(
    dataloader: BaseDataLoader,
    event: str,
    moment: pd.Timestamp,
    markets: list[str],
    poll: pd.Timedelta,
    steps: _Steps,
    hold: _Wait,
) -> RunOutcome | None:
    """Read and log the event on each poll until the moment, returning an outcome only when the run must stop."""
    while (remaining := moment - steps.read_now()) > pd.Timedelta(0):
        try:
            _, O, _ = _read_event(dataloader, event)
            left = str(remaining.floor('s')).removeprefix('0 days ')
            steps.add('poll', f'{event} is preplay, {left} to the moment, {_describe_prices(O, markets)}.')
        except ExecutionError:
            steps.add('poll', f'{event} is no longer among the fixtures, so nothing was placed.')
            return RunOutcome.EVENT_UNLISTED
        except READ_ERRORS as error:
            steps.add('poll', f'Reading {event} failed, so the run waits for the next poll: {error}')
        await hold(min(poll, remaining).total_seconds())
    return None


def _read_at_moment(
    dataloader: BaseDataLoader,
    event: str,
    steps: _Steps,
) -> tuple[pd.DataFrame, pd.DataFrame] | RunOutcome:
    """Return the event's data read at the moment, or the outcome when it cannot be read."""
    try:
        X, O, _ = _read_event(dataloader, event)
    except ExecutionError:
        steps.add('decision', f'{event} is no longer among the fixtures, so nothing was placed.')
        return RunOutcome.EVENT_UNLISTED
    except READ_ERRORS as error:
        steps.add('decision', f'Reading {event} at the moment failed, so nothing was placed: {error}')
        return RunOutcome.DATA_FAILED
    return X, O


async def _place_at_moment(
    venue: BaseVenue,
    intent: PlacementIntent,
    ledger: _Ledger,
    steps: _Steps,
    *,
    live: bool,
) -> PlacementReceipt | RunOutcome:
    """Open the venue, act on the intended bet and release the venue, whatever happens."""
    try:
        if not await _check_venue(venue, steps, 'at the moment'):
            return RunOutcome.VENUE_FAILED
        receipt = await _act_at_venue(venue, intent, ledger, steps, live=live)
    finally:
        await venue.close()
    steps.add('receipt', f'{receipt.status.value}: {receipt.detail}')
    return receipt


def _find_outcome(receipt: PlacementReceipt) -> RunOutcome:
    """Return how a run that reached the venue ended."""
    if receipt.status.value in PLACED_STATUSES:
        return RunOutcome.PLACED
    if receipt.status is PlacementStatus.DRY_RUN:
        return RunOutcome.DRY_RUN
    if receipt.status is PlacementStatus.ALREADY_PLACED:
        return RunOutcome.ALREADY_PLACED
    return RunOutcome.REFUSED


async def _run_after_start(
    event: str,
    bettor: BaseBettor,
    dataloader: BaseDataLoader,
    venue: BaseVenue,
    moment: pd.Timestamp,
    settings: dict,
    steps: _Steps,
) -> tuple[RunOutcome, PlacementReceipt | None]:
    """Check the venue, wait for the moment, decide and act, returning how the run ended and its receipt."""
    try:
        if not await _check_venue(venue, steps, 'at the start'):
            return RunOutcome.VENUE_FAILED, None
    finally:
        await venue.close()
    markets = [str(market) for market in bettor.betting_markets_]
    stopped = await _wait_for_moment(dataloader, event, moment, markets, settings['poll'], steps, settings['wait'])
    if stopped is not None:
        return stopped, None
    read = _read_at_moment(dataloader, event, steps)
    if isinstance(read, RunOutcome):
        return read, None
    return await _decide_and_place(event, bettor, read, venue, settings, steps)


async def _decide_and_place(
    event: str,
    bettor: BaseBettor,
    read: tuple[pd.DataFrame, pd.DataFrame],
    venue: BaseVenue,
    settings: dict,
    steps: _Steps,
) -> tuple[RunOutcome, PlacementReceipt | None]:
    """Apply the bettor to the data read at the moment, check the ledger and act at the venue."""
    intent, reason = decide_bet(event, bettor, *read, venue_key=venue.key, stake=settings['stake'])
    steps.add('decision', reason)
    if intent is None:
        return RunOutcome.NO_VALUE, None
    ledger = _Ledger(settings['ledger_path'])
    held = _check_ledger(ledger, intent)
    if held is not None:
        steps.add('placement', held.detail)
        return _find_outcome(held), held
    placed = await _place_at_moment(venue, intent, ledger, steps, live=settings['live'])
    if isinstance(placed, RunOutcome):
        return placed, None
    return _find_outcome(placed), placed


async def execute_event(
    event: str,
    bettor: BaseBettor,
    dataloader: BaseDataLoader,
    venue: BaseVenue,
    *,
    stake: float,
    ledger_path: str | Path = DEFAULT_LEDGER,
    lead: pd.Timedelta = DEFAULT_LEAD,
    poll: pd.Timedelta = DEFAULT_POLL,
    live: bool = False,
    clock: _Clock | None = None,
    wait: _Wait | None = None,
) -> EventRunRecord:
    """Watch one preplay event and place the model's bet at its moment, once.

    The run checks the venue, then waits for the betting moment, which is the kickoff minus the lead time. On each
    poll it reads the event's data again and logs it. At the moment it reads the data once more, applies the bettor,
    reads the venue's price, and places one bet when the model finds value and the run is armed. A run stakes nothing
    unless you set `live`. A ledger file records each bet before and after it is placed, so a rerun never places it
    twice.

    Args:
        event:
            The one match to act on, as `'Home vs Away'`. Must be among the dataloader's fixtures.
        bettor:
            A fitted bettor. Decides whether to bet and the selection, never the stake.
        dataloader:
            Supplies the event's features and odds, read afresh on each poll, and carries the fitted moment.
        venue:
            Where the bet goes on, an API venue or a browser venue.
        stake:
            The fixed amount to place.
        ledger_path:
            The file that records each bet, kept across runs. Use one file for every run, so a rerun finds the bet.
        lead:
            How long before the kickoff the bet goes on.
        poll:
            The interval between reads of the event's data while waiting.
        live:
            Arms the run. `False` is a dry run that reads the venue's price and stakes nothing.
        clock:
            What returns the current time. Defaults to the real UTC now.
        wait:
            What waits for a number of seconds. Defaults to sleeping.

    Returns:
        record:
            The steps the run went through, how it ended, and the receipt when there is one.

    Raises:
        ExecutionError: If the event is not among the fixtures, the model is in-play, or the lead time is negative.
    """
    steps = _Steps(clock or _read_now)
    _, _, kickoff = _read_event(dataloader, event)
    moment = find_betting_moment(dataloader, kickoff, lead)
    steps.add('start', f'{event}, {"armed" if live else "dry run"}, stake {stake}, betting at {moment}.')
    receipt: PlacementReceipt | None = None
    if steps.read_now() > moment:
        steps.add('moment', f'The betting moment {moment} has passed, so nothing was placed.')
        outcome = RunOutcome.MOMENT_PASSED
    else:
        settings = {
            'stake': stake,
            'ledger_path': ledger_path,
            'poll': poll,
            'live': live,
            'wait': wait or asyncio.sleep,
        }
        outcome, receipt = await _run_after_start(event, bettor, dataloader, venue, moment, settings, steps)
    steps.add('outcome', outcome.value)
    receipts = build_receipts_frame([receipt] if receipt is not None else [])
    return EventRunRecord(event, moment, outcome, build_steps_frame(steps.recorded), receipts)
