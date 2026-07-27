"""Watch one betting event and place the model's bet at its moment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import pandas as pd

from ..core import NON_PREPLAY_EVENT_STATUSES, PREPLAY_EVENT_STATUSES
from ..evaluation import find_latest_odds_column
from ._base import (
    BetIdentity,
    ExecutionError,
    PlacementIntent,
    PlacementReceipt,
    PlacementStatus,
    build_receipts_frame,
)
from ._browser import BrowserSession, PageSnapshot
from ._schedule import find_betting_moment

if TYPE_CHECKING:
    from sportsbet.dataloaders import BaseDataLoader
    from sportsbet.evaluation import BaseBettor

logger = logging.getLogger('sportsbet.execution')

FALLBACK_PRICE = 1.01
DEFAULT_POLL = pd.Timedelta('30s')

Clock = Callable[[], pd.Timestamp]
Wait = Callable[[float], Awaitable[None]]
Placer = Callable[[PlacementIntent, BrowserSession], Awaitable[PlacementReceipt]]


def _now() -> pd.Timestamp:
    """Return the current time."""
    return pd.Timestamp.now(tz='UTC')


def _status(now: pd.Timestamp, kickoff: pd.Timestamp) -> str:
    """Return the event's lifecycle status at a time."""
    return PREPLAY_EVENT_STATUSES[0] if now < kickoff else NON_PREPLAY_EVENT_STATUSES[0]


def _price_of(odds: pd.DataFrame, market: str) -> float | None:
    """Return the latest price the odds carry for a market, or `None`."""
    column = find_latest_odds_column(list(odds.columns), market)
    if column is None:
        return None
    price = odds.iloc[0][column]
    return None if pd.isna(price) else float(price)


def _select_event(dataloader: BaseDataLoader, event: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Return the event's one-row features, odds, and kickoff, or raise if it is not a fixture."""
    X_fix, _, O_fix = dataloader.extract_fixtures_data()
    if not X_fix.empty and O_fix is not None:
        labels = X_fix['home_team'].astype(str) + ' vs ' + X_fix['away_team'].astype(str)
        mask = (labels == event).to_numpy()
        if mask.any():
            return X_fix[mask], O_fix[mask], X_fix.index[mask][0]
    msg = f'`{event}` is not among the fixtures, so there is nothing to act on.'
    raise ExecutionError(msg)


def _build_intent(
    venue_key: str,
    event: str,
    bettor: BaseBettor,
    X_event: pd.DataFrame,
    O_event: pd.DataFrame,
    stake: float,
) -> PlacementIntent | None:
    """Return the one bet the model backs on the event, or `None` when it finds no value."""
    markets = list(bettor.betting_markets_)
    value_bets = pd.DataFrame(bettor.bet(X_event, O_event), columns=markets)
    backed = [market for market in markets if value_bets.iloc[0][market]]
    if not backed:
        return None
    market = backed[0]
    selection = str(X_event.iloc[0]['home_team'])
    price = _price_of(O_event, market)
    return PlacementIntent(
        identity=BetIdentity(venue_key, event, market, selection),
        stake=stake,
        min_price=price if price is not None else FALLBACK_PRICE,
        value_bet=f'{event}|{market}',
    )


def _first_ref(snapshot: PageSnapshot) -> str:
    """Return the first actionable ref in a page snapshot."""
    for line in snapshot.yaml.splitlines():
        if '[ref=' in line:
            return line.split('[ref=')[1].split(']')[0]
    msg = 'The pinned control has no ref to act on.'
    raise ExecutionError(msg)


async def _default_placer(intent: PlacementIntent, session: BrowserSession) -> PlacementReceipt:
    """Drive the pinned stake and confirm controls to place one bet."""
    stake_control = await session.resolve('stake')
    await session.type(_first_ref(stake_control), str(intent.stake))
    confirm_control = await session.resolve('confirm')
    await session.click(_first_ref(confirm_control))
    return PlacementReceipt(
        identity=intent.identity,
        status=PlacementStatus.MATCHED_FULL,
        stake=intent.stake,
        price=intent.min_price,
        value_bet=intent.value_bet,
        placed_at=_now().to_pydatetime(),
    )


async def _match_url(session: BrowserSession, event: str, urls: list[str]) -> bool:
    """Explore the candidate URLs and pin the controls at the one that carries the event."""
    for url in urls:
        snapshot = await session.navigate(url)
        if event in snapshot.yaml:
            if session.fixed_ is not None:
                session.fix(event, dict(session.fixed_.locators))
            return True
    return False


async def _monitor(
    event: str,
    odds: pd.DataFrame,
    market: str,
    moment: pd.Timestamp,
    kickoff: pd.Timestamp,
    poll: pd.Timedelta,
    read_now: Clock,
    hold: Wait,
) -> None:
    """Log the event, its status, and its price on each poll until the betting moment."""
    while True:
        now = read_now()
        remaining = (moment - now).total_seconds()
        if remaining <= 0:
            return
        price = _price_of(odds, market)
        logger.info('%s is %s, price %s.', event, _status(now, kickoff), price if price is not None else 'unavailable')
        await hold(min(poll.total_seconds(), remaining))


async def execute_event(
    event: str,
    bettor: BaseBettor,
    dataloader: BaseDataLoader,
    session: BrowserSession,
    *,
    stake: float,
    urls: list[str],
    live: bool = False,
    placer: Placer | None = None,
    poll: pd.Timedelta = DEFAULT_POLL,
    clock: Clock | None = None,
    wait: Wait | None = None,
) -> pd.DataFrame:
    """Watch one event and place the model's bet at its moment, once.

    The function runs a fixed sequence of steps. It explores the URLs to find the event. It makes sure the session is
    logged in. It monitors the event until its betting moment and logs it. It applies the fitted bettor at that
    moment. It places one bet when the model finds value and the run is armed. A run stakes nothing unless you set
    `live`.

    Args:
        event:
            The one match to act on, as `'Home vs Away'`. Must be among the dataloader's fixtures.
        bettor:
            A fitted bettor. Decides whether to bet and the selection, never the stake.
        dataloader:
            Supplies the event's features and odds and carries the fitted moment.
        session:
            A headless browser session for the bookmaker.
        stake:
            The fixed amount to place.
        urls:
            Candidate bookmaker URLs, explored and matched to the event during setup.
        live:
            Arms the run. `False` is a no-stakes dry run that places nothing.
        placer:
            How to place on the site. Defaults to driving the pinned `stake` and `confirm` controls.
        poll:
            The interval between source polls while monitoring.
        clock:
            What returns the current time. Defaults to the real UTC now.
        wait:
            What waits for a number of seconds. Defaults to sleeping.

    Returns:
        receipts:
            A receipts frame with one row when a bet was placed and no rows otherwise.

    Raises:
        ExecutionError: If the event is not among the fixtures.
    """
    read_now = clock or _now
    hold = wait or asyncio.sleep
    place = placer or _default_placer

    if not await _match_url(session, event, urls):
        logger.info('No candidate URL carried %s, so nothing was placed.', event)
        return build_receipts_frame([])

    try:
        await session.authenticate()
    except ExecutionError:
        logger.info('Authentication for %s failed, so nothing was placed.', session.key)
        return build_receipts_frame([])

    X_event, O_event, kickoff = _select_event(dataloader, event)
    moment = find_betting_moment(dataloader, kickoff)
    if read_now() > moment:
        logger.info('The betting moment for %s has passed, so nothing was placed.', event)
        return build_receipts_frame([])

    await _monitor(event, O_event, next(iter(bettor.betting_markets_)), moment, kickoff, poll, read_now, hold)

    intent = _build_intent(session.key, event, bettor, X_event, O_event, stake)
    if intent is None:
        logger.info('No value bet found for %s, so nothing was placed.', event)
        return build_receipts_frame([])

    logger.info(
        'About to stake %s on %s %s at %s, price %s.',
        stake,
        intent.identity.selection,
        intent.identity.market,
        session.key,
        intent.min_price,
    )
    if live:
        receipt = await place(intent, session)
        logger.info('%s: %s.', event, receipt.status.value)
    else:
        receipt = PlacementReceipt(
            identity=intent.identity,
            status=PlacementStatus.DRY_RUN,
            value_bet=intent.value_bet,
            detail='Dry run, so nothing was staked.',
        )
        logger.info('%s: dry run, nothing staked.', event)
    return build_receipts_frame([receipt])
