"""Place the value bets of the upcoming matches, one match at a time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from math import isclose
from typing import TYPE_CHECKING

import pandas as pd

from ._base import (
    BaseVenue,
    ExecutionError,
    PlacementIntent,
    PlacementReceipt,
    PlacementStatus,
    _build_dry_run_receipts,
    build_receipts_frame,
)
from ._place import TOLERANCE, Staking, build_value_bet_intents

if TYPE_CHECKING:
    from sportsbet.dataloaders import BaseDataLoader
    from sportsbet.evaluation import BaseBettor

    from ._browser import BrowserSession

logger = logging.getLogger('sportsbet.execution')

Clock = Callable[[], pd.Timestamp]
Wait = Callable[[float], Awaitable[None]]
Placer = Callable[[PlacementIntent, 'BrowserSession'], Awaitable[PlacementReceipt]]
Scheduled = list[tuple[PlacementIntent, pd.Timestamp]]


def _now() -> pd.Timestamp:
    """Return the current time."""
    return pd.Timestamp.now(tz='UTC')


def find_betting_moment(dataloader: BaseDataLoader, kickoff: pd.Timestamp) -> pd.Timestamp:
    """Return when the bet goes on for a match.

    A live model bets at the kickoff plus the time into the match it was fitted for. Any other model bets at the
    kickoff.

    Args:
        dataloader:
            The dataloader the model was fitted on.
        kickoff:
            The kickoff of the match.

    Returns:
        moment:
            When the bet goes on.
    """
    if dataloader.target_event_status_ == 'inplay':
        return kickoff + dataloader.target_event_time_
    return kickoff


def select_feasible(
    dataloader: BaseDataLoader,
    fixtures: pd.DataFrame,
    now: pd.Timestamp,
    window: pd.Timedelta | None = None,
) -> pd.Series:
    """Return which matches the bet can still go on for.

    A match is feasible when its moment has not passed and, if a window is set, falls inside it.

    Args:
        dataloader:
            The dataloader the model was fitted on.
        fixtures:
            The upcoming matches, indexed by kickoff.
        now:
            The current time.
        window:
            How far ahead to reach. `None` reaches every upcoming match.

    Returns:
        mask:
            Which matches are feasible.
    """
    live = dataloader.target_event_status_ == 'inplay'
    moment = fixtures.index + dataloader.target_event_time_ if live else fixtures.index
    reachable = moment >= now
    if window is not None:
        reachable = reachable & (moment <= now + window)
    return pd.Series(reachable, index=fixtures.index)


def _scheduled(
    venue_key: str,
    dataloader: BaseDataLoader,
    bettor: BaseBettor,
    stake: Staking,
    now: pd.Timestamp,
    window: pd.Timedelta | None,
    seed: int,
) -> Scheduled:
    """Return the feasible value bets, each with its kickoff, ordered by their moment."""
    X_fix, _, O_fix = dataloader.extract_fixtures_data()
    if X_fix.empty or O_fix is None or O_fix.empty:
        return []
    mask = select_feasible(dataloader, X_fix, now, window).to_numpy()
    if not mask.any():
        return []
    reachable_X, reachable_O = X_fix[mask], O_fix[mask]
    kickoff_of = {
        f'{row["home_team"]} vs {row["away_team"]}': kickoff
        for kickoff, row in zip(reachable_X.index, reachable_X.to_dict('records'), strict=True)
    }
    intents = build_value_bet_intents(
        venue_key,
        bettor,
        reachable_X.reset_index(drop=True),
        reachable_O.reset_index(drop=True),
        stake,
    )
    paired = [(intent, kickoff_of.get(intent.identity.match, now)) for intent in intents]
    shuffled = pd.Series(range(len(paired))).sample(frac=1.0, random_state=seed).tolist()
    ordered = sorted(shuffled, key=lambda position: find_betting_moment(dataloader, paired[position][1]))
    return [paired[position] for position in ordered]


def _refused(intents: list[PlacementIntent], total: float, confirm_total: float | None) -> pd.DataFrame:
    """Return a receipt for every bet, none of them staked."""
    if confirm_total is None:
        detail = f'Nothing was staked. Pass confirm_total={total} to place these bets.'
        status = PlacementStatus.DRY_RUN
    else:
        detail = f'Nothing was staked. The quoted total is {total}, but {confirm_total} was passed back.'
        status = PlacementStatus.REFUSED_UNCONFIRMED
    return _build_dry_run_receipts(intents, status, detail)


async def execute(
    venue: BaseVenue | BrowserSession,
    dataloader: BaseDataLoader,
    bettor: BaseBettor,
    *,
    stake: Staking,
    placer: Placer | None = None,
    max_stake: float = 0.0,
    max_exposure: float = 0.0,
    confirm_total: float | None = None,
    window: pd.Timedelta | None = None,
    seed: int = 0,
    clock: Clock | None = None,
    wait: Wait | None = None,
) -> pd.DataFrame:
    """Place the value bets of the upcoming matches, one match at a time.

    Four steps, in order. It authenticates and stops if that fails. It selects the matches the bettor bets on and can
    still reach, sized by `stake`. It orders them by their moment. It places them in turn, waiting until each moment,
    and stakes nothing until `confirm_total` matches the total.

    A venue with an API is placed at by the library. A browser session is placed at by `placer`, which drives the site
    for one bet and returns its receipt.

    Args:
        venue:
            Where the bets go, a `BaseVenue` or a `BrowserSession`.
        dataloader:
            The dataloader the model was fitted on. It supplies the input data.
        bettor:
            The fitted bettor.
        stake:
            A number to stake the same on every bet, or a mapping keyed by `(match, market, selection)` to size each
            one, as computed offline.
        placer:
            Required for a browser session. It takes an intent and the session, drives the site, and returns a receipt.
        max_stake:
            The most to stake on one bet. Zero leaves it open.
        max_exposure:
            The most to have at stake at once. Zero leaves it open.
        confirm_total:
            The quoted total, passed back to place the bets. `None` stakes nothing.
        window:
            How long to keep placing. `None` reaches every upcoming match.
        seed:
            The seed for the random order.
        clock:
            What returns the current time. Defaults to now.
        wait:
            What waits for a number of seconds. Defaults to sleeping.

    Returns:
        receipts:
            What happened to each bet.

    Raises:
        ExecutionError: If authentication fails or a browser session has no `placer`.
    """
    read_now = clock or _now
    hold = wait or asyncio.sleep
    started = read_now()
    on_api = isinstance(venue, BaseVenue)
    if not on_api and placer is None:
        msg = 'A browser session needs a `placer`, since the library cannot place at a website.'
        raise ExecutionError(msg)

    try:
        await venue.authenticate()
    except Exception:
        logger.info('Authentication at %s failed, so nothing was placed.', venue.key)
        raise

    scheduled = _scheduled(venue.key, dataloader, bettor, stake, read_now(), window, seed)
    if not scheduled:
        logger.info('No feasible value bets in the upcoming matches.')
        return build_receipts_frame([])

    intents = [intent for intent, _ in scheduled]
    total = round(sum(intent.stake for intent in intents), 2)
    logger.info('%d feasible value bets, %s to stake in total.', len(intents), total)
    if confirm_total is None or not isclose(confirm_total, total, abs_tol=TOLERANCE):
        logger.info('Nothing staked. Pass confirm_total=%s to place these bets.', total)
        return _refused(intents, total, confirm_total)

    receipts: list[PlacementReceipt] = []
    for intent, kickoff in scheduled:
        ahead = (find_betting_moment(dataloader, kickoff) - read_now()).total_seconds()
        if ahead > 0:
            logger.info('Waiting %.0fs for %s.', ahead, intent.identity.match)
            await hold(ahead)
        if window is not None and read_now() - started > window:
            logger.info('Window closed, %d matches left unplaced.', len(scheduled) - len(receipts))
            break
        logger.info('Placing %s %s at %s.', intent.identity.match, intent.identity.market, venue.key)
        if isinstance(venue, BaseVenue):
            receipt = await venue.place(intent)
        else:
            assert placer is not None
            receipt = await placer(intent, venue)
        logger.info('%s: %s.', intent.identity.match, receipt.status.value)
        receipts.append(receipt)
        if receipt.status is PlacementStatus.BLOCKED:
            logger.info('The venue blocked automation, so placing stopped.')
            break
    return build_receipts_frame(receipts)
