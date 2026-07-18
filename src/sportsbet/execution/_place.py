"""Quote a batch and place it, refusing until the quoted total is passed back.

The bets go on one at a time with the exposure counted before each, which is what keeps the limit readable and stops a
retry from staking twice.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from math import isclose

import pandas as pd

from ..evaluation import BaseBettor
from ..evaluation._base import latest_odds_column
from ._base import (
    STAKED,
    BaseVenue,
    BetIdentity,
    ExposureLimits,
    PlacementIntent,
    PlacementQuote,
    PlacementReceipt,
    PlacementStatus,
    VenueBlockedError,
    receipts_frame,
)

TOLERANCE = 0.005
FALLBACK_PRICE = 1.01

Staking = float | Mapping[tuple[str, str, str], float]


def _stake_of(stake: Staking, match: str, market: str, selection: str) -> float | None:
    """Return what to stake on an event, or `None` to skip it.

    A number stakes the same on every bet. A mapping keyed by the event stakes per event, and an event it omits is not
    bet on.
    """
    if isinstance(stake, Mapping):
        return stake.get((match, market, selection))
    return stake


def value_bet_intents(
    venue: str,
    bettor: BaseBettor,
    X_fix: pd.DataFrame,
    O_fix: pd.DataFrame,
    stake: Staking,
) -> list[PlacementIntent]:
    """Return an intent for each value bet a model found.

    The minimum price of each one is the price the value bet was computed at, since below it the bet is no longer a
    value bet.

    Args:
        venue:
            What the venue is called.
        bettor:
            A fitted bettor.
        X_fix:
            The upcoming matches.
        O_fix:
            The odds of the upcoming matches.
        stake:
            A number to stake the same on every value bet, or a mapping keyed by `(match, market, selection)` to stake
            per event. A mapping stakes only the events it holds, so sizing computed offline drops the rest.

    Returns:
        intents:
            The bets the model means to place.
    """
    markets = list(bettor.betting_markets_)
    odds_columns = {market: latest_odds_column(list(O_fix.columns), market) for market in markets}
    value_bets = pd.DataFrame(bettor.bet(X_fix, O_fix), columns=markets)
    intents = []
    for position in range(len(value_bets)):
        game = X_fix.iloc[position]
        match = f'{game["home_team"]} vs {game["away_team"]}'
        selection = str(game['home_team'])
        for market in markets:
            if not value_bets.iloc[position][market]:
                continue
            amount = _stake_of(stake, match, market, selection)
            if not amount:
                continue
            column = odds_columns[market]
            price = O_fix.iloc[position][column] if column is not None else None
            intents.append(
                PlacementIntent(
                    identity=BetIdentity(venue, match, market, selection),
                    stake=amount,
                    min_price=float(price) if price is not None and not pd.isna(price) else FALLBACK_PRICE,
                    value_bet=f'{match}|{market}',
                ),
            )
    return intents


async def quote(venue: BaseVenue, intents: list[PlacementIntent], limits: ExposureLimits) -> PlacementQuote:
    """Return what is about to be staked, before anything is.

    Args:
        venue:
            Where the bets would go.
        intents:
            The bets to place.
        limits:
            The ceilings the batch answers to.

    Returns:
        quoted:
            Every bet, its stake and its price, and the totals to pass back to place them.
    """
    _, open_exposure = await venue.read_balance()
    total_stake = round(sum(intent.stake for intent in intents), 2)
    return PlacementQuote(
        intents=list(intents),
        total_stake=total_stake,
        total_exposure=round(total_stake + open_exposure, 2),
        quoted_at=pd.Timestamp.now(tz='UTC').to_pydatetime(),
    )


def _confirmed(quoted: PlacementQuote, confirm_stake: float | None, confirm_exposure: float | None) -> bool:
    """Return whether the caller passed back the figures that were quoted."""
    if confirm_stake is None or confirm_exposure is None:
        return False
    return isclose(confirm_stake, quoted.total_stake, abs_tol=TOLERANCE) and isclose(
        confirm_exposure,
        quoted.total_exposure,
        abs_tol=TOLERANCE,
    )


def _unconfirmed(quoted: PlacementQuote, confirm_stake: float | None, confirm_exposure: float | None) -> str:
    """Return what to say when the figures do not match, which names the real ones."""
    if confirm_stake is None or confirm_exposure is None:
        return (
            f'Nothing was staked. To place these bets, pass back the quoted stake of {quoted.total_stake} '
            f'and the quoted exposure of {quoted.total_exposure}.'
        )
    return (
        f'Nothing was staked. The quoted stake is {quoted.total_stake} and the quoted exposure is '
        f'{quoted.total_exposure}, but {confirm_stake} and {confirm_exposure} were passed back.'
    )


def _dry_run(quoted: PlacementQuote, detail: str, status: PlacementStatus) -> pd.DataFrame:
    """Return a receipt for every bet, none of them staked."""
    return receipts_frame(
        [
            PlacementReceipt(
                identity=intent.identity,
                status=status,
                price=intent.min_price,
                value_bet=intent.value_bet,
                detail=detail,
            )
            for intent in quoted.intents
        ],
    )


def _over_limit(intent: PlacementIntent, limits: ExposureLimits, running: float) -> str | None:
    """Return which ceiling a bet would breach, if it would breach one."""
    if limits.max_stake_per_bet and intent.stake > limits.max_stake_per_bet:
        return f'The stake of {intent.stake} is over the maximum stake per bet of {limits.max_stake_per_bet}.'
    if limits.max_total_exposure and running + intent.stake > limits.max_total_exposure:
        return (
            f'The stake of {intent.stake} would take the exposure to {round(running + intent.stake, 2)}, '
            f'over the maximum total exposure of {limits.max_total_exposure}.'
        )
    return None


async def _price_of(venue: BaseVenue, intent: PlacementIntent) -> float | None:
    """Return what the venue is offering for a bet right now."""
    markets = await venue.list_markets([intent.identity.match])
    if markets.empty:
        return None
    wanted = markets[
        (markets['market'] == intent.identity.market) & (markets['selection'] == intent.identity.selection)
    ]
    if wanted.empty:
        return None
    price = wanted.iloc[0]['price']
    return None if pd.isna(price) else float(price)


async def place(
    venue: BaseVenue,
    quoted: PlacementQuote,
    limits: ExposureLimits,
    confirm_stake: float | None = None,
    confirm_exposure: float | None = None,
) -> pd.DataFrame:
    """Place a quoted batch, staking nothing unless the quoted figures are passed back exactly.

    Args:
        venue:
            Where the bets go.
        quoted:
            What `quote` returned.
        limits:
            The ceilings the batch answers to.
        confirm_stake:
            The quoted stake, passed back to place the bets.
        confirm_exposure:
            The quoted exposure, passed back to place the bets.

    Returns:
        receipts:
            What happened to each bet.
    """
    if limits.killed:
        return _dry_run(quoted, 'The kill switch is on, so nothing was staked.', PlacementStatus.REFUSED_KILLED)
    if not _confirmed(quoted, confirm_stake, confirm_exposure):
        detail = _unconfirmed(quoted, confirm_stake, confirm_exposure)
        status = (
            PlacementStatus.DRY_RUN
            if confirm_stake is None and confirm_exposure is None
            else PlacementStatus.REFUSED_UNCONFIRMED
        )
        return _dry_run(quoted, detail, status)
    _, open_exposure = await venue.read_balance()
    running = open_exposure
    receipts: list[PlacementReceipt] = []
    for intent in quoted.intents:
        receipt = await _place_one(venue, intent, limits, running)
        receipts.append(receipt)
        if receipt.status is PlacementStatus.BLOCKED:
            receipts.extend(_stopped(quoted, intent))
            break
        if receipt.status in STAKED:
            running = round(running + receipt.stake, 2)
    return receipts_frame(receipts)


def _stopped(quoted: PlacementQuote, reached: PlacementIntent) -> list[PlacementReceipt]:
    """Return refusals for the bets a block stopped."""
    rest = quoted.intents[quoted.intents.index(reached) + 1 :]
    return [
        PlacementReceipt(
            identity=intent.identity,
            status=PlacementStatus.BLOCKED,
            value_bet=intent.value_bet,
            detail='The venue blocked automated access, so placing stopped.',
        )
        for intent in rest
    ]


async def _place_one(
    venue: BaseVenue,
    intent: PlacementIntent,
    limits: ExposureLimits,
    running: float,
) -> PlacementReceipt:
    """Place one bet, refusing before the venue is reached where a rule says to."""
    if limits.killed:
        return PlacementReceipt(
            identity=intent.identity,
            status=PlacementStatus.REFUSED_KILLED,
            value_bet=intent.value_bet,
            detail='The kill switch is on, so nothing was staked.',
        )
    breached = _over_limit(intent, limits, running)
    if breached is not None:
        return PlacementReceipt(
            identity=intent.identity,
            status=PlacementStatus.REFUSED_LIMIT,
            value_bet=intent.value_bet,
            detail=breached,
        )
    try:
        price = await _price_of(venue, intent)
        if price is not None and price < intent.min_price:
            return PlacementReceipt(
                identity=intent.identity,
                status=PlacementStatus.REFUSED_PRICE,
                price=price,
                value_bet=intent.value_bet,
                detail=f'The price of {price} is below the minimum of {intent.min_price}.',
            )
        await asyncio.sleep(getattr(venue, 'min_interval', 0.0))
        return await venue.place(intent)
    except VenueBlockedError as blocked:
        return PlacementReceipt(
            identity=intent.identity,
            status=PlacementStatus.BLOCKED,
            value_bet=intent.value_bet,
            detail=str(blocked),
        )
