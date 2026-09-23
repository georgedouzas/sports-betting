"""Define the venue base class and the types a bet placement produces."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import blake2s
from typing import Annotated

import pandas as pd
import pandera.pandas as pa


class ExecutionError(Exception):
    """Raised when placing cannot go ahead."""


class CancellationUnsupportedError(ExecutionError):
    """Raised when a venue is asked to cancel and cannot."""


class VenueBlockedError(ExecutionError):
    """Raised when a venue blocks automated access."""


class PlacementStatus(StrEnum):
    """What became of an intended bet."""

    DRY_RUN = 'dry_run'
    ACCEPTED = 'accepted'
    MATCHED_FULL = 'matched_full'
    MATCHED_PARTIAL = 'matched_partial'
    ALREADY_PLACED = 'already_placed'
    REFUSED_LIMIT = 'refused_limit'
    REFUSED_PRICE = 'refused_price'
    REFUSED_UNCONFIRMED = 'refused_unconfirmed'
    REFUSED_KILLED = 'refused_killed'
    BLOCKED = 'blocked'
    REJECTED = 'rejected'


_REF_BYTES = 16


@dataclass(frozen=True)
class BetIdentity:
    """What makes two bets the same bet.

    Two bets with the same venue, match, market and selection are the same bet.

    Args:
        venue: The venue the bet is placed at.
        match: The match it is on.
        market: The market within the match.
        selection: The side backed.

    Examples:
        >>> from sportsbet.execution import BetIdentity
        >>> identity = BetIdentity('exchange', 'Arsenal vs Chelsea', 'home_win', 'Arsenal')
        >>> identity.ref_
        '2f3152b1a4f9f6333904f3624320d921'
        >>> BetIdentity('exchange', 'Arsenal vs Chelsea', 'home_win', 'Arsenal').ref_ == identity.ref_
        True
    """

    venue: str
    match: str
    market: str
    selection: str

    @property
    def ref_(self: BetIdentity) -> str:
        """The reference a venue carries for this bet."""
        seed = f'{self.venue}|{self.match}|{self.market}|{self.selection}'
        return blake2s(seed.encode(), digest_size=_REF_BYTES).hexdigest()


@dataclass(frozen=True)
class PlacementIntent:
    """What the caller means to do.

    Args:
        identity: The bet to place.
        stake: The amount to stake.
        min_price: The lowest price to accept.
        value_bet: The value bet the intent came from.
    """

    identity: BetIdentity
    stake: float
    min_price: float
    value_bet: str = ''


@dataclass(frozen=True)
class PlacementReceipt:
    """What happened to an intended bet.

    Args:
        identity: The bet.
        status: What became of it.
        stake: The amount staked.
        price: The price it was matched at.
        venue_bet_id: The venue's identifier for the bet.
        value_bet: The value bet it came from.
        placed_at: When it was placed.
        detail: A human-readable note.
    """

    identity: BetIdentity
    status: PlacementStatus
    stake: float = 0.0
    price: float | None = None
    venue_bet_id: str | None = None
    value_bet: str = ''
    placed_at: datetime | None = None
    detail: str = ''


class _PlacementReceiptSchema(pa.DataFrameModel):
    """The receipts a placement returns."""

    ref: str = pa.Field()
    venue: str = pa.Field()
    match: str = pa.Field()
    market: str = pa.Field()
    selection: str = pa.Field()
    status: str = pa.Field()
    stake: float = pa.Field(ge=0.0)
    price: float = pa.Field(nullable=True)
    venue_bet_id: str = pa.Field(nullable=True)
    value_bet: str = pa.Field(nullable=True)
    placed_at: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = pa.Field(nullable=True)
    detail: str = pa.Field(nullable=True)

    class Config:
        """Allow no column beyond the ones named, and coerce the dtypes."""

        strict = True
        coerce = True


def build_receipts_frame(receipts: list[PlacementReceipt]) -> pd.DataFrame:
    """Return receipts as a validated frame.

    Args:
        receipts: The receipts to frame.

    Returns:
        frame: The receipts as a validated frame.
    """
    records = [
        {
            'ref': receipt.identity.ref_,
            'venue': receipt.identity.venue,
            'match': receipt.identity.match,
            'market': receipt.identity.market,
            'selection': receipt.identity.selection,
            'status': receipt.status.value,
            'stake': receipt.stake,
            'price': receipt.price,
            'venue_bet_id': receipt.venue_bet_id,
            'value_bet': receipt.value_bet,
            'placed_at': receipt.placed_at,
            'detail': receipt.detail,
        }
        for receipt in receipts
    ]
    frame = pd.DataFrame.from_records(records, columns=list(_PlacementReceiptSchema.to_schema().columns))
    frame['placed_at'] = pd.to_datetime(frame['placed_at'], utc=True)
    result: pd.DataFrame = _PlacementReceiptSchema.validate(frame)
    return result


class BaseVenue(abc.ABC):
    """A place where a user holds an account and can back a selection.

    A venue that implements this contract places a bet once and only once for each identity.
    """

    key: str = ''
    can_cancel: bool = False

    @abc.abstractmethod
    async def authenticate(self: BaseVenue) -> None:
        """Authenticate, reading the secret from the variable the venue names."""

    @abc.abstractmethod
    async def list_markets(self: BaseVenue, matches: list[str]) -> pd.DataFrame:
        """Return the markets on offer for the given matches, with their current prices.

        Args:
            matches: Matches to read the markets of.

        Returns:
            One row per market on offer, with its current price.
        """

    @abc.abstractmethod
    async def read_balance(self: BaseVenue) -> tuple[float, float]:
        """Return the balance and the exposure currently open.

        Returns:
            The balance and the exposure, in that order.
        """

    @abc.abstractmethod
    async def place(self: BaseVenue, intent: PlacementIntent) -> PlacementReceipt:
        """Place one bet, once and only once for its identity.

        Args:
            intent: The bet to place, carrying the identity that makes it unique.

        Returns:
            What the venue recorded for the bet.
        """

    @abc.abstractmethod
    async def read_status(self: BaseVenue, identities: list[BetIdentity]) -> pd.DataFrame:
        """Return what the venue holds for these identities.

        Args:
            identities: Identities to read the status of.

        Returns:
            One row per identity, with the status the venue holds for it.
        """

    @abc.abstractmethod
    async def cancel(self: BaseVenue, identity: BetIdentity) -> PlacementReceipt:
        """Cancel a bet.

        Args:
            identity: Identity of the bet to cancel.

        Returns:
            What the venue recorded for the cancellation.

        Raises:
            CancellationUnsupportedError: If the venue cannot cancel a bet.
        """
