"""It provides the placing of the bets a bettor found.

A bettor produces value bets and stops. This is what takes them to a venue, and it refuses by default: nothing is staked
until the caller passes back the exact figures that were quoted.
"""

from __future__ import annotations

from ._base import (
    BaseVenue,
    BetIdentity,
    CancellationUnsupportedError,
    ExecutionError,
    ExposureLimits,
    PlacementIntent,
    PlacementQuote,
    PlacementReceipt,
    PlacementReceiptSchema,
    PlacementStatus,
    VenueBlockedError,
    receipts_frame,
)
from ._betfair import BetfairVenue
from ._credentials import CredentialError, CredentialRef, resolve
from ._place import place, quote

__all__: list[str] = [
    'BaseVenue',
    'BetIdentity',
    'BetfairVenue',
    'CancellationUnsupportedError',
    'CredentialError',
    'CredentialRef',
    'ExecutionError',
    'ExposureLimits',
    'PlacementIntent',
    'PlacementQuote',
    'PlacementReceipt',
    'PlacementReceiptSchema',
    'PlacementStatus',
    'VenueBlockedError',
    'place',
    'quote',
    'receipts_frame',
    'resolve',
]
