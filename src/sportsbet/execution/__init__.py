"""It provides the placing of the bets a bettor found.

A bettor produces value bets and stops. This is what takes them to a venue, and it refuses by default: nothing is staked
until the caller passes back the exact figures that were quoted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._browser import BrowserSession, FixedSession, PageSnapshot

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
from ._credentials import CredentialError, CredentialRef, resolve
from ._place import place, quote, value_bet_intents
from ._schedule import betting_moment, execute, feasible

__all__: list[str] = [
    'BaseVenue',
    'BetIdentity',
    'BrowserSession',
    'CancellationUnsupportedError',
    'CredentialError',
    'CredentialRef',
    'ExecutionError',
    'ExposureLimits',
    'FixedSession',
    'PageSnapshot',
    'PlacementIntent',
    'PlacementQuote',
    'PlacementReceipt',
    'PlacementReceiptSchema',
    'PlacementStatus',
    'VenueBlockedError',
    'betting_moment',
    'execute',
    'feasible',
    'place',
    'quote',
    'receipts_frame',
    'resolve',
    'value_bet_intents',
]


_BROWSER = {'BrowserSession', 'FixedSession', 'PageSnapshot'}
_MISSING = "Driving a site needs the execution extra. Install it with `pip install 'sports-betting[execution]'`."


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Return a name that needs the browser, importing it only when it is asked for.

    The browser lives behind the optional extra, so importing it here would make the whole module need it. Everything
    that reaches a venue through its own API needs nothing beyond what the library already installs.
    """
    if name in _BROWSER:
        try:
            from . import _browser  # noqa: PLC0415
        except ImportError as missing:
            raise ImportError(_MISSING) from missing
        return getattr(_browser, name)
    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)
