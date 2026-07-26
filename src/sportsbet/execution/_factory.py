"""Build a venue from a reference to where it lives."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core import BuildError
from ..core._reference import _load_object
from ._base import BaseVenue

if TYPE_CHECKING:
    from ._browser import BrowserSession

EXECUTION_EXTRA = "Placing bets needs the execution extra. Install it with `pip install 'sports-betting[execution]'`."


def build_venue(venue: str) -> BaseVenue | BrowserSession:
    """Build a venue from a reference to where it lives.

    Args:
        venue:
            Where the venue lives, as in `venue.py:VENUE`. The library ships no bookmaker: a venue with an API
            is a `BaseVenue` you write, and a bookmaker's website is a `BrowserSession` you configure.

    Returns:
        built:
            The venue, or the browser session.
    """
    if ':' not in venue:
        msg = f'`{venue}` should name a venue in a Python file, as in `venue.py:VENUE`. The library ships none.'
        raise BuildError(msg)
    built = _load_object(venue)
    try:
        from ._browser import BrowserSession  # noqa: PLC0415
    except ImportError as missing:
        raise BuildError(EXECUTION_EXTRA) from missing
    if not isinstance(built, BaseVenue | BrowserSession):
        msg = f'`{venue}` is not a venue and is not a browser session.'
        raise BuildError(msg)
    return built
