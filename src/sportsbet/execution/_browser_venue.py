"""Place a bet at a bookmaker's website, through hooks written for that bookmaker."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import abc
import re

import pandas as pd

from ..core import ExecutionError
from ._base import BaseVenue, BetIdentity, Confirmation, PlacementIntent, PlacementReceipt, PlacementStatus
from ._browser import BrowserSession

SPACES = re.compile(r'\s+')


class BaseBrowserVenue(BaseVenue):
    """A bookmaker's website as a venue, completed per bookmaker by writing seven hooks.

    The base drives the hooks in a fixed order and keeps every rule that protects the stake: it opens the event, adds
    the selection to the bet slip, reads the slip, checks the slip holds the intended selection, checks the price
    against the minimum, enters the stake, confirms and reads the confirmation. A step that fails stops the sequence
    before the stake is confirmed, and a receipt carries only what the site confirmed.

    A hook reaches the page through `session.page_`, the playwright page, with ordinary playwright locators. Use
    `sportsbet execution page read` and `page act` to find them. Driving a bookmaker's website breaches essentially
    every bookmaker's terms of service and risks the account being closed and the balance lost.

    Args:
        key:
            What to call this venue.
        session:
            The browser session, opened from the profile you logged in with.

    Examples:
        >>> from sportsbet.execution import BaseBrowserVenue, BrowserSession
        >>> class Book(BaseBrowserVenue):
        ...     async def check_login(self): return True
        ...     async def find_event(self, match): return None
        ...     async def read_slip(self): return None
        ...     async def enter_stake(self, stake): return None
        ...     async def confirm_bet(self): return None
        ...     async def read_confirmation(self): return None
        >>> try:
        ...     Book('book', BrowserSession(key='book', url='https://example.invalid'))
        ... except TypeError as error:
        ...     print('add_selection' in str(error))
        True
    """

    def __init__(self: BaseBrowserVenue, key: str, session: BrowserSession) -> None:
        self.key = key
        self.session = session

    @abc.abstractmethod
    async def check_login(self: BaseBrowserVenue) -> bool:
        """Return whether the open site shows a logged-in session."""

    @abc.abstractmethod
    async def find_event(self: BaseBrowserVenue, match: str) -> str | None:
        """Return the URL of the event's page.

        Args:
            match: The event, as `Home vs Away`.

        Returns:
            The URL, or `None` when the site does not list the event.
        """

    @abc.abstractmethod
    async def add_selection(self: BaseBrowserVenue, identity: BetIdentity) -> None:
        """Add the selection to the bet slip, on the event's page.

        Args:
            identity: The market and selection to add.
        """

    @abc.abstractmethod
    async def read_slip(self: BaseBrowserVenue) -> tuple[str, float] | None:
        """Return the selection label and the price the bet slip shows.

        Returns:
            The label and the price, or `None` when the slip is empty.
        """

    @abc.abstractmethod
    async def enter_stake(self: BaseBrowserVenue, stake: float) -> None:
        """Type the stake into the bet slip.

        Args:
            stake: The amount to stake.
        """

    @abc.abstractmethod
    async def confirm_bet(self: BaseBrowserVenue) -> None:
        """Confirm the bet on the slip."""

    @abc.abstractmethod
    async def read_confirmation(self: BaseBrowserVenue) -> Confirmation | None:
        """Return what the site showed after the bet was confirmed.

        Returns:
            The confirmation, or `None` when the site showed none within its timeout.
        """

    def match_selection(self: BaseBrowserVenue, identity: BetIdentity, label: str) -> bool:
        """Return whether the slip's label names the intended selection.

        It compares without case and with runs of whitespace collapsed. Override it for a site that translates names.

        Args:
            identity: The intended bet.
            label: The selection label the slip shows.

        Returns:
            Whether they name the same selection.
        """

        def normalize(text: str) -> str:
            return SPACES.sub(' ', text).strip().casefold()

        return normalize(identity.selection) == normalize(label)

    async def authenticate(self: BaseBrowserVenue) -> None:
        """Open the site from the saved profile and check the session is logged in.

        Raises:
            VenueBlockedError: If the site blocks automated access.
            ExecutionError: If the profile holds no login.
        """
        await self.session.navigate(self.session.url)
        if not await self.check_login():
            profile = self.session.user_data_dir or 'no profile directory'
            msg = (
                f'`{self.key}` is not logged in with the profile at {profile}. Log in once with a visible browser, '
                'by building the session with `headless=False` and the same `user_data_dir`, then run again.'
            )
            raise ExecutionError(msg)

    async def _open_slip(self: BaseBrowserVenue, identity: BetIdentity) -> tuple[str, float] | str:
        """Put the selection on the slip and return its label and price, or why it could not."""
        url = await self.find_event(identity.match)
        if url is None:
            return f'The event {identity.match} was not found on the site.'
        await self.session.navigate(url)
        await self.add_selection(identity)
        slip = await self.read_slip()
        if slip is None:
            return f'The bet slip is empty after adding {identity.selection}.'
        label, price = slip
        if not self.match_selection(identity, label):
            return f'The slip did not match: it holds {label}, not {identity.selection}.'
        return label, price

    async def read_price(self: BaseBrowserVenue, identity: BetIdentity) -> float | None:
        """Return the price the slip shows for a bet, without entering a stake.

        Args:
            identity: The event, market and selection to read the price of.

        Returns:
            The price, or `None` when the event or the selection is not offered.
        """
        opened = await self._open_slip(identity)
        return None if isinstance(opened, str) else opened[1]

    async def place(self: BaseBrowserVenue, intent: PlacementIntent) -> PlacementReceipt:
        """Place one bet in the fixed order, stopping before the stake at the first step that fails.

        Args:
            intent: The bet to place.

        Returns:
            What the site confirmed, or why nothing was staked.
        """
        opened = await self._open_slip(intent.identity)
        if isinstance(opened, str):
            status = PlacementStatus.REJECTED
            return PlacementReceipt(intent.identity, status, value_bet=intent.value_bet, detail=opened)
        _, price = opened
        if price < intent.min_price:
            detail = f'The slip shows {price}, below the minimum of {intent.min_price}, so nothing was staked.'
            return PlacementReceipt(
                intent.identity,
                PlacementStatus.REFUSED_PRICE,
                price=price,
                value_bet=intent.value_bet,
                detail=detail,
            )
        await self.enter_stake(intent.stake)
        await self.confirm_bet()
        confirmation = await self.read_confirmation()
        if confirmation is None:
            detail = 'The site showed no confirmation. Check the venue for the bet before running again.'
            return PlacementReceipt(
                intent.identity,
                PlacementStatus.REFUSED_UNCONFIRMED,
                value_bet=intent.value_bet,
                detail=detail,
            )
        placed = confirmation.status in {
            PlacementStatus.ACCEPTED,
            PlacementStatus.MATCHED_FULL,
            PlacementStatus.MATCHED_PARTIAL,
        }
        return PlacementReceipt(
            intent.identity,
            confirmation.status,
            stake=intent.stake if placed else 0.0,
            price=confirmation.price,
            venue_bet_id=confirmation.venue_bet_id,
            value_bet=intent.value_bet,
            placed_at=pd.Timestamp.now(tz='UTC').to_pydatetime() if placed else None,
            detail=confirmation.detail,
        )

    async def close(self: BaseBrowserVenue) -> None:
        """Close the browser."""
        await self.session.stop()
