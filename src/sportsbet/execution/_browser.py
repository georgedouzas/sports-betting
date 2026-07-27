"""Drive a bookmaker's website for a venue with no API."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ._base import ExecutionError, VenueBlockedError

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Page

BLOCKED_STATUS = frozenset({403, 429})
REF_PATTERN = re.compile(r'^e\d+$')
BROWSER_EXTRA = "Driving a site needs playwright's browser. Install it with `python -m playwright install chromium`."


@dataclass(frozen=True)
class PageSnapshot:
    """A page in a form an agent can reason about and act on.

    Args:
        yaml: The page as an accessibility tree.
        url: The page URL.
    """

    yaml: str
    url: str


@dataclass(frozen=True)
class FixedSession:
    """The locators found during exploration, pinned to a match.

    Args:
        match: The match the session is pinned to.
        url: The page URL it was pinned at.
        locators: The pinned locators, by name.
    """

    match: str
    url: str
    locators: dict[str, str] = field(default_factory=dict)


class BrowserSession:
    """A bookmaker's website, driven in a real browser on your own account.

    You provide everything the session knows about the site.

    Args:
        key:
            What to call this venue.
        url:
            The bookmaker's site.
        notes:
            What you know about the site.
        credential_env:
            The names of the variables holding the username and the password.
        user_data_dir:
            Where the browser keeps its profile.
        min_interval:
            The seconds to leave between actions.
        timeout:
            The milliseconds to wait for an element before giving up on it.
        headless:
            Whether to hide the browser window. `False` opens it.

    Examples:
        >>> from sportsbet.execution import BrowserSession
        >>> session = BrowserSession(
        ...     key='example',
        ...     url='https://example.invalid/betting',
        ...     notes='The slip opens on the right after clicking a price.',
        ... )
        >>> session.key
        'example'
        >>> session.notes
        'The slip opens on the right after clicking a price.'
        >>> hasattr(session, 'place')
        False
    """

    def __init__(
        self: BrowserSession,
        key: str,
        url: str,
        *,
        notes: str | None = None,
        credential_env: tuple[str, str] | None = None,
        user_data_dir: str | Path | None = None,
        min_interval: float = 1.0,
        timeout: float = 30000.0,
        headless: bool = True,
    ) -> None:
        """Keep what the session was configured with."""
        self.key = key
        self.url = url
        self.notes = notes
        self.credential_env = credential_env
        self.user_data_dir = user_data_dir
        self.min_interval = min_interval
        self.timeout = timeout
        self.headless = headless
        self.context_: BrowserContext | None = None
        self.fixed_: FixedSession | None = None
        self._playwright: object | None = None

    async def authenticate(self: BrowserSession) -> None:
        """Open the browser at the site, starting from the saved profile."""
        await self.navigate(self.url)

    async def start(self: BrowserSession) -> None:
        """Open the browser and keep it open."""
        from playwright.async_api import async_playwright  # noqa: PLC0415  # defer the optional browser extra

        if self.context_ is not None:
            return
        driver = await async_playwright().start()
        self._playwright = driver
        profile = Path(self.user_data_dir) if self.user_data_dir else None
        if profile is None:
            browser = await driver.chromium.launch(headless=self.headless)
            self.context_ = await browser.new_context()
        else:
            self.context_ = await driver.chromium.launch_persistent_context(str(profile), headless=self.headless)

    async def stop(self: BrowserSession) -> None:
        """Close the browser."""
        if self.context_ is not None:
            await self.context_.close()
            self.context_ = None
        if self._playwright is not None:
            await self._playwright.stop()  # type: ignore[attr-defined]  # driver untyped to keep the extra optional
            self._playwright = None

    def _read_page(self: BrowserSession) -> Page:
        """Return the page being driven."""
        if self.context_ is None:
            msg = 'The browser is not open. Call `start` first.'
            raise ExecutionError(msg)
        if not self.context_.pages:
            msg = 'The browser is open at no page. Call `navigate` first.'
            raise ExecutionError(msg)
        return self.context_.pages[0]

    async def _paced(self: BrowserSession) -> None:
        """Leave the configured interval between actions."""
        await asyncio.sleep(self.min_interval)

    async def _capture(self: BrowserSession, selector: str | None = None, depth: int | None = None) -> PageSnapshot:
        """Return the page as it is now."""
        page = self._read_page()
        target = page.locator(selector) if selector else page.locator('body')
        return PageSnapshot(yaml=await target.aria_snapshot(mode='ai', depth=depth), url=page.url)

    async def navigate(self: BrowserSession, url: str) -> PageSnapshot:
        """Go to a page and return it.

        Args:
            url:
                Where to go.

        Returns:
            snapshot:
                The page, with a ref for every element that can be acted on.

        Raises:
            VenueBlockedError: If the site blocks automated access.
        """
        if self.context_ is None:
            await self.start()
        assert self.context_ is not None
        page = self.context_.pages[0] if self.context_.pages else await self.context_.new_page()
        await self._paced()
        response = await page.goto(url)
        if response is not None and response.status in BLOCKED_STATUS:
            msg = f'`{self.key}` refused automated access with status {response.status}.'
            raise VenueBlockedError(msg)
        return await self._capture()

    async def read_snapshot(
        self: BrowserSession,
        selector: str | None = None,
        depth: int | None = None,
    ) -> PageSnapshot:
        """Return the page, or a part of it.

        Args:
            selector:
                The part to read.
            depth:
                How far down to read.

        Returns:
            snapshot:
                The page, with a ref for every element that can be acted on.
        """
        return await self._capture(selector, depth)

    async def click(self: BrowserSession, ref: str) -> PageSnapshot:
        """Click an element and return the page it produced.

        Args:
            ref:
                The ref of the element, from a snapshot.

        Returns:
            snapshot:
                The page after the click.
        """
        await self._paced()
        await self._read_page().locator(f'aria-ref={ref}').click(timeout=self.timeout)
        return await self._capture()

    async def type(self: BrowserSession, ref: str, text: str) -> PageSnapshot:
        """Fill an element and return the page it produced.

        Args:
            ref:
                The ref of the element, from a snapshot.
            text:
                What to put in it.

        Returns:
            snapshot:
                The page after the text went in.
        """
        await self._paced()
        await self._read_page().locator(f'aria-ref={ref}').fill(text, timeout=self.timeout)
        return await self._capture()

    async def select(self: BrowserSession, ref: str, value: str) -> PageSnapshot:
        """Choose an option and return the page it produced.

        Args:
            ref:
                The ref of the element, from a snapshot.
            value:
                The option to choose.

        Returns:
            snapshot:
                The page after the option was chosen.
        """
        await self._paced()
        await self._read_page().locator(f'aria-ref={ref}').select_option(value, timeout=self.timeout)
        return await self._capture()

    def fix(self: BrowserSession, match: str, locators: dict[str, str]) -> FixedSession:
        """Pin the locators exploring found for a match.

        Args:
            match:
                The match this session is pinned to.
            locators:
                What was found, by name, as in `{'stake': 'textbox[name="Stake"]'}`.

        Returns:
            fixed:
                The pinned session.

        Raises:
            ExecutionError: If a locator is a ref or looks like a price.
        """
        for name, locator in locators.items():
            if REF_PATTERN.match(locator) or locator.startswith('aria-ref='):
                msg = (
                    f'`{name}` is pinned to the ref `{locator}`, which belongs to one state of the page. '
                    f'Pin a role and an accessible name instead, as in `textbox[name="Stake"]`.'
                )
                raise ExecutionError(msg)
            if 'price' in name or 'odds' in name:
                msg = f'`{name}` looks like a price. A price is read when the bet is placed rather than pinned.'
                raise ExecutionError(msg)
        self.fixed_ = FixedSession(match=match, url=self._read_page().url, locators=dict(locators))
        return self.fixed_

    async def resolve(self: BrowserSession, name: str) -> PageSnapshot:
        """Read what a pinned locator points at now.

        Args:
            name:
                The name it was pinned under.

        Returns:
            snapshot:
                What is there now.

        Raises:
            ExecutionError: If nothing is pinned or the name is not pinned.
        """
        if self.fixed_ is None:
            msg = 'Nothing is pinned. Call `fix` with what exploring found.'
            raise ExecutionError(msg)
        if name not in self.fixed_.locators:
            msg = f'`{name}` is not pinned. Pinned: {", ".join(sorted(self.fixed_.locators))}.'
            raise ExecutionError(msg)
        return await self._capture(self.fixed_.locators[name])
