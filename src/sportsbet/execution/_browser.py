"""Drive a bookmaker's website for a venue with no API."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from ..core import ExecutionError, VenueBlockedError

BLOCKED_STATUS = frozenset({403, 429})


class _Response(Protocol):
    """What the session reads from a navigation response."""

    @property
    def status(self: _Response) -> int:
        """Status the site answered with."""


class _Locator(Protocol):
    """What the session does with an element it has located."""

    async def aria_snapshot(self: _Locator, *, mode: Literal['ai', 'default'] | None, depth: int | None) -> str:
        """Return the element as an accessibility tree.

        Args:
            mode: Shape of the tree to return.
            depth: How deep to walk, or None for the whole subtree.

        Returns:
            The tree.
        """
        ...

    async def click(self: _Locator, *, timeout: float) -> None:
        """Click the element.

        Args:
            timeout: Milliseconds to wait for it.
        """
        ...

    async def fill(self: _Locator, value: str, *, timeout: float) -> None:
        """Type into the element.

        Args:
            value: Text to type.
            timeout: Milliseconds to wait for it.
        """
        ...

    async def select_option(self: _Locator, value: str, *, timeout: float) -> list[str]:
        """Choose an option in the element.

        Args:
            value: Option to choose.
            timeout: Milliseconds to wait for it.

        Returns:
            The options that ended up chosen.
        """
        ...


class _Page(Protocol):
    """What the session does with the page it drives."""

    @property
    def url(self: _Page) -> str:
        """URL the page is at."""

    def locator(self: _Page, selector: str) -> _Locator:
        """Locate an element.

        Args:
            selector: Selector to locate it by.

        Returns:
            The located element.
        """
        ...

    async def goto(self: _Page, url: str) -> _Response | None:
        """Go to a URL.

        Args:
            url: URL to go to.

        Returns:
            The response, or None where the site answered none.
        """
        ...


class _Context(Protocol):
    """What the session does with the browser context it holds open."""

    @property
    def pages(self: _Context) -> Sequence[_Page]:
        """Pages the context has open."""

    async def new_page(self: _Context) -> _Page:
        """Open a page.

        Returns:
            The opened page.
        """
        ...

    async def close(self: _Context) -> None:
        """Close the context."""
        ...


@dataclass(frozen=True)
class PageSnapshot:
    """A page in a form an agent can reason about and act on.

    Args:
        yaml: The page as an accessibility tree.
        url: The page URL.
    """

    yaml: str
    url: str


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
        user_data_dir: str | Path | None = None,
        min_interval: float = 1.0,
        timeout: float = 30000.0,
        headless: bool = True,
    ) -> None:
        self.key = key
        self.url = url
        self.notes = notes
        self.user_data_dir = user_data_dir
        self.min_interval = min_interval
        self.timeout = timeout
        self.headless = headless
        self.context_: _Context | None = None
        self._playwright: object | None = None

    @property
    def page_(self: BrowserSession) -> Any:  # noqa: ANN401  # the playwright page, untyped to keep the extra optional
        """The page the browser is driving, as a playwright page.

        A browser venue reaches the site through it, with ordinary playwright locators.

        Raises:
            ExecutionError: If the browser is not open or is open at no page.
        """
        return self._read_page()

    def _read_page(self: BrowserSession) -> _Page:
        """Return the page being driven."""
        if self.context_ is None:
            msg = 'The browser is not open. Call `start` first.'
            raise ExecutionError(msg)
        if not self.context_.pages:
            msg = 'The browser is open at no page. Call `navigate` first.'
            raise ExecutionError(msg)
        return self.context_.pages[0]

    async def _wait_interval(self: BrowserSession) -> None:
        """Leave the configured interval between actions."""
        await asyncio.sleep(self.min_interval)

    async def _capture(self: BrowserSession, selector: str | None = None, depth: int | None = None) -> PageSnapshot:
        """Return the page as it is now."""
        page = self._read_page()
        target = page.locator(selector) if selector else page.locator('body')
        return PageSnapshot(yaml=await target.aria_snapshot(mode='ai', depth=depth), url=page.url)

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
        await self._wait_interval()
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
        await self._wait_interval()
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
        await self._wait_interval()
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
        await self._wait_interval()
        await self._read_page().locator(f'aria-ref={ref}').select_option(value, timeout=self.timeout)
        return await self._capture()
