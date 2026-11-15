"""Offline venues, data and a served bookmaker for the single-event run tests.

Nothing here reaches a bookmaker. The API venue is a reference held in memory, and the browser venue drives pages served
over loopback, which is the only kind of site a test may open.
"""

import asyncio
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from sportsbet.core import VenueBlockedError
from sportsbet.dataloaders import BaseDataLoader
from sportsbet.evaluation import BaseBettor
from sportsbet.execution import (
    BaseBrowserVenue,
    BaseVenue,
    BetIdentity,
    BrowserSession,
    Confirmation,
    CredentialRef,
    EventRunRecord,
    PlacementIntent,
    PlacementReceipt,
    PlacementStatus,
    execute_event,
    resolve,
)
from tests.conftest import SnapshotsDataLoader

EVENT = 'Arsenal vs Chelsea'
STAKE = 10.0
SOURCE_PRICE = 2.0
BOOK_KEY = (EVENT, 'home_win', 'Arsenal')
SITE = Path(__file__).parent / 'site'
CONFIRMATION_TIMEOUT = 400.0


class StubBettor:
    """A bettor a test hands a fixed decision, standing in for a fitted model."""

    def __init__(
        self,
        markets: tuple[str, ...] = ('home_win',),
        *,
        value: bool = True,
        backed: set[str] | None = None,
    ) -> None:
        """Keep the markets and which of them are value bets, all of them by default."""
        self.betting_markets_ = np.array(list(markets))
        self._backed = set(markets) if backed is None else set(backed)
        self._value = value

    def bet(self, X: pd.DataFrame, O: pd.DataFrame) -> np.ndarray:
        """Return the fixed decision for every row, in the order of the markets."""
        row = [self._value and market in self._backed for market in self.betting_markets_]
        return np.array([row for _ in range(len(X))])


class ScriptedDataLoader:
    """A dataloader whose fixtures change from one read to the next, as a live source's do.

    Each read takes the next entry of the script: a mapping of market to price that rewrites the odds, `'unlisted'` for
    an event that left the fixtures, or an exception to raise. The last entry repeats.
    """

    target_event_status_ = 'preplay'

    def __init__(self, loader: SnapshotsDataLoader, script: list) -> None:
        """Keep the real fixtures and the script of reads."""
        self.X, _, self.O = loader.extract_fixtures_data()
        self.script = list(script)
        self.reads = 0

    def extract_fixtures_data(self) -> tuple[pd.DataFrame, None, pd.DataFrame]:
        """Return the fixtures as the next entry of the script has them."""
        entry = self.script[min(self.reads, len(self.script) - 1)]
        self.reads += 1
        if isinstance(entry, Exception):
            raise entry
        if entry == 'unlisted':
            return self.X.iloc[0:0], None, self.O.iloc[0:0]
        odds = self.O.copy()
        for market, price in entry.items():
            for column in [col for col in odds.columns if f'__{market}__' in col]:
                odds[column] = price
        return self.X, None, odds


class ReferenceVenue(BaseVenue):
    """A venue with an API, held in memory: a book of prices and the bets placed against it.

    It is the reference the contract is proven against, and the shape a user's own API venue takes.
    """

    key = 'reference'

    def __init__(
        self,
        book: dict[tuple[str, str, str], float] | None = None,
        *,
        drift: float | None = None,
        limit: float | None = None,
        confirms: bool = True,
        token_env: str | None = None,
        blocked: bool = False,
    ) -> None:
        """Keep the book and the switches the test arranged."""
        self.book = dict(book or {})
        self.drift = drift
        self.limit = limit
        self.confirms = confirms
        self.token_env = token_env
        self.blocked = blocked
        self.placements: list[PlacementIntent] = []
        self.logins = 0
        self.closes = 0
        self._token: str | None = None

    async def authenticate(self) -> None:
        """Accept the login, reading a token from its variable when one is named."""
        if self.blocked:
            msg = '`reference` blocked automated access.'
            raise VenueBlockedError(msg)
        if self.token_env is not None:
            self._token = resolve(CredentialRef(self.token_env))
        self.logins += 1

    async def read_price(self, identity: BetIdentity) -> float | None:
        """Return the book's price for the selection, or nothing."""
        return self.book.get((identity.match, identity.market, identity.selection))

    async def place(self, intent: PlacementIntent) -> PlacementReceipt:
        """Check the price, then place the bet and answer from the book."""
        identity = intent.identity
        key = (identity.match, identity.market, identity.selection)
        if self.drift is not None and key in self.book:
            self.book[key] = self.drift
        price = self.book.get(key)
        if price is None:
            return PlacementReceipt(identity, PlacementStatus.REJECTED, detail='The selection is not offered.')
        if price < intent.min_price:
            return PlacementReceipt(identity, PlacementStatus.REFUSED_PRICE, price=price, detail='The price drifted.')
        if self.limit is not None and intent.stake > self.limit:
            detail = f'The stake exceeds the account limit of {self.limit}.'
            return PlacementReceipt(identity, PlacementStatus.REFUSED_LIMIT, detail=detail)
        self.placements.append(intent)
        if not self.confirms:
            detail = 'No confirmation came back. Check the venue for the bet.'
            return PlacementReceipt(identity, PlacementStatus.REFUSED_UNCONFIRMED, detail=detail)
        return PlacementReceipt(
            identity,
            PlacementStatus.MATCHED_FULL,
            stake=intent.stake,
            price=price,
            venue_bet_id=f'REF-{len(self.placements)}',
            value_bet=intent.value_bet,
            placed_at=pd.Timestamp.now(tz='UTC').to_pydatetime(),
        )

    async def close(self) -> None:
        """Count the release, since a run must release what it opened."""
        self.closes += 1


@dataclass
class ServedSite:
    """The fake bookmaker, and the state its pages are rendered from."""

    url: str
    state: dict = field(default_factory=dict)


class _SiteHandler(BaseHTTPRequestHandler):
    """Serve the fake bookmaker's pages, rendered from the state the test arranged."""

    def __init__(self, state: dict, *args: Any, **rest: Any) -> None:  # noqa: ANN401
        """Keep the state the pages are rendered from."""
        self.state = state
        super().__init__(*args, **rest)

    def do_GET(self) -> None:
        """Answer with the page for this path."""
        pages = {'/': 'index.html', '/events': 'events.html', '/event/arsenal-chelsea': 'event.html'}
        name = pages.get(self.path)
        status = self.state['status'] if name else 404
        body = (SITE / name).read_text() if name else 'not found'
        account = '<a id="account" href="/account">My account</a>' if self.state['logged_in'] else '<a>Log in</a>'
        values = {**self.state, 'account': account}
        for placeholder, value in values.items():
            body = body.replace(f'{{{{{placeholder}}}}}', str(value))
        self.send_response(status)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args: object) -> None:
        """Say nothing, since a test is not a web server."""


@pytest.fixture
def served_site() -> Iterator[ServedSite]:
    """Serve the fake bookmaker over loopback on a free port."""
    state = {
        'status': 200,
        'logged_in': True,
        'home_price': 2.5,
        'draw_price': 3.4,
        'away_price': 3.6,
        'slip_label': '',
        'confirmation': 'appear',
    }
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(_SiteHandler, state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield ServedSite(url=f'http://127.0.0.1:{server.server_port}', state=state)
    server.shutdown()


class LoopbackBook(BaseBrowserVenue):
    """The browser venue a user would write for the fake bookmaker, recording each step it took."""

    def __init__(self, key: str, session: BrowserSession) -> None:
        """Keep the key and the session, and start with no steps taken."""
        super().__init__(key, session)
        self.taken: list[str] = []

    async def check_login(self) -> bool:
        """Return whether the page shows the account link."""
        return await self.session.page_.locator('#account').count() > 0

    async def find_event(self, match: str) -> str | None:
        """Return the event's page from the list of football events, or nothing."""
        await self.session.navigate(f'{self.session.url}/events')
        link = self.session.page_.get_by_role('link', name=match, exact=True)
        if await link.count() == 0:
            return None
        return f'{self.session.url}{await link.get_attribute("href")}'

    async def add_selection(self, identity: BetIdentity) -> None:
        """Click the selection's button, when the page has one."""
        self.taken.append('add_selection')
        button = self.session.page_.locator(f'button.selection[data-label="{identity.selection}"]')
        if await button.count() > 0:
            await button.click(timeout=self.session.timeout)

    async def read_slip(self) -> tuple[str, float] | None:
        """Return the slip's label and price, or nothing when it is hidden."""
        page = self.session.page_
        if not await page.locator('#slip').is_visible():
            return None
        return await page.locator('#slip-label').inner_text(), float(await page.locator('#slip-price').inner_text())

    async def enter_stake(self, stake: float) -> None:
        """Type the stake."""
        self.taken.append('enter_stake')
        await self.session.page_.locator('#stake').fill(str(stake), timeout=self.session.timeout)

    async def confirm_bet(self) -> None:
        """Click the confirm button."""
        self.taken.append('confirm_bet')
        await self.session.page_.locator('#confirm').click(timeout=self.session.timeout)

    async def read_confirmation(self) -> Confirmation | None:
        """Return what the confirmation says, or nothing when none appears in time."""
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError  # noqa: PLC0415

        shown = self.session.page_.locator('#confirmation')
        try:
            await shown.wait_for(state='visible', timeout=CONFIRMATION_TIMEOUT)
        except PlaywrightTimeoutError:
            return None
        price = await shown.get_attribute('data-price')
        return Confirmation(
            status=PlacementStatus(await shown.get_attribute('data-status')),
            venue_bet_id=await shown.get_attribute('data-bet-id'),
            price=float(price) if price else None,
            detail=await shown.inner_text(),
        )


@pytest.fixture
def book(served_site: ServedSite) -> LoopbackBook:
    """Return the browser venue for the served bookmaker, with no pacing."""
    session = BrowserSession(key='loopback', url=served_site.url, min_interval=0.0, timeout=1200.0)
    return LoopbackBook('loopback', session)


@pytest.fixture
def run() -> Iterator[Callable]:
    """Run coroutines in one event loop for the whole test, since playwright binds a browser to its loop."""
    loop = asyncio.new_event_loop()
    yield loop.run_until_complete
    loop.close()


class FakeWait:
    """A wait that advances a fake clock instead of sleeping, and records each wait."""

    def __init__(self, state: dict) -> None:
        """Keep the clock's state and start with no waits."""
        self.state = state
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        """Advance the clock by the seconds waited."""
        self.waits.append(seconds)
        self.state['now'] = self.state['now'] + pd.Timedelta(seconds=seconds)


@pytest.fixture
def clock_and_wait() -> Callable[[pd.Timestamp], tuple[Callable[[], pd.Timestamp], FakeWait]]:
    """Return a factory for a fake clock and a wait that advances it."""

    def build(start: pd.Timestamp) -> tuple[Callable[[], pd.Timestamp], FakeWait]:
        state = {'now': start}

        def clock() -> pd.Timestamp:
            return state['now']

        return clock, FakeWait(state)

    return build


@pytest.fixture
def event_dataloader(long_snapshots: tuple[pd.DataFrame, pd.DataFrame]) -> Callable[[], SnapshotsDataLoader]:
    """Return a factory for a dataloader carrying the unplayed `Arsenal vs Chelsea` fixture."""

    def build() -> SnapshotsDataLoader:
        stats, odds = long_snapshots
        loader = SnapshotsDataLoader(stats, odds)
        loader.extract_train_data(odds_type='market_average')
        return loader

    return build


def identity_of(market: str = 'home_win', selection: str = 'Arsenal', venue: str = 'reference') -> BetIdentity:
    """Return the identity of a bet on the event."""
    return BetIdentity(venue, EVENT, market, selection)


@pytest.fixture
def scripted(event_dataloader: Callable[[], SnapshotsDataLoader]) -> Callable[..., ScriptedDataLoader]:
    """Return a factory for a dataloader whose reads follow a script, home win priced at 2.0 by default."""

    def build(*script: object) -> ScriptedDataLoader:
        return ScriptedDataLoader(event_dataloader(), list(script) or [{'home_win': SOURCE_PRICE}])

    return build


@pytest.fixture
def start(clock_and_wait: Callable) -> Callable[..., tuple]:
    """Return a factory for a clock and a wait starting some time before the betting moment."""

    def build(loader: ScriptedDataLoader, before_moment: str = '1min') -> tuple:
        kickoff = loader.extract_fixtures_data()[0].index[0]
        loader.reads = 0
        return clock_and_wait(kickoff - pd.Timedelta('5min') - pd.Timedelta(before_moment))

    return build


def run_event(
    loader: ScriptedDataLoader,
    venue: BaseVenue,
    clock: Callable,
    wait: Callable,
    ledger_path: Path,
    *,
    bettor: StubBettor | None = None,
    live: bool = False,
    **settings: object,
) -> EventRunRecord:
    """Run the event to its end and return its record."""
    return asyncio.run(
        execute_event(
            EVENT,
            cast(BaseBettor, bettor or StubBettor()),
            cast(BaseDataLoader, loader),
            venue,
            stake=STAKE,
            ledger_path=ledger_path,
            live=live,
            clock=clock,
            wait=wait,
            **settings,
        ),
    )
