"""Offline fakes for the single-event runner tests."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from sportsbet.execution import (
    ExecutionError,
    FixedSession,
    PageSnapshot,
    PlacementIntent,
    PlacementReceipt,
    PlacementStatus,
)
from tests.conftest import SnapshotsDataLoader

EVENT = 'Arsenal vs Chelsea'


class StubBettor:
    """A bettor a test hands a fixed decision, standing in for a fitted model."""

    def __init__(self, markets: tuple[str, ...] = ('home_win',), *, value: bool = True) -> None:
        """Keep the markets and whether every market is a value bet."""
        self.betting_markets_ = np.array(list(markets))
        self._value = value

    def bet(self, X: pd.DataFrame, O: pd.DataFrame) -> np.ndarray:
        """Return the fixed decision for every row and market."""
        return np.array([[self._value] * len(self.betting_markets_) for _ in range(len(X))])


class StubSession:
    """A browser session a test holds in its hand, driving no real browser."""

    key = 'stub'

    def __init__(
        self,
        page_event: str | None = EVENT,
        *,
        fail_auth: bool = False,
        locators: dict[str, str] | None = None,
    ) -> None:
        """Keep what the test arranged."""
        self.page_event = page_event
        self.fail_auth = fail_auth
        self.fixed_ = FixedSession(match='', url='', locators=dict(locators)) if locators else None
        self.navigated: list[str] = []
        self.typed: list[tuple[str, str]] = []
        self.clicked: list[str] = []
        self.authenticated = False

    async def navigate(self, url: str) -> PageSnapshot:
        """Return a page that carries the arranged event, or nothing."""
        self.navigated.append(url)
        return PageSnapshot(yaml=self.page_event or 'no event here', url=url)

    async def authenticate(self) -> None:
        """Accept the login, unless the test arranged a failure."""
        if self.fail_auth:
            msg = '`stub` refused the login.'
            raise ExecutionError(msg)
        self.authenticated = True

    def fix(self, match: str, locators: dict[str, str]) -> FixedSession:
        """Pin the locators for the match."""
        self.fixed_ = FixedSession(match=match, url='stub://', locators=dict(locators))
        return self.fixed_

    async def resolve(self, name: str) -> PageSnapshot:
        """Return a snapshot carrying a ref for the pinned control."""
        return PageSnapshot(yaml=f'textbox "{name}" [ref=e1]', url='stub://')

    async def type(self, ref: str, text: str) -> PageSnapshot:
        """Record a fill and return the page it produced."""
        self.typed.append((ref, text))
        return PageSnapshot(yaml='typed', url='stub://')

    async def click(self, ref: str) -> PageSnapshot:
        """Record a click and return the page it produced."""
        self.clicked.append(ref)
        return PageSnapshot(yaml='clicked', url='stub://')

    async def stop(self) -> None:
        """Close nothing, since a stub opens nothing."""


class RecordingPlacer:
    """A placer that records the intent it was handed and reports a matched bet."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[PlacementIntent] = []

    async def __call__(self, intent: PlacementIntent, session: StubSession) -> PlacementReceipt:
        """Record the intent and return a matched receipt for it."""
        self.calls.append(intent)
        return PlacementReceipt(
            identity=intent.identity,
            status=PlacementStatus.MATCHED_FULL,
            stake=intent.stake,
            price=intent.min_price,
            value_bet=intent.value_bet,
            placed_at=pd.Timestamp.now(tz='UTC').to_pydatetime(),
        )


@pytest.fixture
def clock_and_wait() -> Callable[[pd.Timestamp], tuple[Callable[[], pd.Timestamp], Callable[[float], object]]]:
    """Return a factory for a fake clock and a wait that advances it."""

    def build(start: pd.Timestamp) -> tuple[Callable[[], pd.Timestamp], Callable[[float], object]]:
        state = {'now': start}

        def clock() -> pd.Timestamp:
            return state['now']

        async def wait(seconds: float) -> None:
            state['now'] = state['now'] + pd.Timedelta(seconds=seconds)

        return clock, wait

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
