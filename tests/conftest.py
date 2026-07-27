"""Configuration for the pytest test suite."""

import socket
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytest
from click.testing import CliRunner

from sportsbet.cli import main
from sportsbet.dataloaders import BaseDataLoader, DataLoader
from sportsbet.execution import (
    BaseVenue,
    BetIdentity,
    CancellationUnsupportedError,
    PlacementIntent,
    PlacementReceipt,
    PlacementStatus,
    VenueBlockedError,
)
from sportsbet.sources import (
    BaseOddsSchema,
    BaseStatsSchema,
    SampleSoccerOdds,
    SampleSoccerStats,
    derive_market_outcomes,
    optional_col,
    required_col,
)


class SnapshotsDataLoader(BaseDataLoader):
    """A dataloader of snapshots a test built itself.

    The library ships no way of doing this on purpose: data comes from sources, so that a user always knows where theirs
    came from. A test is the one place a bespoke moment has to be arranged, and the way to arrange it is the way anyone
    else would, by implementing `_load_snapshots`.
    """

    def __init__(
        self: 'SnapshotsDataLoader',
        stats: pd.DataFrame,
        odds: pd.DataFrame | None = None,
        param_grid: dict | None = None,
    ) -> None:
        """Keep the snapshots the test provided."""
        super().__init__(param_grid)
        self.stats = stats
        self.odds = BaseDataLoader._build_empty_odds() if odds is None else odds

    def _load_snapshots(self: 'SnapshotsDataLoader') -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the snapshots the test provided."""
        return self.stats, self.odds


class FakeVenue(BaseVenue):
    """A venue a test can hold in its hand, and the only kind a test may reach.

    No venue offers a placement sandbox, so a fake is not the second-best way of proving this code. It is the only way
    of proving it that does not spend money.

    It keeps what it was asked to do and answers whether it already holds a bet, which is the one behaviour the once-
    only promise rests on. The faults are injectable because a retry, a crash and a lost connection are exactly when
    double staking happens.
    """

    key = 'fake'
    can_cancel = True

    def __init__(
        self: 'FakeVenue',
        prices: dict[tuple[str, str, str], float] | None = None,
        balance: float = 1000.0,
        exposure: float = 0.0,
        fail_after_accept: int | None = None,
        blocked: bool = False,
        cancels: bool = True,
        matched: float | None = None,
        min_interval: float = 0.0,
    ) -> None:
        """Keep what the test arranged."""
        self.prices = prices or {}
        self.balance = balance
        self.exposure = exposure
        self.fail_after_accept = fail_after_accept
        self.blocked = blocked
        self.can_cancel = cancels
        self.matched = matched
        self.min_interval = min_interval
        self.orders: dict[str, dict] = {}
        self.attempts: list[str] = []
        self.placed_order: list[str] = []

    async def authenticate(self: 'FakeVenue') -> None:
        """Accept any caller, since a fake holds nothing worth guarding."""
        if self.blocked:
            msg = '`fake` refused the login.'
            raise VenueBlockedError(msg)

    async def list_markets(self: 'FakeVenue', matches: list[str]) -> pd.DataFrame:
        """Return the prices the test arranged."""
        records = [
            {'match': match, 'market': market, 'selection': selection, 'price': price}
            for (match, market, selection), price in self.prices.items()
            if not matches or match in matches
        ]
        return pd.DataFrame.from_records(records, columns=['match', 'market', 'selection', 'price'])

    async def read_balance(self: 'FakeVenue') -> tuple[float, float]:
        """Return the balance and the exposure the test arranged."""
        return self.balance, self.exposure

    async def place(self: 'FakeVenue', intent: PlacementIntent) -> PlacementReceipt:
        """Record a bet, unless one is already recorded for its identity."""
        ref = intent.identity.ref_
        self.attempts.append(ref)
        if self.blocked:
            msg = '`fake` blocked automated access.'
            raise VenueBlockedError(msg)
        if ref in self.orders:
            order = self.orders[ref]
            return PlacementReceipt(
                identity=intent.identity,
                status=PlacementStatus.ALREADY_PLACED,
                stake=order['stake'],
                price=order['price'],
                venue_bet_id=order['bet_id'],
                value_bet=intent.value_bet,
                detail='`fake` already holds a bet for this selection.',
            )
        price = self.prices.get((intent.identity.match, intent.identity.market, intent.identity.selection))
        self.orders[ref] = {
            'stake': intent.stake,
            'price': price or intent.min_price,
            'bet_id': f'bet-{len(self.orders) + 1}',
        }
        self.placed_order.append(ref)
        if self.fail_after_accept is not None and len(self.orders) == self.fail_after_accept:
            msg = 'The connection dropped after the venue accepted the bet.'
            raise TimeoutError(msg)
        staked = intent.stake if self.matched is None else self.matched
        status = PlacementStatus.MATCHED_PARTIAL if staked < intent.stake else PlacementStatus.MATCHED_FULL
        return PlacementReceipt(
            identity=intent.identity,
            status=status,
            stake=staked,
            price=price or intent.min_price,
            venue_bet_id=self.orders[ref]['bet_id'],
            value_bet=intent.value_bet,
            placed_at=pd.Timestamp.now(tz='UTC').to_pydatetime(),
        )

    async def read_status(self: 'FakeVenue', identities: list[BetIdentity]) -> pd.DataFrame:
        """Return what the fake holds for these identities."""
        records = [
            {'ref': identity.ref_, **self.orders[identity.ref_]}
            for identity in identities
            if identity.ref_ in self.orders
        ]
        return pd.DataFrame.from_records(records)

    async def cancel(self: 'FakeVenue', identity: BetIdentity) -> PlacementReceipt:
        """Cancel a bet, saying so when the fake was arranged not to cancel."""
        if not self.can_cancel:
            msg = '`fake` cannot cancel a bet.'
            raise CancellationUnsupportedError(msg)
        self.orders.pop(identity.ref_, None)
        return PlacementReceipt(identity=identity, status=PlacementStatus.REJECTED, detail='`fake` cancelled the bet.')


@pytest.fixture(autouse=True, scope='session')
def pandas_terminal_width() -> None:
    """Set options to display data."""
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that reaches the network, unless it is marked `network`.

    A test must not reach the network, so fetching is made impossible and the suite is watched to stay green. The tests
    that do need real data are marked `network`.
    """
    if request.node.get_closest_marker('network'):
        return

    loopback = {'127.0.0.1', '::1', 'localhost'}

    def blocked(real: Callable[..., object]) -> Callable[..., object]:
        def guard(self: socket.socket, address: object, *args: object, **kwargs: object) -> object:
            host = address[0] if isinstance(address, tuple) else address
            if host in loopback:
                return real(self, address, *args, **kwargs)
            msg = 'The test suite must not reach the network. Use a recorded payload or mark the test `network`.'
            raise RuntimeError(msg)

        return guard

    monkeypatch.setattr(socket.socket, 'connect', blocked(socket.socket.connect))
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked(socket.socket.connect_ex))


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def offline_dataloader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hand the surfaces the sample data, so the commands are exercised without a network.

    The sample sources are not offered by the commands themselves, since a user asking for data should be choosing a
    real feed rather than the one the library brought with it.
    """

    def build(stats: str, leagues: list[str] | None = None, **rest: object) -> DataLoader:
        dataloader = DataLoader(
            param_grid={'league': list(leagues)} if leagues else None,
            stats=SampleSoccerStats(),
            odds=SampleSoccerOdds(),
        )
        dataloader.extract_train_data(odds_type='market_average')
        return dataloader

    monkeypatch.setattr('sportsbet.cli._building.build_dataloader', build)
    monkeypatch.setattr('sportsbet.mcp._server.build_dataloader', build)


@pytest.fixture
def offline_fixtures_dataloader(long_snapshots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hand the surfaces a dataloader that has a match still to be played.

    The sample season is finished, so it has no fixtures, and a command that predicts value bets needs one to predict.
    """
    stats, odds = long_snapshots

    def build(**rest: object) -> BaseDataLoader:
        return SnapshotsDataLoader(stats, odds)

    monkeypatch.setattr('sportsbet.cli._building.build_dataloader', build)
    monkeypatch.setattr('sportsbet.mcp._server.build_dataloader', build)


@pytest.fixture
def saved_dataloader(cli_runner: CliRunner, offline_dataloader: None, tmp_path: Path) -> Path:
    """Extract the sample training data to a saved dataloader and return its path."""
    path = tmp_path / 'dataloader.pkl'
    result = cli_runner.invoke(
        main,
        [
            'dataloader',
            'train',
            'extract',
            '--stats',
            'football-data',
            '--odds',
            'football-data',
            '--league',
            'England',
            '--league',
            'Spain',
            '--odds-type',
            'market_average',
            '--target-event-status',
            'postplay',
            '-o',
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    return path


@pytest.fixture
def saved_fixtures_dataloader(cli_runner: CliRunner, offline_fixtures_dataloader: None, tmp_path: Path) -> Path:
    """Extract a dataloader that still has a match to play and return its path."""
    path = tmp_path / 'dataloader.pkl'
    result = cli_runner.invoke(
        main,
        [
            'dataloader',
            'train',
            'extract',
            '--stats',
            'football-data',
            '--odds',
            'football-data',
            '--odds-type',
            'market_average',
            '-o',
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    return path


@pytest.fixture
def stats() -> pd.DataFrame:
    """Load statistics data."""
    stats_file = files('tests') / 'sources' / 'samples' / 'stats.csv'
    data = pd.read_csv(stats_file, parse_dates=['date'])
    data['event_time'] = pd.to_timedelta(data['event_time'], unit='m').astype('timedelta64[ns]')
    data['date'] = pd.to_datetime(data['date'], utc=True).astype('datetime64[ns, UTC]')
    return data


@pytest.fixture
def odds() -> pd.DataFrame:
    """Load odds data."""
    odds_file = files('tests') / 'sources' / 'samples' / 'odds.csv'
    data = pd.read_csv(odds_file, parse_dates=['date'])
    data['event_time'] = pd.to_timedelta(data['event_time'], unit='m').astype('timedelta64[ns]')
    data['date'] = pd.to_datetime(data['date'], utc=True).astype('datetime64[ns, UTC]')
    return data


@pytest.fixture
def stats_schema() -> BaseStatsSchema:
    """Load statistics schema."""

    class StatsSchema(BaseStatsSchema):
        """Statistics schema."""

        date: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = required_col()
        league: str = required_col()
        division: int = required_col()
        year: int = required_col()
        home_team: str = required_col()
        away_team: str = required_col()
        home_goals: int = optional_col(['inplay'], False)
        away_goals: int = optional_col(['inplay'], False)
        home_latest_streak: int = optional_col(['preplay'], True)
        away_latest_streak: int = optional_col(['preplay'], True)

    return StatsSchema


@pytest.fixture
def odds_schema() -> BaseOddsSchema:
    """Load odds schema."""

    class OddsSchema(BaseOddsSchema):
        """Odds schema."""

        date: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = required_col()
        league: str = required_col()
        division: int = required_col()
        year: int = required_col()
        home_team: str = required_col()
        away_team: str = required_col()
        provider: str = optional_col(['preplay'], True)
        home_win: float = optional_col(['preplay', 'inplay'], False)
        away_win: float = optional_col(['preplay', 'inplay'], False)

    return OddsSchema


_MARKETS = ['home_win', 'draw', 'away_win', 'over_2.5', 'under_2.5']
_PROVIDERS = ['market_average', 'market_maximum']
_FEATURES = ['home_points_avg', 'away_points_avg']
_IDENTITY = ['date', 'league', 'division', 'year', 'home_team', 'away_team']
_BASE_ODDS = {'home_win': 1.80, 'draw': 3.40, 'away_win': 4.20, 'over_2.5': 1.90, 'under_2.5': 1.95}
_PROVIDER_FACTOR = {'market_average': 0.98, 'market_maximum': 1.06}
_MATCHES: list[dict] = [
    {
        'date': '2024-08-16',
        'league': 'England',
        'home': 'Man United',
        'away': 'Fulham',
        'form': (2.1, 1.2),
        'score': (1, 0),
    },
    {
        'date': '2024-08-24',
        'league': 'England',
        'home': 'Newcastle',
        'away': 'Tottenham',
        'form': (1.5, 1.8),
        'score': (1, 2),
    },
    {
        'date': '2024-09-01',
        'league': 'England',
        'home': 'Brighton',
        'away': 'Everton',
        'form': (1.3, 0.9),
        'score': (0, 0),
    },
    {
        'date': '2024-09-14',
        'league': 'England',
        'home': 'Chelsea',
        'away': 'West Ham',
        'form': (1.9, 1.4),
        'score': (3, 1),
    },
    {
        'date': '2024-10-05',
        'league': 'England',
        'home': 'Liverpool',
        'away': 'Crystal Palace',
        'form': (2.4, 1.1),
        'score': (2, 1),
    },
    {
        'date': '2024-11-02',
        'league': 'England',
        'home': 'Aston Villa',
        'away': 'Wolves',
        'form': (1.6, 1.0),
        'score': (1, 1),
    },
    {
        'date': '2025-01-18',
        'league': 'England',
        'home': 'Man City',
        'away': 'Brentford',
        'form': (2.6, 1.3),
        'score': (4, 0),
    },
    {
        'date': '2025-03-08',
        'league': 'England',
        'home': 'Nottingham',
        'away': 'Bournemouth',
        'form': (1.4, 1.5),
        'score': (2, 2),
    },
    {
        'date': '2025-05-25',
        'league': 'Spain',
        'home': 'Barcelona',
        'away': 'Real Madrid',
        'form': (2.3, 2.2),
        'score': (3, 2),
    },
    {
        'date': '2025-09-01',
        'league': 'England',
        'home': 'Arsenal',
        'away': 'Chelsea',
        'form': (2.0, 1.7),
        'score': None,
    },
]


def _kick_off(match: dict) -> str:
    """Return the kick-off of a match, putting the one that has not been played in the future.

    A fixture is a match that has not been played, so its kick-off has not happened. A date written into the file would
    stop being a fixture the moment it went past, and the test would begin failing on a Tuesday for no reason.
    """
    if match['score'] is not None:
        return str(match['date'])
    return (pd.Timestamp.now(tz='UTC') + pd.Timedelta('7D')).strftime('%Y-%m-%d')


def _timeline(match: dict) -> list[tuple[str, int, int, int]]:
    """Return the `(event_status, minutes, home_goals, away_goals)` snapshots of a match."""
    if match['score'] is None:
        return [('preplay', 0, 0, 0), ('inplay', 30, 0, 0)]
    home_goals, away_goals = match['score']
    return [
        ('preplay', 0, 0, 0),
        ('inplay', 30, 0, 0),
        ('inplay', 60, (home_goals + 1) // 2, (away_goals + 1) // 2),
        ('inplay', 90, home_goals, away_goals),
        ('postplay', 0, home_goals, away_goals),
    ]


@pytest.fixture
def long_snapshots() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long `(stats, odds)` snapshots priced at several moments, including in play.

    The odds are made up, and that is why they live here rather than in the library. No free feed records the price that
    was on offer at minute 30, so a sample that carried one would be a sample of something that does not exist. A test
    needs that moment to exist, so it arranges it, and says so.
    """
    stats_records: list[dict] = []
    odds_records: list[dict] = []
    for match in _MATCHES:
        identity = {
            'date': _kick_off(match),
            'league': match['league'],
            'division': 1,
            'year': 2025,
            'home_team': match['home'],
            'away_team': match['away'],
        }
        for status, minutes, home_goals, away_goals in _timeline(match):
            markets = (
                derive_market_outcomes(pd.Series([home_goals]), pd.Series([away_goals]), _MARKETS).iloc[0].to_dict()
            )
            features = (
                dict(zip(_FEATURES, match['form'], strict=True)) if status == 'preplay' else dict.fromkeys(_FEATURES)
            )
            stats_records.append(
                {
                    **identity,
                    'event_status': status,
                    'event_time': pd.Timedelta(minutes=minutes),
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    **markets,
                    **features,
                },
            )
            if status == 'postplay':
                continue
            for provider in _PROVIDERS:
                factor = _PROVIDER_FACTOR[provider] * (1 + 0.004 * minutes)
                odds_records.append(
                    {
                        **identity,
                        'event_status': status,
                        'event_time': pd.Timedelta(minutes=minutes),
                        'provider': provider,
                        **{market: round(base * factor, 2) for market, base in _BASE_ODDS.items()},
                    },
                )
    stats = pd.DataFrame(stats_records)[
        ['event_status', 'event_time', *_IDENTITY, 'home_goals', 'away_goals', *_MARKETS, *_FEATURES]
    ]
    odds = pd.DataFrame(odds_records)[['event_status', 'event_time', *_IDENTITY, 'provider', *_MARKETS]]
    return stats, odds
