"""Configuration for the pytest test suite."""

import socket
from importlib.resources import files
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytest
from click.testing import CliRunner

from sportsbet.cli import main
from sportsbet.dataloaders import BaseDataLoader, DataLoader
from sportsbet.sources import (
    BaseOddsSchema,
    BaseStatsSchema,
    SampleSoccerOdds,
    SampleSoccerStats,
    market_outcomes,
    optional_col,
    required_col,
)


class SnapshotsDataLoader(BaseDataLoader):
    """A dataloader of snapshots a test built itself.

    The library ships no way of doing this on purpose: data comes from sources, so that a user always knows where theirs
    came from. A test is the one place a bespoke moment has to be arranged, and the way to arrange it is the way anyone
    else would, by implementing `_snapshots`.
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
        self.odds = BaseDataLoader.no_odds() if odds is None else odds

    def _snapshots(self: 'SnapshotsDataLoader') -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the snapshots the test provided."""
        return self.stats, self.odds


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

    def guard(*args: object, **kwargs: object) -> None:
        msg = 'The test suite must not reach the network. Use a recorded payload or mark the test `network`.'
        raise RuntimeError(msg)

    monkeypatch.setattr(socket.socket, 'connect', guard)
    monkeypatch.setattr(socket.socket, 'connect_ex', guard)


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

    monkeypatch.setattr('sportsbet.cli._utils.build_dataloader', build)
    monkeypatch.setattr('sportsbet.mcp._server.build_dataloader', build)


@pytest.fixture
def offline_fixtures_dataloader(long_snapshots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hand the surfaces a dataloader that has a match still to be played.

    The sample season is finished, so it has no fixtures, and a command that predicts value bets needs one to predict.
    """
    stats, odds = long_snapshots

    def build(**rest: object) -> BaseDataLoader:
        return SnapshotsDataLoader(stats, odds)

    monkeypatch.setattr('sportsbet.cli._utils.build_dataloader', build)
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
    stats_file = files('tests') / 'samples' / 'stats.csv'
    data = pd.read_csv(stats_file, parse_dates=['date'])
    data['event_time'] = pd.to_timedelta(data['event_time'], unit='m').astype('timedelta64[ns]')
    data['date'] = pd.to_datetime(data['date'], utc=True).astype('datetime64[ns, UTC]')
    return data


@pytest.fixture
def odds() -> pd.DataFrame:
    """Load odds data."""
    odds_file = files('tests') / 'samples' / 'odds.csv'
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
        season: int = required_col()
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
        season: int = required_col()
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
            markets = market_outcomes(pd.Series([home_goals]), pd.Series([away_goals]), _MARKETS).iloc[0].to_dict()
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
