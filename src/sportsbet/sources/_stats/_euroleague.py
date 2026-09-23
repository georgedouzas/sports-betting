"""Read the EuroLeague statistics from its official API."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import json
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

from ...core import ParamGrid
from .._base import BaseStatsSource, RawItem, RawPayload
from .._common._basketball import DIVISION, SEASONS_KEY, _build_snapshots

URL = 'https://api-live.euroleague.net/v2/competitions/E'
SEASONS_URL = f'{URL}/seasons'
GAMES_URL = f'{URL}/seasons/E{{season}}/games'
LEAGUE = 'Euroleague'


def _read_games(content: bytes, year: int) -> pd.DataFrame:
    """Return the games of a season, taking the tip-off from the API's UTC field."""
    games: list[dict[str, Any]] = json.loads(content).get('data', [])
    records = []
    for game in games:
        home, away = game.get('local', {}), game.get('road', {})
        home_name = home.get('club', {}).get('name')
        away_name = away.get('club', {}).get('name')
        if not home_name or not away_name or not game.get('utcDate'):
            continue
        played = bool(game.get('played'))
        records.append(
            {
                'date': game['utcDate'],
                'league': LEAGUE,
                'division': DIVISION,
                'year': year,
                'home_team': home_name,
                'away_team': away_name,
                'home_points': int(home.get('score', -1)) if played else -1,
                'away_points': int(away.get('score', -1)) if played else -1,
            },
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame['date'] = pd.to_datetime(frame['date'], utc=True, format='ISO8601').dt.tz_localize(None)
    return frame.sort_values('date').reset_index(drop=True)


class EuroLeagueStats(BaseStatsSource):
    """The EuroLeague schedule and final scores from its official API, with home and away markets.

    Read more in the [user guide][user-guide].

    Examples:
        >>> from sportsbet.dataloaders import DataLoader
        >>> from sportsbet.sources import EuroLeagueStats, OddsApi
        >>> source = EuroLeagueStats()
        >>> source.name, source.kind, source.sport
        ('euroleague', 'stats', 'basketball')
        >>> # A whole season arrives in one request, and asking what it publishes costs one more.
        >>> len(source.list_index_items())
        1
        >>> # Pair the statistics with an odds source.
        >>> dataloader = DataLoader(
        ...     param_grid={'league': ['Euroleague'], 'division': [1], 'year': [2025]},
        ...     stats=source,
        ...     odds=OddsApi(key_env='ODDS_API_KEY', markets=['h2h']),
        ... )
        >>> dataloader.sport_
        'basketball'
    """

    sport: ClassVar[str | None] = 'basketball'
    name: ClassVar[str] = 'euroleague'

    def list_index_items(self: Self, selection: ParamGrid | None = None) -> list[RawItem]:
        """Return the seasons the competition publishes.

        Args:
            selection:
                What is being looked for. `None` asks for everything.

        Returns:
            items:
                The items whose payloads describe the catalogue.
        """
        return [RawItem(source=self.name, key=SEASONS_KEY, url=SEASONS_URL)]

    def read_catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the seasons the competition publishes, each named by the year it ends in.

        Args:
            payloads:
                The payloads of the index items.

        Returns:
            params:
                The available `league`, `division` and `year` combinations.
        """
        if not payloads:
            return []
        seasons = json.loads(payloads[0].content).get('data', [])
        return sorted(
            ({'league': LEAGUE, 'division': DIVISION, 'year': int(season['year']) + 1} for season in seasons),
            key=lambda params: params['year'],
        )

    def list_required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return one item per selected season.

        Args:
            params:
                The selected parameter combinations.

            schedule:
                The matches of the selected parameters, with their kick-off instants.

        Returns:
            items:
                The items to read. Deterministic for the same parameters.
        """
        return [
            RawItem(
                source=self.name,
                key=f'{LEAGUE}_{param["division"]}_{param["year"]}',
                url=GAMES_URL.format(season=param['year'] - 1),
            )
            for param in params
            if param['league'] == LEAGUE
        ]

    def to_snapshots(self: Self, payloads: list[RawPayload]) -> pd.DataFrame:
        """Transform the seasons into the long statistics snapshots.

        Args:
            payloads:
                The payloads of the required items.

        Returns:
            snapshots:
                The long snapshots.
        """
        frames = []
        for payload in payloads:
            year = int(payload.item.key.rsplit('_', 1)[-1])
            games = _read_games(payload.content, year)
            if not games.empty:
                frames.append(_build_snapshots(games))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).replace({np.nan: None}).infer_objects()
