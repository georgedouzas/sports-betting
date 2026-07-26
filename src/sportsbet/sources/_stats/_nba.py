"""Read the NBA statistics from ESPN."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import calendar
import json
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

from ...core import ParamGrid
from .._base import BaseStatsSource, RawItem, RawPayload
from .._common._basketball import DIVISION, SEASONS_KEY, _snapshots

SEASONS_URL = 'https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons?limit=100'
GAMES_URL = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={start}-{end}&limit=1000'
LEAGUE = 'NBA'
EXHIBITION = 'ALLSTAR'
PRESEASON = 1
MONTHS = [(-1, month) for month in (9, 10, 11, 12)] + [(0, month) for month in (1, 2, 3, 4, 5, 6, 7)]


def _wanted(event: dict[str, Any]) -> bool:
    """Return whether an event is a game of the competition rather than an exhibition."""
    competitions = event.get('competitions') or [{}]
    season_type = event.get('season', {}).get('type')
    competition_type = competitions[0].get('type', {}).get('abbreviation')
    return season_type != PRESEASON and competition_type != EXHIBITION


def _games(content: bytes, year: int) -> pd.DataFrame:
    """Return the games of a month, with the API's UTC tip-off and its played flag."""
    events = json.loads(content).get('events', [])
    records = []
    for event in events:
        if not _wanted(event):
            continue
        competition = (event.get('competitions') or [{}])[0]
        competitors = {side.get('homeAway'): side for side in competition.get('competitors', [])}
        home, away = competitors.get('home'), competitors.get('away')
        if home is None or away is None or not event.get('date'):
            continue
        played = bool(competition.get('status', {}).get('type', {}).get('completed'))
        records.append(
            {
                'date': event['date'],
                'league': LEAGUE,
                'division': DIVISION,
                'year': year,
                'home_team': home.get('team', {}).get('displayName'),
                'away_team': away.get('team', {}).get('displayName'),
                'home_points': int(home.get('score', -1)) if played else -1,
                'away_points': int(away.get('score', -1)) if played else -1,
            },
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame['date'] = pd.to_datetime(frame['date'], utc=True, format='ISO8601').dt.tz_localize(None)
    return frame


class NBAStats(BaseStatsSource):
    """The NBA schedule and final scores from ESPN, updated through a season, with home and away markets.

    Read more in the [user guide][user-guide].

    Examples:
        >>> from sportsbet.dataloaders import DataLoader
        >>> from sportsbet.sources import NBAStats, OddsApi
        >>> source = NBAStats()
        >>> source.name, source.kind, source.sport
        ('nba', 'stats', 'basketball')
        >>> # A season is named by the year it ends in, so 2026 is the 2025-26 season.
        >>> dataloader = DataLoader(
        ...     param_grid={'league': ['NBA'], 'year': [2026]},
        ...     stats=source,
        ...     odds=OddsApi(key_env='ODDS_API_KEY', markets=['h2h']),
        ... )
        >>> dataloader.sport_
        'basketball'
        >>> # A league is a source. The same sport is the same dataloader.
        >>> from sportsbet.sources import EuroLeagueStats
        >>> NBAStats().sport == EuroLeagueStats().sport
        True
    """

    sport: ClassVar[str | None] = 'basketball'
    name: ClassVar[str] = 'nba'

    def list_index_items(self: Self, selection: ParamGrid | None = None) -> list[RawItem]:
        """Return the seasons the competition publishes."""
        return [RawItem(source=self.name, key=SEASONS_KEY, url=SEASONS_URL)]

    def read_catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the seasons the competition publishes, each named by the year it ends in."""
        if not payloads:
            return []
        seasons = json.loads(payloads[0].content).get('items', [])
        years = {int(season['$ref'].rsplit('/', 1)[-1].split('?')[0]) for season in seasons if '$ref' in season}
        return sorted(
            ({'league': LEAGUE, 'division': DIVISION, 'year': year} for year in years),
            key=lambda params: params['year'],
        )

    def list_required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return one item per month of each selected season."""
        items = []
        for param in params:
            if param['league'] != LEAGUE:
                continue
            year = param['year']
            for offset, month in MONTHS:
                start = pd.Timestamp(year=year + offset, month=month, day=1)
                last = calendar.monthrange(start.year, month)[1]
                items.append(
                    RawItem(
                        source=self.name,
                        key=f'{LEAGUE}_{param["division"]}_{year}_{start.year}{month:02d}',
                        url=GAMES_URL.format(start=f'{start.year}{month:02d}01', end=f'{start.year}{month:02d}{last}'),
                    ),
                )
        return items

    def to_snapshots(self: Self, payloads: list[RawPayload]) -> pd.DataFrame:
        """Transform the months into the long statistics snapshots."""
        seasons: dict[int, list[pd.DataFrame]] = {}
        for payload in payloads:
            year = int(payload.item.key.split('_')[2])
            games = _games(payload.content, year)
            if not games.empty:
                seasons.setdefault(year, []).append(games)
        frames = []
        for year in sorted(seasons):
            games = pd.concat(seasons[year], ignore_index=True).sort_values('date').reset_index(drop=True)
            frames.append(_snapshots(games))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).replace({np.nan: None}).infer_objects()
