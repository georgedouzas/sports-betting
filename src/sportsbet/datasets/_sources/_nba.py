"""Implements the statistics source of the NBA, backed by ESPN."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import calendar
import json
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

from .._utils import market_outcomes
from ._base import BaseStatsSource, RawItem, RawPayload

SEASONS_URL = 'https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons?limit=100'
GAMES_URL = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={start}-{end}&limit=1000'

LEAGUE = 'NBA'
DIVISION = 1
MARKETS = ['home_win', 'away_win']
SEASONS_KEY = 'seasons'
ROLLING_GAMES = 3
FEATURES = ['points_for', 'points_against', 'wins']
IDENTITY = ['date', 'league', 'division', 'year', 'home_team', 'away_team']

PRESEASON = 1
EXHIBITION = 'ALLSTAR'
MONTHS = [(-1, month) for month in (9, 10, 11, 12)] + [(0, month) for month in (1, 2, 3, 4, 5, 6, 7)]


def _wanted(event: dict[str, Any]) -> bool:
    """Return whether an event is a game of the competition rather than an exhibition.

    The two labels an event carries each lie on their own. The all-star games are filed under the regular season, and
    their teams are inventions, so keeping the regular season keeps them. The play-off rounds are not filed as standard
    games, so keeping the standard games drops every one of them.

    It is written as an exclusion so an unfamiliar label is dropped rather than admitted. A missing game is a visible
    gap; an invented team in the roster is a silent corruption that breaks the pairing with the odds for the whole
    competition.
    """
    competitions = event.get('competitions') or [{}]
    season_type = event.get('season', {}).get('type')
    competition_type = competitions[0].get('type', {}).get('abbreviation')
    return season_type != PRESEASON and competition_type != EXHIBITION


def _games(content: bytes, year: int) -> pd.DataFrame:
    """Return the games of a month, as the API publishes them.

    The tip-off is the instant the API gives, which it gives in UTC. Whether a game was played is the flag it carries on
    that game, never the season being over: a finished season still holds games that were postponed and never made up,
    and those have no result to learn from.
    """
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


def _form(games: pd.DataFrame) -> pd.DataFrame:
    """Return what each team had done before each of its games.

    The upcoming games are already part of the frame, so one of them carries the form of the games before it. Every
    average is shifted by one, so a game never sees its own result.

    Points scored, points conceded, and the wins that follow from them. The feed carries a score line and nothing else,
    so there is nothing else to build.
    """
    played = games['home_points'].ge(0) & games['away_points'].ge(0)
    sides = [
        pd.DataFrame(
            {
                'team': games[f'{side}_team'],
                'date': games['date'],
                'points_for': games[f'{side}_points'].where(played),
                'points_against': games[f'{other}_points'].where(played),
                'wins': (games[f'{side}_points'] > games[f'{other}_points']).where(played).astype(float),
            },
        )
        for side, other in (('home', 'away'), ('away', 'home'))
    ]
    form = pd.concat(sides).set_index(['team', 'date']).sort_index()

    averages = [f'{col}_avg' for col in FEATURES]
    latest = [f'{col}_latest_avg' for col in FEATURES]
    form[averages] = form.groupby('team')[FEATURES].expanding().mean().to_numpy()
    form[averages] = form.groupby('team')[averages].shift(1)
    form[latest] = form.groupby('team')[FEATURES].rolling(window=ROLLING_GAMES, min_periods=1).mean().to_numpy()
    form[latest] = form.groupby('team')[latest].shift(1)
    return form.drop(columns=FEATURES).reset_index()


def _snapshots(games: pd.DataFrame) -> pd.DataFrame:
    """Return the long snapshots of a season.

    A game that has not been played gets a pre-play snapshot and nothing else, so it becomes a fixture and never a
    training row.
    """
    if games.empty:
        return games
    form = _form(games)
    feature_cols = [col for col in form.columns if col.endswith('avg')]
    preplay = games[IDENTITY].copy()
    for side in ('home', 'away'):
        sided = [f'{side}_{col}' for col in feature_cols]
        side_form = form.rename(columns=dict(zip(feature_cols, sided, strict=True)))
        preplay = preplay.merge(
            side_form[['team', 'date', *sided]],
            left_on=['date', f'{side}_team'],
            right_on=['date', 'team'],
            how='left',
        ).drop(columns='team')
    preplay = preplay.assign(event_status='preplay', event_time=0)

    played = games['home_points'].ge(0) & games['away_points'].ge(0)
    postplay = games.loc[played, IDENTITY].assign(
        home_points=games.loc[played, 'home_points'].astype(int),
        away_points=games.loc[played, 'away_points'].astype(int),
        event_status='postplay',
        event_time=0,
    )
    outcomes = market_outcomes(postplay['home_points'], postplay['away_points'], MARKETS)
    postplay = pd.concat([postplay, outcomes], axis=1)

    snapshots = pd.concat([preplay, postplay], ignore_index=True)
    sided = [f'{side}_{col}' for side in ('home', 'away') for col in feature_cols]
    order = ['event_status', 'event_time', *IDENTITY, 'home_points', 'away_points', *MARKETS, *sided]
    return snapshots.reindex(columns=[col for col in order if col in snapshots.columns])


class NBAStats(BaseStatsSource):
    """The statistics of the NBA, as ESPN publishes them.

    It is free and needs no key. It carries the schedule and the final score of every game, which is what the targets
    and the form of a team are built from, and it carries them while a season is being played rather than months after
    it has ended.

    There is no draw in basketball, since a tie goes to overtime, so the outcome is two-way. There is no totals market
    either: a bookmaker sets a different line for every game, and a market whose line moves is not a column.

    The odds are another source, and there is no free one for basketball anywhere.

    Read more in the [user guide][user-guide].
    """

    name: ClassVar[str] = 'nba'

    def index_items(self: Self) -> list[RawItem]:
        """Return the seasons the competition publishes, which is free to read."""
        return [RawItem(source=self.name, key=SEASONS_KEY, url=SEASONS_URL, volatile=True)]

    def catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the seasons the competition publishes.

        A season is named by the year it ends in, as everywhere else in the library, and that is already the year the
        feed gives. There is nothing to convert here, unlike the EuroLeague, whose `E2024` is the 2024-25 season.
        """
        if not payloads:
            return []
        seasons = json.loads(payloads[0].content).get('items', [])
        years = {int(season['$ref'].rsplit('/', 1)[-1].split('?')[0]) for season in seasons if '$ref' in season}
        return sorted(
            ({'league': LEAGUE, 'division': DIVISION, 'year': year} for year in years),
            key=lambda params: params['year'],
        )

    def required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return one item per month of each selected season.

        The feed returns at most a thousand games and says nothing when it has more, so asking for a whole season, which
        is about fourteen hundred, quietly loses a quarter of it. A month holds two hundred and forty at its busiest, so
        a month is an answer the feed can give in full. Widening the window to save requests would lose games and raise
        nothing.
        """
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
                        volatile=True,
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
