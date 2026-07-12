"""Implements the statistics source backed by the EuroLeague's official API."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import json
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

from .._utils import market_outcomes
from ._base import BaseStatsSource, RawItem, RawPayload

URL = 'https://api-live.euroleague.net/v2/competitions/E'
SEASONS_URL = f'{URL}/seasons'
GAMES_URL = f'{URL}/seasons/E{{season}}/games'

LEAGUE = 'Euroleague'
DIVISION = 1
MARKETS = ['home_win', 'away_win']
SEASONS_KEY = 'seasons'
ROLLING_GAMES = 3
FEATURES = ['points_for', 'points_against', 'wins']
IDENTITY = ['date', 'league', 'division', 'year', 'home_team', 'away_team']


def _games(content: bytes, year: int) -> pd.DataFrame:
    """Return the games of a season, as the API publishes them.

    The tip-off is taken from the field the API gives in UTC. The one it calls `date` is in its own head-office time,
    whatever country the game is played in — a game in Istanbul reads 18:30 there and tips off at 20:30 locally — so
    reading that one would be a guess, and a wrong guess would move every game by an hour or two without saying so.
    """
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
    training row with a result nobody knows.
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


class EuroLeagueStats(BaseStatsSource):
    """The statistics of the EuroLeague's official API.

    It is free and needs no key. It carries the schedule and the final score of every game, which is what the targets
    and the form of a team are built from.

    There is no draw in basketball, since a tie goes to overtime, so the outcome is two-way. There is no totals market
    either: the total points of a game run from about 125 to 229 and a bookmaker sets a different line for every one of
    them, and a market whose line moves is not a column.

    Read more in the [user guide][user-guide].
    """

    name: ClassVar[str] = 'euroleague'

    def index_items(self: Self) -> list[RawItem]:
        """Return the seasons the competition publishes, which is free to read."""
        return [RawItem(source=self.name, key=SEASONS_KEY, url=SEASONS_URL, volatile=True)]

    def catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the seasons the competition publishes.

        A season is named by the year it ends in, as everywhere else in the library, so the API's `E2024` is 2025.
        """
        if not payloads:
            return []
        seasons = json.loads(payloads[0].content).get('data', [])
        return sorted(
            ({'league': LEAGUE, 'division': DIVISION, 'year': int(season['year']) + 1} for season in seasons),
            key=lambda params: params['year'],
        )

    def required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return one item per selected season.

        A whole season comes back in a single response, so a season costs one request rather than one per round.
        """
        return [
            RawItem(
                source=self.name,
                key=f'{LEAGUE}_{param["division"]}_{param["year"]}',
                url=GAMES_URL.format(season=param['year'] - 1),
                volatile=True,
            )
            for param in params
            if param['league'] == LEAGUE
        ]

    def to_snapshots(self: Self, payloads: list[RawPayload]) -> pd.DataFrame:
        """Transform the seasons into the long statistics snapshots."""
        frames = []
        for payload in payloads:
            year = int(payload.item.key.rsplit('_', 1)[-1])
            games = _games(payload.content, year)
            if not games.empty:
                frames.append(_snapshots(games))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).replace({np.nan: None}).infer_objects()
