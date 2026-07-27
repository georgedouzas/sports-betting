"""Build the basketball statistics shared by its competition sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import pandas as pd

from ...core import IDENTITY_COLS
from .._utils import derive_market_outcomes

DIVISION = 1
SEASONS_KEY = 'seasons'
MARKETS = ['home_win', 'away_win']
ROLLING_GAMES = 3
FEATURES = ['points_for', 'points_against', 'wins']


def _form(games: pd.DataFrame) -> pd.DataFrame:
    """Return each team's scoring form from the games before each one."""
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
    """Return the long snapshots of a season, an unplayed game keeping only its pre-play row."""
    if games.empty:
        return games
    form = _form(games)
    feature_cols = [col for col in form.columns if col.endswith('avg')]
    preplay = games[IDENTITY_COLS].copy()
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
    postplay = games.loc[played, IDENTITY_COLS].assign(
        home_points=games.loc[played, 'home_points'].astype(int),
        away_points=games.loc[played, 'away_points'].astype(int),
        event_status='postplay',
        event_time=0,
    )
    outcomes = derive_market_outcomes(postplay['home_points'], postplay['away_points'], MARKETS)
    postplay = pd.concat([postplay, outcomes], axis=1)

    snapshots = pd.concat([preplay, postplay], ignore_index=True)
    sided = [f'{side}_{col}' for side in ('home', 'away') for col in feature_cols]
    order = ['event_status', 'event_time', *IDENTITY_COLS, 'home_points', 'away_points', *MARKETS, *sided]
    return snapshots.reindex(columns=[col for col in order if col in snapshots.columns])
