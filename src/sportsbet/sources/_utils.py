"""Includes utilities shared by the sports."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import pandas as pd


def market_outcomes(home_goals: pd.Series, away_goals: pd.Series, markets: list[str]) -> pd.DataFrame:
    """Derive boolean outcomes for the given markets from home/away goals.

    A convenience for a source that builds its own outcome columns; the dataloader reads the markets straight from the
    data.

    It works for any sport. A sport with no draw asks only for the two win
    markets, and the outcome comes out two-way.

    Args:
        home_goals:
            The home team goals per snapshot.
        away_goals:
            The away team goals per snapshot.
        markets:
            The markets to derive (e.g. `home_win`, `over_2.5`).

    Returns:
        A dataframe with one integer 0/1 column per requested market.

    Examples:
        >>> import pandas as pd
        >>> from sportsbet.sources import market_outcomes
        >>> home_goals = pd.Series([2, 1, 0])
        >>> away_goals = pd.Series([1, 1, 3])
        >>> market_outcomes(home_goals, away_goals, ['home_win', 'draw', 'away_win', 'over_2.5'])
           home_win  draw  away_win  over_2.5
        0         1     0         0         1
        1         0     1         0         0
        2         0     0         1         1
        >>> # A sport that cannot be drawn simply does not ask for a draw, and the outcome comes out two-way.
        >>> list(market_outcomes(home_goals, away_goals, ['home_win', 'away_win']).columns)
        ['home_win', 'away_win']
    """
    total = home_goals + away_goals
    outcomes = {}
    for market in markets:
        if market == 'home_win':
            outcomes[market] = home_goals > away_goals
        elif market == 'draw':
            outcomes[market] = home_goals == away_goals
        elif market == 'away_win':
            outcomes[market] = away_goals > home_goals
        elif market.startswith('over_'):
            outcomes[market] = total > float(market.removeprefix('over_'))
        elif market.startswith('under_'):
            outcomes[market] = total < float(market.removeprefix('under_'))
    return pd.DataFrame(outcomes).astype(int)
