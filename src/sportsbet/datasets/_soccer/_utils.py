"""Includes utilities for soccer data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import pandas as pd


def market_outcomes(home_goals: pd.Series, away_goals: pd.Series, markets: list[str]) -> pd.DataFrame:
    """Derive boolean outcomes for the given markets from home/away goals.

    A convenience for producing market-outcome columns (e.g. when preparing a
    frame for `from_dataframe`); the loader itself never derives markets.

    Args:
        home_goals:
            The home team goals per snapshot.
        away_goals:
            The away team goals per snapshot.
        markets:
            The markets to derive (e.g. `home_win`, `over_2.5`).

    Returns:
        A dataframe with one integer 0/1 column per requested market.
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
