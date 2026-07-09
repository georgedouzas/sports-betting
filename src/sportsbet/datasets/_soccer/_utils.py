"""Includes utilities for soccer data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import pandas as pd

OVER_UNDER_LINES: tuple[float, ...] = (2.5, 3.5)
MARKETS: list[str] = [
    'home_win',
    'draw',
    'away_win',
    *[f'over_{line}' for line in OVER_UNDER_LINES],
    *[f'under_{line}' for line in OVER_UNDER_LINES],
]


def market_outcomes(home_goals: pd.Series, away_goals: pd.Series) -> pd.DataFrame:
    """Derive boolean market outcomes from home/away goals at each snapshot.

    Args:
        home_goals:
            The home team goals per snapshot.
        away_goals:
            The away team goals per snapshot.

    Returns:
        outcomes:
            A dataframe with one boolean column per market in `MARKETS`.
    """
    total = home_goals + away_goals
    outcomes = {
        'home_win': home_goals > away_goals,
        'draw': home_goals == away_goals,
        'away_win': away_goals > home_goals,
    }
    for line in OVER_UNDER_LINES:
        outcomes[f'over_{line}'] = total > line
        outcomes[f'under_{line}'] = total < line
    return pd.DataFrame(outcomes).astype(int)
