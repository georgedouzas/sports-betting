"""Derive the market outcomes shared by the sports."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import pandas as pd
from sklearn.utils import check_consistent_length, column_or_1d


def derive_market_outcomes(home_points: Any, away_points: Any, markets: list[str]) -> pd.DataFrame:  # noqa: ANN401
    """Derive boolean outcomes for the given markets from home and away points.

    Args:
        home_points:
            The home team points per snapshot.
        away_points:
            The away team points per snapshot.
        markets:
            The markets to derive (e.g. `home_win`, `over_2.5`).

    Returns:
        A dataframe with one integer 0/1 column per requested market.

    Examples:
        >>> import pandas as pd
        >>> from sportsbet.sources import derive_market_outcomes
        >>> home_points = pd.Series([2, 1, 0])
        >>> away_points = pd.Series([1, 1, 3])
        >>> derive_market_outcomes(home_points, away_points, ['home_win', 'draw', 'away_win', 'over_2.5'])
           home_win  draw  away_win  over_2.5
        0         1     0         0         1
        1         0     1         0         0
        2         0     0         1         1
    """
    index = home_points.index if isinstance(home_points, pd.Series | pd.DataFrame) else None
    home_points = column_or_1d(home_points, dtype=int)
    away_points = column_or_1d(away_points, dtype=int)
    check_consistent_length(home_points, away_points)
    total = home_points + away_points
    outcomes = {}
    for market in markets:
        if market == 'home_win':
            outcomes[market] = home_points > away_points
        elif market == 'draw':
            outcomes[market] = home_points == away_points
        elif market == 'away_win':
            outcomes[market] = away_points > home_points
        elif market.startswith('over_'):
            outcomes[market] = total > float(market.removeprefix('over_'))
        elif market.startswith('under_'):
            outcomes[market] = total < float(market.removeprefix('under_'))
    return pd.DataFrame(outcomes, index=index).astype(int)
