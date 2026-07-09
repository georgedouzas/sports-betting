"""Includes utilities for soccer data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import Any

import pandas as pd

IDENTITY_COLS = ['date', 'league', 'division', 'year', 'home_team', 'away_team']
EVENT_COLS = ['event_status', 'event_time']
GOAL_COLS = ['home_goals', 'away_goals']
STATUSES = ['preplay', 'inplay', 'postplay']


def odds_columns(columns: list[str]) -> list[str]:
    """Return the odds columns of a wide frame (the ``{provider}__{market}`` ones)."""
    return [col for col in columns if '__' in col]


def parse_odds_column(col: str) -> tuple[str, str]:
    """Split a ``{provider}__{market}`` odds column into its provider and market."""
    provider, market = col.split('__', maxsplit=1)
    return provider, market


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


def derive_metadata(data: pd.DataFrame, value_cols: list[str]) -> dict[str, dict[str, Any]]:
    """Derive per-column `include`/`fixed`/`type` metadata from a long snapshot frame.

    Nothing is assumed about a column's role: `include` is the set of statuses at
    which the column actually carries values, and `fixed` is whether the column is
    constant within every match.

    Args:
        data:
            A long snapshot frame with `event_status` and identity columns.
        value_cols:
            The value columns to describe (non-identity, non-event).

    Returns:
        Mapping of column to `{'type', 'include', 'fixed'}`.
    """

    def _is_constant(values: pd.Series) -> bool:
        non_null = values.dropna()
        return non_null.empty or bool(non_null.min() == non_null.max())

    grouped = data.groupby(IDENTITY_COLS, dropna=False)
    metadata = {}
    for col in value_cols:
        include = [status for status in STATUSES if data.loc[data['event_status'] == status, col].notna().any()]
        fixed = bool(grouped[col].apply(_is_constant).all())
        col_type = int if pd.api.types.is_integer_dtype(data[col]) else float
        metadata[col] = {'type': col_type, 'include': include, 'fixed': fixed}
    return metadata
