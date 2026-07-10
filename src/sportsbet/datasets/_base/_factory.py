"""In-memory dataloader and factory functions built from user-provided data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import Self

import pandas as pd

from ... import ParamGrid
from ._dataloader import BaseDataLoader
from ._schema import IDENTITY_COLS, odds_columns, parse_odds_column


class _SnapshotsDataLoader(BaseDataLoader):
    """Concrete dataloader backed by in-memory snapshots, used by the factory functions."""

    def _snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._provided_snapshots is None:
            msg = 'No snapshots were provided.'
            raise NotImplementedError(msg)
        return self._provided_snapshots


def _wide_to_snapshots(
    data: pd.DataFrame,
    event_status: str,
    event_time: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a wide single-moment frame into long `stats`/`odds` snapshots."""
    odds_cols = odds_columns(list(data.columns))
    stats = data.drop(columns=odds_cols).assign(event_status=event_status, event_time=event_time)
    by_provider: dict[str, dict[str, str]] = {}
    for col in odds_cols:
        provider, market = parse_odds_column(col)
        by_provider.setdefault(provider, {})[market] = col
    records = []
    for _, row in data.iterrows():
        identity = {col: row[col] for col in IDENTITY_COLS}
        for provider, markets in by_provider.items():
            record = {**identity, 'event_status': event_status, 'event_time': event_time, 'provider': provider}
            record.update({market: row[col] for market, col in markets.items()})
            records.append(record)
    return stats, pd.DataFrame(records)


def from_snapshots(
    stats: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    param_grid: ParamGrid | None = None,
) -> BaseDataLoader:
    """Build a dataloader from canonical long `stats` and `odds` snapshots.

    Use this when the data already follows the exported long format, i.e. one row
    per match and moment with explicit `event_status`/`event_time` columns (`stats`
    carrying the values, `odds` carrying `{provider}` and the markets). No moment is
    assumed — every row states its own.

    Args:
        stats:
            Long statistics snapshots.
        odds:
            Long odds snapshots.
        param_grid:
            Optional selection, mirroring the dataloader constructor.

    Returns:
        A dataloader that reads the provided snapshots instead of downloading them.

    Examples:
        >>> import pandas as pd
        >>> from sportsbet.datasets import from_snapshots
        >>> identity = dict(date='2024-08-16', league='England', division=1, year=2025,
        ...                 home_team='A', away_team='B')
        >>> stats = pd.DataFrame([
        ...     {'event_status': 'preplay', 'event_time': 0, **identity, 'home_win': None},
        ...     {'event_status': 'inplay', 'event_time': 45, **identity, 'home_win': 1},
        ...     {'event_status': 'postplay', 'event_time': 0, **identity, 'home_win': 1},
        ... ])
        >>> odds = pd.DataFrame([
        ...     {'event_status': 'preplay', 'event_time': 0, **identity, 'provider': 'bookie', 'home_win': 1.8},
        ... ])
        >>> loader = from_snapshots(stats, odds)
        >>> loader.get_odds_types()
        ['bookie']
        >>> X, Y, O = loader.extract_train_data(odds_type='bookie')
        >>> list(Y.columns)
        ['home_win__postplay__0min']
    """
    loader = _SnapshotsDataLoader(param_grid=param_grid)
    loader._provided_snapshots = (stats, odds)
    return loader


def from_dataframe(
    data: pd.DataFrame,
    *,
    event_status: str,
    event_time: pd.Timedelta,
    param_grid: ParamGrid | None = None,
) -> BaseDataLoader:
    """Build a dataloader from a user's wide match table taken at a single moment.

    Every row of `data` is treated as a snapshot at the caller-declared
    `event_status`/`event_time` — no moment is assumed. `data` must carry the
    identity columns (`date`, `league`, `division`, `year`, `home_team`,
    `away_team`), any number of value columns (goals, market outcomes, features),
    and `{provider}__{market}` odds columns. For several moments, provide long
    snapshots to [`from_snapshots`][sportsbet.datasets.from_snapshots] instead, or
    call this per moment.

    Args:
        data:
            One row per match at a single moment.
        event_status:
            The status the rows represent, e.g. `'preplay'` or `'postplay'`.
        event_time:
            The time into the match the rows represent.
        param_grid:
            Optional selection, mirroring the dataloader constructor.

    Returns:
        A dataloader that reads the provided data instead of downloading it.
    """
    loader = _SnapshotsDataLoader(param_grid=param_grid)
    loader._provided_snapshots = _wide_to_snapshots(data, event_status, event_time)
    return loader
