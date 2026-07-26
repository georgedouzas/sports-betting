"""Validate statistics and odds snapshots against their schemas."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import Any, Self

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Timedelta


def required_col(alias: str | None = None) -> Any:  # noqa: ANN401
    """Define a required snapshot-identity column.

    Args:
        alias:
            The column name to use when it differs from the field's Python
            identifier.

    Examples:
        >>> from sportsbet.sources import BaseStatsSchema, required_col
        >>>
        >>> class MyStatsSchema(BaseStatsSchema):
        ...     'The columns a statistics feed of your own must always carry.'
        ...
        ...     home_team: str = required_col()
        ...     away_team: str = required_col()
        >>>
        >>> # A required column may not be missing, so an identity is never half of one.
        >>> MyStatsSchema.to_schema().columns['home_team'].nullable
        False
    """
    return pa.Field(nullable=False, metadata={'snapshot': True}, alias=alias)


def optional_col(include: list[str], fixed: bool, alias: str | None = None) -> Any:  # noqa: ANN401
    """Define an optional feature or odds column.

    Args:
        include:
            The event statuses at which the column is meaningful.
        fixed:
            Whether the column is time-invariant within a match.
        alias:
            The column name to use when it differs from the field's Python
            identifier.

    Examples:
        >>> from sportsbet.sources import BaseStatsSchema, optional_col, required_col
        >>>
        >>> class MyStatsSchema(BaseStatsSchema):
        ...     'The columns a statistics feed of your own may carry.'
        ...
        ...     home_team: str = required_col()
        ...     away_team: str = required_col()
        ...     home_goals: float = optional_col(include=['inplay', 'postplay'], fixed=False)
        ...     stadium_capacity: float = optional_col(include=['preplay'], fixed=True)
        >>>
        >>> # There is no score before the match starts.
        >>> MyStatsSchema.to_schema().columns['home_goals'].metadata['include']
        ['inplay', 'postplay']
        >>> # A stadium does not change size at half time, so it is carried once rather than per moment.
        >>> MyStatsSchema.to_schema().columns['stadium_capacity'].metadata['fixed']
        True
    """
    return pa.Field(nullable=True, metadata={'include': include, 'fixed': fixed}, alias=alias)


class _BaseSchema(pa.DataFrameModel):
    """Sport-agnostic base schema for event snapshots."""

    event_status: str = required_col()
    event_time: Timedelta = required_col()

    @pa.dataframe_check
    @classmethod
    def check_event_time_vs_status(cls: type[Self], df: pd.DataFrame) -> pd.Series:
        """Check the event time is consistent with the event status."""
        preplay_check = (df['event_status'] == 'preplay') & (df['event_time'] >= pd.Timedelta(0))
        inplay_check = (df['event_status'] == 'inplay') & (df['event_time'] > pd.Timedelta(0))
        postplay_check = (df['event_status'] == 'postplay') & (df['event_time'] == pd.Timedelta(0))
        status_check = df['event_status'].isin(['preplay', 'inplay', 'postplay'])
        return status_check & (preplay_check | inplay_check | postplay_check)

    @classmethod
    def list_snapshot_cols(cls: type[Self]) -> list[str]:
        """Return the snapshot-identity columns."""
        schema = cls.to_schema()
        return [
            name
            for name, col in schema.columns.items()
            if ((col.properties or {}).get('metadata') or {}).get('snapshot', False)
        ]

    @classmethod
    def get_col_metadata(cls: type[Self], col: str) -> dict[str, Any]:
        """Return the `include`/`fixed`/`snapshot` metadata of a column."""
        schema_col = dict(cls.to_schema().columns)[col]
        return (schema_col.properties or {}).get('metadata') or {}

    @pa.dataframe_check
    @classmethod
    def check_snapshot_unique(cls: type[Self], df: pd.DataFrame) -> bool:
        """Check that no two rows share the same snapshot identity."""
        return not df.duplicated(subset=cls.list_snapshot_cols()).any()

    class Config:
        strict = True


class BaseStatsSchema(_BaseSchema):
    """Base schema for statistics snapshots.

    Examples:
        >>> from sportsbet.sources import BaseStatsSchema, optional_col, required_col
        >>>
        >>> class MyStatsSchema(BaseStatsSchema):
        ...     'The statistics of a feed of your own.'
        ...
        ...     home_team: str = required_col()
        ...     away_team: str = required_col()
        ...     home_goals: float = optional_col(include=['inplay', 'postplay'], fixed=False)
    """


class BaseOddsSchema(_BaseSchema):
    """Base schema for odds snapshots.

    Examples:
        >>> from sportsbet.sources import BaseOddsSchema, optional_col, required_col
        >>>
        >>> class MyOddsSchema(BaseOddsSchema):
        ...     'The odds of a feed of your own.'
        ...
        ...     home_team: str = required_col()
        ...     away_team: str = required_col()
        ...     provider: str = required_col()
        ...     home_win: float = optional_col(include=['preplay', 'inplay'], fixed=False)
        ...     away_win: float = optional_col(include=['preplay', 'inplay'], fixed=False)
    """

    @classmethod
    def list_odds_cols(cls) -> list[str]:
        """Return the odds (market) columns."""
        schema_cols = list(cls.to_schema().columns.keys())
        return [col for col in schema_cols if col not in cls.list_snapshot_cols() and col != 'provider']

    @pa.dataframe_check
    @classmethod
    def check_postplay_missing_odds(cls, df: pd.DataFrame) -> pd.Series:
        """Check that post-match snapshots carry no odds."""
        odds_cols = cls.list_odds_cols()
        if not odds_cols:
            return pd.Series(True, index=df.index)
        is_post = df['event_status'].eq('postplay')
        ok_post = df.loc[is_post, odds_cols].isna().all(axis=1)
        out = pd.Series(True, index=df.index)
        out.loc[is_post] = ok_post
        return out
