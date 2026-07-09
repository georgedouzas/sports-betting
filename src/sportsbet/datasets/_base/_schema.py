"""Schemas for validating statistics and odds data."""

from typing import Any, Self

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Timedelta


def required_col(alias: str | None = None) -> Any:  # noqa: ANN401
    """Define a required (snapshot-identity) column.

    Args:
        alias:
            The column name to use when it differs from the field's Python
            identifier (e.g. a dotted market name).
    """
    return pa.Field(nullable=False, metadata={'snapshot': True}, alias=alias)


def optional_col(include: list[str], fixed: bool, alias: str | None = None) -> Any:  # noqa: ANN401
    """Define an optional (feature/odds) column.

    Args:
        include:
            The event statuses at which the column is meaningful.
        fixed:
            Whether the column is time-invariant within a match.
        alias:
            The column name to use when it differs from the field's Python
            identifier (e.g. a dotted market name).
    """
    return pa.Field(nullable=True, metadata={'include': include, 'fixed': fixed}, alias=alias)


class BaseSchema(pa.DataFrameModel):
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
    def snapshot_cols(cls: type[Self]) -> list[str]:
        """Return the snapshot-identity columns."""
        schema = cls.to_schema()
        return [
            name
            for name, col in schema.columns.items()
            if ((col.properties or {}).get('metadata') or {}).get('snapshot', False)
        ]

    @classmethod
    def col_metadata(cls: type[Self], col: str) -> dict[str, Any]:
        """Return the `include`/`fixed`/`snapshot` metadata of a column."""
        schema_col = dict(cls.to_schema().columns)[col]
        return (schema_col.properties or {}).get('metadata') or {}

    @pa.dataframe_check
    @classmethod
    def snapshot_unique(cls: type[Self], df: pd.DataFrame) -> bool:
        """Check that no two rows share the same snapshot identity."""
        return not df.duplicated(subset=cls.snapshot_cols()).any()

    class Config:
        strict = True


class BaseStatsSchema(BaseSchema):
    """Base schema for statistics snapshots."""


class BaseOddsSchema(BaseSchema):
    """Base schema for odds snapshots."""

    @classmethod
    def odds_cols(cls) -> list[str]:
        """Return the odds (market) columns."""
        schema_cols = list(cls.to_schema().columns.keys())
        return [col for col in schema_cols if col not in cls.snapshot_cols() and col != 'provider']

    @pa.dataframe_check
    @classmethod
    def postplay_missing_odds(cls, df: pd.DataFrame) -> pd.Series:
        """Check that post-match snapshots carry no odds (settled outcomes)."""
        odds_cols = cls.odds_cols()
        if not odds_cols:
            return pd.Series(True, index=df.index)
        is_post = df['event_status'].eq('postplay')
        ok_post = df.loc[is_post, odds_cols].isna().all(axis=1)
        out = pd.Series(True, index=df.index)
        out.loc[is_post] = ok_post
        return out
