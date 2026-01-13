"""Schemas for validating statistics and odds data."""

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Timedelta


def required_col() -> pa.Field:
    """Define a required column."""
    return pa.Field(nullable=False, metadata={'snapshot': True})


def optional_col(include: list[str], fixed: bool) -> pa.Field:
    """Define an optional column."""
    return pa.Field(nullable=True, metadata={'include': include, 'fixed': fixed})


class BaseSchema(pa.DataFrameModel):
    event_status: str = required_col()
    event_time: Timedelta = required_col()

    @pa.dataframe_check
    @classmethod
    def check_event_time_vs_status(cls: type[pa.DataFrameModel], df: pd.DataFrame) -> pd.Series:
        preplay_check = (df['event_status'] == 'preplay') & (df['event_time'] >= pd.Timedelta(0))
        inplay_check = (df['event_status'] == 'inplay') & (df['event_time'] > pd.Timedelta(0))
        postplay_check = (df['event_status'] == 'postplay') & (df['event_time'] == pd.Timedelta(0))
        status_check = df['event_status'].isin(['preplay', 'inplay', 'postplay'])
        return status_check & (preplay_check | inplay_check | postplay_check)

    @classmethod
    def snapshot_cols(cls: type[pa.DataFrameModel]) -> list[str]:
        schema = cls.to_schema()
        return [
            name
            for name, col in schema.columns.items()
            if ((col.properties or {}).get('metadata') or {}).get('snapshot', False)
        ]

    @classmethod
    def col_metadata(cls: type[pa.DataFrameModel], col: str) -> list[str]:
        schema_col = dict(cls.to_schema().columns)[col]
        return (schema_col.properties or {}).get('metadata') or {}

    @pa.dataframe_check
    @classmethod
    def snapshot_unique(cls: type[pa.DataFrameModel], df: pd.DataFrame) -> bool:
        return not df.duplicated(subset=cls.snapshot_cols()).any()

    class Config:
        strict = True


class BaseStatsSchema(BaseSchema):
    pass


class BaseOddsSchema(BaseSchema):

    @classmethod
    def odds_cols(cls) -> list[str]:
        schema_cols = list(cls.to_schema().columns.keys())
        return [col for col in schema_cols if col not in cls.snapshot_cols() and col != 'provider']

    @pa.dataframe_check
    @classmethod
    def postplay_missing_odds(cls, df: pd.DataFrame) -> pd.Series:
        odds_cols = cls.odds_cols()
        if not odds_cols:
            return pd.Series(True, index=df.index)
        is_post = df['event_status'].eq('postplay')
        ok_post = df.loc[is_post, odds_cols].isna().all(axis=1)
        out = pd.Series(True, index=df.index)
        out.loc[is_post] = ok_post
        return out
