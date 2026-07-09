"""Schemas for validating statistics and odds data."""

from typing import Annotated, Any, Self

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Timedelta

# Columns that identify a match snapshot and place it in time.
IDENTITY_COLS = ['date', 'league', 'division', 'year', 'home_team', 'away_team']
EVENT_COLS = ['event_status', 'event_time']
# The possible event statuses, ordered.
STATUSES = ['preplay', 'inplay', 'postplay']
# Types of the identity columns, used to build derived schemas.
IDENTITY_FIELDS = {
    'date': Annotated[pd.DatetimeTZDtype, 'ns', 'utc'],
    'league': str,
    'division': int,
    'year': int,
    'home_team': str,
    'away_team': str,
}


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


def odds_columns(columns: list[str]) -> list[str]:
    """Return the odds columns of a wide frame (the ``{provider}__{market}`` ones)."""
    return [col for col in columns if '__' in col]


def parse_odds_column(col: str) -> tuple[str, str]:
    """Split a ``{provider}__{market}`` odds column into its provider and market."""
    provider, market = col.split('__', maxsplit=1)
    return provider, market


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


def _field_name(col: str) -> str:
    """Turn a column name into a valid Python identifier (``over_2.5`` -> ``over_2_5``)."""
    return col.replace('.', '_')


def _value_namespace(metadata: dict[str, dict[str, Any]]) -> tuple[dict, dict]:
    """Build the annotations and fields for the value columns from their metadata."""
    annotations: dict = {}
    namespace: dict = {}
    for col, meta in metadata.items():
        field = _field_name(col)
        annotations[field] = meta['type']
        alias = col if field != col else None
        namespace[field] = optional_col(meta['include'], fixed=meta['fixed'], alias=alias)
    return annotations, namespace


def build_stats_schema(metadata: dict[str, dict[str, Any]]) -> type[BaseStatsSchema]:
    """Build a statistics schema from the derived value-column metadata."""
    annotations: dict = dict(IDENTITY_FIELDS)
    namespace: dict = {col: required_col() for col in IDENTITY_FIELDS}
    value_annotations, value_namespace = _value_namespace(metadata)
    annotations.update(value_annotations)
    namespace.update(value_namespace)
    namespace['__annotations__'] = annotations
    return type('StatsSchema', (BaseStatsSchema,), namespace)


def build_odds_schema(metadata: dict[str, dict[str, Any]]) -> type[BaseOddsSchema]:
    """Build an odds schema from the derived market-column metadata."""
    annotations: dict = dict(IDENTITY_FIELDS)
    namespace: dict = {col: required_col() for col in IDENTITY_FIELDS}
    annotations['provider'] = str
    namespace['provider'] = optional_col(['preplay'], fixed=True)
    value_annotations, value_namespace = _value_namespace(metadata)
    annotations.update(value_annotations)
    namespace.update(value_namespace)
    namespace['__annotations__'] = annotations
    return type('OddsSchema', (BaseOddsSchema,), namespace)
