"""Schemas for validating statistics and odds data."""

from typing import Any

import pandas as pd

from .._params import IDENTITY_COLS, IDENTITY_FIELDS, STATUSES
from ..sources import BaseOddsSchema, BaseStatsSchema, optional_col, required_col


def derive_metadata(
    data: pd.DataFrame,
    value_cols: list[str],
    allow_fixed: bool = True,
) -> dict[str, dict[str, Any]]:
    """Derive per-column `include`/`fixed`/`type` metadata from a long snapshot frame.

    Every column's role is read from the data: `include` is the set of statuses at
    which the column actually carries values, and `fixed` is whether the column is
    constant within every match.

    A price is always time-varying. It belongs to a provider and to a moment, so
    it keeps them in its name even when only one provider offers it, which the
    whole odds grammar rests on.

    Args:
        data:
            A long snapshot frame with `event_status` and identity columns.
        value_cols:
            The value columns to describe (non-identity, non-event).
        allow_fixed:
            Whether a column may be constant within a match. `False` for odds.

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
        fixed = allow_fixed and bool(grouped[col].apply(_is_constant).all())
        col_type = int if pd.api.types.is_integer_dtype(data[col]) else float
        metadata[col] = {'type': col_type, 'include': include, 'fixed': fixed}
    return metadata


def _field_name(col: str) -> str:
    """Turn a column name into a valid Python identifier (``over_2.5`` -> ``over_2_5``)."""
    return col.replace('.', '_')


def build_value_namespace(metadata: dict[str, dict[str, Any]]) -> tuple[dict, dict]:
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
    value_annotations, value_namespace = build_value_namespace(metadata)
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
    value_annotations, value_namespace = build_value_namespace(metadata)
    annotations.update(value_annotations)
    namespace.update(value_namespace)
    namespace['__annotations__'] = annotations
    return type('OddsSchema', (BaseOddsSchema,), namespace)
