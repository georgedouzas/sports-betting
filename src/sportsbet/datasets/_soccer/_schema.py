"""Builders for soccer statistics and odds schemas derived from the snapshot data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd

from .._base._schema import BaseOddsSchema, BaseStatsSchema, optional_col, required_col

_IDENTITY_FIELDS = {
    'date': Annotated[pd.DatetimeTZDtype, 'ns', 'utc'],
    'league': str,
    'division': int,
    'year': int,
    'home_team': str,
    'away_team': str,
}


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
    annotations: dict = dict(_IDENTITY_FIELDS)
    namespace: dict = {col: required_col() for col in _IDENTITY_FIELDS}
    value_annotations, value_namespace = _value_namespace(metadata)
    annotations.update(value_annotations)
    namespace.update(value_namespace)
    namespace['__annotations__'] = annotations
    return type('SoccerStatsSchema', (BaseStatsSchema,), namespace)


def build_odds_schema(metadata: dict[str, dict[str, Any]]) -> type[BaseOddsSchema]:
    """Build an odds schema from the derived market-column metadata."""
    annotations: dict = dict(_IDENTITY_FIELDS)
    namespace: dict = {col: required_col() for col in _IDENTITY_FIELDS}
    annotations['provider'] = str
    namespace['provider'] = optional_col(['preplay'], fixed=True)
    value_annotations, value_namespace = _value_namespace(metadata)
    annotations.update(value_annotations)
    namespace.update(value_namespace)
    namespace['__annotations__'] = annotations
    return type('SoccerOddsSchema', (BaseOddsSchema,), namespace)
