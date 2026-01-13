"""It provides the tools to extract sports betting data."""

from __future__ import annotations

from ._base._dataloader import BaseDataLoader
from ._base._schema import (
    BaseOddsSchema,
    BaseStatsSchema,
    optional_col,
    required_col,
)

__all__: list[str] = [
    'BaseDataLoader',
    'BaseOddsSchema',
    'BaseStatsSchema',
    'optional_col',
    'required_col',
]
