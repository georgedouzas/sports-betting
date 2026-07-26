"""Read the sample data shared by its statistics and odds sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import io
from pathlib import Path
from typing import ClassVar, Self

import pandas as pd
from sklearn.model_selection import ParameterGrid

from ...core import ParamGrid
from .._base import RawItem, RawPayload

DATA = Path(__file__).parent.parent / 'data'
PARAMS: ParamGrid = {'league': ['England', 'Spain'], 'division': [1], 'year': [2024]}


class _SampleSource:
    """The half of a sample source shared by its statistics and odds, backed by the bundled files."""

    name: ClassVar[str] = 'sample_soccer'
    kind: ClassVar[str]
    sport: ClassVar[str | None] = 'soccer'

    def list_index_items(self: Self, selection: ParamGrid | None = None) -> list[RawItem]:
        """Return no items."""
        return []

    def read_catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the leagues, divisions and seasons the sample carries."""
        return list(ParameterGrid(PARAMS))

    def list_required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return the bundled file of every selected season."""
        items = []
        for param in params:
            key = f'{param["league"]}_{param["division"]}_{param["year"]}_{self.kind}'
            path = DATA / f'{key}.csv.gz'
            if path.exists():
                items.append(RawItem(source=self.name, key=key, url=path.as_uri()))
        return items

    def to_snapshots(self: Self, payloads: list[RawPayload]) -> pd.DataFrame:
        """Return the long snapshots of the bundled files."""
        if not payloads:
            return pd.DataFrame()
        frames = [pd.read_csv(io.BytesIO(payload.content), compression='gzip') for payload in payloads]
        snapshots = pd.concat(frames, ignore_index=True)
        snapshots['date'] = pd.to_datetime(snapshots['date'], utc=True)
        snapshots['event_time'] = pd.to_timedelta(snapshots['event_time'])
        return snapshots
