"""It provides the dataloaders that turn what the sources carry into data to model."""

from __future__ import annotations

from ._base import BaseDataLoader, load_dataloader
from ._dummy import DummySoccerDataLoader
from ._factory import from_dataframe, from_snapshots
from ._sourced import DataLoader

__all__: list[str] = [
    'BaseDataLoader',
    'DataLoader',
    'DummySoccerDataLoader',
    'from_dataframe',
    'from_snapshots',
    'load_dataloader',
]
