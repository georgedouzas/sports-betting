"""It provides the dataloaders that shape the data for modelling."""

from __future__ import annotations

from ._base import BaseDataLoader, load_dataloader
from ._sourced import DataLoader

__all__: list[str] = [
    'BaseDataLoader',
    'DataLoader',
    'load_dataloader',
]
