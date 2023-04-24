"""It provides the tools to extract sports betting data."""

from __future__ import annotations

from ._base import load_dataloader
from ._dummy import DummySoccerDataLoader
from ._soccer._data import SoccerDataLoader

__all__: list[str] = [
    'SoccerDataLoader',
    'DummySoccerDataLoader',
    'load_dataloader',
]
