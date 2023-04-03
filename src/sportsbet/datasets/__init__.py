"""It provides the tools to extract sports betting data."""

from ._base import load
from ._dummy import DummySoccerDataLoader
from ._soccer._data import SoccerDataLoader

__all__: list[str] = [
    'SoccerDataLoader',
    'DummySoccerDataLoader',
    'load',
]
