"""
The :mod:`sportsbet.datasets` module provides the
tools to extract sports betting data.
"""

from ._base import load
from ._soccer._data import SoccerDataLoader
from ._dummy import DummySoccerDataLoader

__all__ = [
    'SoccerDataLoader',
    'DummySoccerDataLoader',
    'load',
]
