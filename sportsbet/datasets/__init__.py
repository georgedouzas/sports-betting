"""
The :mod:`sportsbet.datasets` module provides the
tools to extract sports betting data.
"""

from ._base import load
from ._dummy import DummySoccerDataLoader, DummyBasketballDataLoader
from ._soccer._data import SoccerDataLoader

__all__ = [
    'DummySoccerDataLoader',
    'DummyBasketballDataLoader',
    'SoccerDataLoader',
    'load',
]
