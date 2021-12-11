"""
The :mod:`sportsbet.datasets` module provides the
tools to download and transform sports betting data.
"""

from ._base import load
from ._soccer._combined import SoccerDataLoader
from ._soccer._fd import FDSoccerDataLoader
from ._soccer._fte import FTESoccerDataLoader
from ._soccer._dummy import DummySoccerDataLoader

__all__ = [
    'SoccerDataLoader',
    'FDSoccerDataLoader',
    'FTESoccerDataLoader',
    'DummySoccerDataLoader',
    'load',
]
