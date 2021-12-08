"""
The :mod:`sportsbed.datasets` provides the tools to download
and transform sports betting data.
"""

from ._base import load
from ._soccer._combined import SoccerDataLoader
from ._soccer._fd import FDDataLoader
from ._soccer._fte import FTEDataLoader
from ._soccer._dummy import DummyDataLoader

__all__ = [
    'SoccerDataLoader',
    'FDDataLoader',
    'FTEDataLoader',
    'DummyDataLoader',
    'load',
]
