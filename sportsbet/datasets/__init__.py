"""
The :mod:`sportsbed.datasets` provides the tools to download 
and transform sports betting data.
"""

from ._soccer._fte import load_from_five_thirty_eight_soccer_data
from ._soccer._fd import load_from_football_data_soccer_data

__all__ = [
    'load_from_five_thirty_eight_soccer_data',
    'load_from_football_data_soccer_data'
]
