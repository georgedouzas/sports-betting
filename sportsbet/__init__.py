"""Collection of sports betting AI tools.

``sports-betting`` is a set of tools for sports betting. It
provides python functions to download data and test the 
performance of machine learning models.

Subpackages
-----------
datasets
    Module which provides functions to down and transform sports betting 
    datasets.
backtesting
    Module which provides functions to test the performance of machine
    learning models.
utils
    Module including various utilities.
"""

from . import datasets
from . import backtesting
from . import utils
from ._version import __version__

__all__ = [
    'datasets',
    'backtesting',
    'utils',
    '__version__'
]