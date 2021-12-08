"""Collection of sports betting AI tools.

``sports-betting`` is a set of tools for sports betting. It
provides python functions to download data and test the
performance of machine learning models.

Subpackages
-----------
datasets
    Module which provides functions to download and transform sports betting
    datasets.
"""

from . import datasets
from ._version import __version__

__all__ = ['datasets', '__version__']
