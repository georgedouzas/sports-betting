"""
The :mod:`sportsbet` package is a collection of sports betting AI
tools. It provides classes to extract sports betting data and create
predictive models. It contains two main modules:

datasets
    Module which provides classes to extract sports betting
    datasets.
evaluation
    Module which provides classes to create and evaluate sports betting
    predictive models.
"""

from . import datasets, evaluation
from ._version import __version__

__all__ = ['datasets', 'evaluation', '__version__']
