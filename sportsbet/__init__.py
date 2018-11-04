"""
Defines constants and helper functions/classes.
"""

from collections import Counter
from pathlib import Path
from os.path import join
from sys import path

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import SCORERS, make_scorer

# Define default path
PATH = join(str(Path.home()), '.sports-betting')

# Append default path
path.append(PATH)
