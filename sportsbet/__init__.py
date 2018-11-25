"""
Defines constants and helper functions/classes.
"""

from pathlib import Path
from os.path import join
from sys import path

from sklearn.metrics import SCORERS, make_scorer

# Define default path
PATH = join(str(Path.home()), '.sports-betting')
path.append(PATH)





