"""
Defines constants and helper functions/classes.
"""

from pathlib import Path
from os.path import join
from sys import path

from sklearn.metrics import SCORERS, make_scorer

from .utils import total_profit_score, mean_profit_score

# Define default path
PATH = join(str(Path.home()), '.sports-betting')

# Append default path
path.append(PATH)

# Append scorers
for name, scorer in {'total_profit': total_profit_score, 'mean_profit': mean_profit_score}.items():
    SCORERS[name] = make_scorer(scorer)


