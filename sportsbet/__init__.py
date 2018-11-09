"""
Defines constants and helper functions/classes.
"""

from pathlib import Path
from os.path import join
from sys import path

from sklearn.metrics import SCORERS, make_scorer

from .utils import mean_profit_score, f1_multi

# Define default path
PATH = join(str(Path.home()), '.sports-betting')

# Append default path
path.append(PATH)

# Append scorers
named_scorers = {'mean_profit': mean_profit_score, 'f1_multi': f1_multi}
for name, scorer in named_scorers.items():
    SCORERS[name] = make_scorer(scorer)


