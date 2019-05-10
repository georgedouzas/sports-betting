"""
Defines the soccer module parameters.
"""

import numpy as np

TARGETS = {
    'H': lambda score1, score2: score1 > score2,
    'A': lambda score1, score2: score2 > score1,
    'D': lambda score1, score2: score1 == score2,
    'H+D': lambda score1, score2: score1 >= score2,
    'A+D': lambda score1, score2: score2 >= score1,
    'H+A': lambda score1, score2: score1 != score2,
    'over_2.5': lambda score1, score2: score1 + score2 > 2.5,
    'under_2.5': lambda score1, score2: score1 + score2 < 2.5
}



