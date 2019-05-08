"""
Defines the soccer module parameters.
"""

import numpy as np

TARGETS = [
    ('H', lambda score1, score2, offset: score1 > score2 + offset, 0.0),
    ('A', lambda score1, score2, offset: score2 > score1 + offset, 0.0),
    ('D', lambda score1, score2, offset: np.abs(score1 - score2) <= offset, 0.0),
    ('H+D', lambda score1, score2, offset: (score1 > score2 + offset) | (np.abs(score1 - score2) <= offset), 0.0),
    ('A+D', lambda score1, score2, offset: (score2 > score1 + offset) | (np.abs(score1 - score2) <= offset), 0.0),
    ('H+A', lambda score1, score2, offset: (score1 > score2 + offset) | (score2 > score1 + offset), 0.0),
    ('over_2.5', lambda score1, score2, offset: score1 + score2 > 2.5 + offset, 0.0),
    ('under_2.5', lambda score1, score2, offset: score1 + score2 < 2.5 - offset, 0.0)
]



