"""
Defines the soccer module parameters.
"""

import numpy as np

TARGETS = ['H', 'A', 'D', 'H+D', 'A+D', 'H+A', 'over_2.5', 'under_2.5']
OFFSETS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
SCORES_MAPPING = {
    'H': lambda score1, score2, offset: score1 > score2 + offset, 
    'A': lambda score1, score2, offset: score2 > score1 + offset,
    'D': lambda score1, score2, offset: np.abs(score1 - score2) <= offset,
    'H+D': lambda score1, score2, offset: (score1 > score2 + offset) | (np.abs(score1 - score2) <= offset), 
    'A+D': lambda score1, score2, offset: (score2 > score1 + offset) | (np.abs(score1 - score2) <= offset),
    'H+A': lambda score1, score2, offset: (score1 > score2 + offset) | (score2 > score1 + offset),
    'over_2.5': lambda score1, score2, offset: score1 + score2 > 2.5 + offset,
    'under_2.5': lambda score1, score2, offset: score1 + score2 < 2.5 - offset
}
FEATURES = ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'BbAv>2.5', 'BbAv<2.5', 'BbAHh', 'BbAvAHH', 'BbAvAHA', 'quality', 'importance', 'rating', 'sum_proj_score']

