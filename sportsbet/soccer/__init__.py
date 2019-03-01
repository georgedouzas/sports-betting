"""
Defines the soccer module parameters.
"""

TARGET_TYPES_MAPPING = {
    'H': (lambda y: (y['score1'] > y['score2']).values.astype(int), lambda y: (y['avg_score1'] >= y['avg_score2'] + 1).values.astype(int)),
    'A': (lambda y: (y['score1'] < y['score2']).values.astype(int), lambda y: (y['avg_score1'] + 1 <= y['avg_score2']).values.astype(int)),
    'D': (lambda y: (y['score1'] == y['score2']).values.astype(int), lambda y: ((y['avg_score1'] - y['avg_score2']).abs() < 1).values.astype(int)),
    'H+D': (lambda y: (y['score1'] >= y['score2']).values.astype(int), lambda y: (y['avg_score1'] > y['avg_score2'] - 1).values.astype(int)),
    'A+D': (lambda y: (y['score1'] <= y['score2']).values.astype(int), lambda y: (y['avg_score1'] - 1 < y['avg_score2']).values.astype(int)),
    'H+A': (lambda y: (y['score1'] != y['score2']).values.astype(int), lambda y: ((y['avg_score1'] - y['avg_score2']).abs() >= 1).values.astype(int)),
    'over_2.5': (lambda y: (y['score1'] + y['score2'] > 2.5).values.astype(int), lambda y: (y['avg_score1'] + y['avg_score2'] > 2.5).values.astype(int)),
    'under_2.5': (lambda y: (y['score1'] + y['score2'] < 2.5).values.astype(int), lambda y: (y['score1'] + y['score2'] < 2.5).values.astype(int))
}


