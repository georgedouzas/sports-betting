"""
Defines the soccer module parameters.
"""

TARGET_TYPES_MAPPING = {
    'H': lambda y: (y['score1'] > y['score2']).values.astype(int),
    'D': lambda y: (y['score1'] == y['score2']).values.astype(int),
    'A': lambda y: (y['score1'] < y['score2']).values.astype(int),
    'A+D': lambda y: (y['score1'] <= y['score2']).values.astype(int),
    'H+D': lambda y: (y['score1'] >= y['score2']).values.astype(int),
    'over_2.5': lambda y: (y['score1'] + y['score2'] > 2.5).values.astype(int),
    'under_2.5': lambda y: (y['score1'] + y['score2'] < 2.5).values.astype(int),
    'both_score': lambda y: (y['score1'] * y['score2'] > 0).values.astype(int),
    'no_nil_score': lambda y: (y['score1'] + y['score2'] > 0).values.astype(int)
}


