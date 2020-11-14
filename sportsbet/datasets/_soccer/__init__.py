"""
The :mod:`sportsbed.datasets._soccer` includes functions
to fetch soccer historical and fixtures data.
"""

import numpy as np

HOME_WIN = lambda outputs, col1, col2, offset: outputs[col1] - outputs[col2] > offset
AWAY_WIN = lambda outputs, col1, col2, offset: outputs[col1] - outputs[col2] < -offset
DRAW = lambda outputs, col1, col2, offset: np.abs(outputs[col1] - outputs[col2]) <= offset
OVER = lambda outputs, col1, col2, offset: outputs[col1] - outputs[col2] > offset
UNDER = lambda outputs, col1, col2, offset: outputs[col1] - outputs[col2] < offset
TARGETS = [
    
    ('home_win__full_time_goals', lambda outputs: HOME_WIN(outputs, 'home_team__full_time_goals', 'away_team__full_time_goals', 0.0)),
    ('away_win__full_time_goals', lambda outputs: AWAY_WIN(outputs, 'home_team__full_time_goals', 'away_team__full_time_goals', 0.0)),
    ('draw__full_time_goals', lambda outputs: DRAW(outputs, 'home_team__full_time_goals', 'away_team__full_time_goals', 0.0)),
    ('over_1.5__full_time_goals', lambda outputs: OVER(outputs, 'home_team__full_time_goals', 'away_team__full_time_goals', 1.5)),
    ('over_2.5__full_time_goals', lambda outputs: OVER(outputs, 'home_team__full_time_goals', 'away_team__full_time_goals', 2.5)),
    ('over_3.5__full_time_goals', lambda outputs: OVER(outputs, 'home_team__full_time_goals', 'away_team__full_time_goals', 3.5)),
    ('over_4.5__full_time_goals', lambda outputs: OVER(outputs, 'home_team__full_time_goals', 'away_team__full_time_goals', 4.5)),
    ('under_1.5__full_time_goals', lambda outputs: UNDER(outputs, 'home_team__full_time_goals', 'away_team__full_time_goals', 1.5)),
    ('under_2.5__full_time_goals', lambda outputs: UNDER(outputs, 'home_team__full_time_goals', 'away_team__full_time_goals', 2.5)),
    ('under_3.5__full_time_goals', lambda outputs: UNDER(outputs, 'home_team__full_time_goals', 'away_team__full_time_goals', 3.5)),
    ('under_4.5__full_time_goals', lambda outputs: UNDER(outputs, 'home_team__full_time_goals', 'away_team__full_time_goals', 4.5)),
    
    ('home_win__full_time_adjusted_goals', lambda outputs: HOME_WIN(outputs, 'home_team__full_time_adjusted_goals', 'away_team__full_time_adjusted_goals', 0.5)),
    ('away_win__full_time_adjusted_goals', lambda outputs: AWAY_WIN(outputs, 'home_team__full_time_adjusted_goals', 'away_team__full_time_adjusted_goals', 0.5)),
    ('draw__full_time_adjusted_goals', lambda outputs: DRAW(outputs, 'home_team__full_time_adjusted_goals', 'away_team__full_time_adjusted_goals', 0.5)),
    ('over_1.5__full_time_adjusted_goals', lambda outputs: OVER(outputs, 'home_team__full_time_adjusted_goals', 'away_team__full_time_adjusted_goals', 1.5)),
    ('over_2.5__full_time_adjusted_goals', lambda outputs: OVER(outputs, 'home_team__full_time_adjusted_goals', 'away_team__full_time_adjusted_goals', 2.5)),
    ('over_3.5__full_time_adjusted_goals', lambda outputs: OVER(outputs, 'home_team__full_time_adjusted_goals', 'away_team__full_time_adjusted_goals', 3.5)),
    ('over_4.5__full_time_adjusted_goals', lambda outputs: OVER(outputs, 'home_team__full_time_adjusted_goals', 'away_team__full_time_adjusted_goals', 4.5)),
    ('under_1.5__full_time_adjusted_goals', lambda outputs: UNDER(outputs, 'home_team__full_time_adjusted_goals', 'away_team__full_time_adjusted_goals', 1.5)),
    ('under_2.5__full_time_adjusted_goals', lambda outputs: UNDER(outputs, 'home_team__full_time_adjusted_goals', 'away_team__full_time_adjusted_goals', 2.5)),
    ('under_3.5__full_time_adjusted_goals', lambda outputs: UNDER(outputs, 'home_team__full_time_adjusted_goals', 'away_team__full_time_adjusted_goals', 3.5)),
    ('under_4.5__full_time_adjusted_goals', lambda outputs: UNDER(outputs, 'home_team__full_time_adjusted_goals', 'away_team__full_time_adjusted_goals', 4.5))
    
]
