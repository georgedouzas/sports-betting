"""
The :mod:`sportsbed.datasets._soccer` includes functions
to fetch soccer historical and fixtures data.
"""

OUTCOMES = [
    (
        'home_win__full_time_goals',
        lambda outputs: outputs['home_team__full_time_goals']
        > outputs['away_team__full_time_goals'],
    ),
    (
        'away_win__full_time_goals',
        lambda outputs: outputs['home_team__full_time_goals']
        < outputs['away_team__full_time_goals'],
    ),
    (
        'draw__full_time_goals',
        lambda outputs: outputs['home_team__full_time_goals']
        == outputs['away_team__full_time_goals'],
    ),
    (
        'over_1.5__full_time_goals',
        lambda outputs: outputs['home_team__full_time_goals']
        + outputs['away_team__full_time_goals']
        > 1.5,
    ),
    (
        'over_2.5__full_time_goals',
        lambda outputs: outputs['home_team__full_time_goals']
        + outputs['away_team__full_time_goals']
        > 2.5,
    ),
    (
        'over_3.5__full_time_goals',
        lambda outputs: outputs['home_team__full_time_goals']
        + outputs['away_team__full_time_goals']
        > 3.5,
    ),
    (
        'over_4.5__full_time_goals',
        lambda outputs: outputs['home_team__full_time_goals']
        + outputs['away_team__full_time_goals']
        > 4.5,
    ),
    (
        'under_1.5__full_time_goals',
        lambda outputs: outputs['home_team__full_time_goals']
        + outputs['away_team__full_time_goals']
        < 1.5,
    ),
    (
        'under_2.5__full_time_goals',
        lambda outputs: outputs['home_team__full_time_goals']
        + outputs['away_team__full_time_goals']
        < 2.5,
    ),
    (
        'under_3.5__full_time_goals',
        lambda outputs: outputs['home_team__full_time_goals']
        + outputs['away_team__full_time_goals']
        < 3.5,
    ),
    (
        'under_4.5__full_time_goals',
        lambda outputs: outputs['home_team__full_time_goals']
        + outputs['away_team__full_time_goals']
        < 4.5,
    ),
]
