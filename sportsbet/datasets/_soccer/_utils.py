"""
The :mod:`sportsbed.datasets._utils` includes utilities
for soccer data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import pandas as pd

OUTPUTS = [
    (
        'output__home_win__full_time_goals',
        lambda data: data['target__home_team__full_time_goals']
        > data['target__away_team__full_time_goals'],
    ),
    (
        'output__away_win__full_time_goals',
        lambda data: data['target__home_team__full_time_goals']
        < data['target__away_team__full_time_goals'],
    ),
    (
        'output__draw__full_time_goals',
        lambda data: data['target__home_team__full_time_goals']
        == data['target__away_team__full_time_goals'],
    ),
    (
        'output__over_1.5__full_time_goals',
        lambda data: data['target__home_team__full_time_goals']
        + data['target__away_team__full_time_goals']
        > 1.5,
    ),
    (
        'output__over_2.5__full_time_goals',
        lambda data: data['target__home_team__full_time_goals']
        + data['target__away_team__full_time_goals']
        > 2.5,
    ),
    (
        'output__over_3.5__full_time_goals',
        lambda data: data['target__home_team__full_time_goals']
        + data['target__away_team__full_time_goals']
        > 3.5,
    ),
    (
        'output__over_4.5__full_time_goals',
        lambda data: data['target__home_team__full_time_goals']
        + data['target__away_team__full_time_goals']
        > 4.5,
    ),
    (
        'output__under_1.5__full_time_goals',
        lambda data: data['target__home_team__full_time_goals']
        + data['target__away_team__full_time_goals']
        < 1.5,
    ),
    (
        'output__under_2.5__full_time_goals',
        lambda data: data['target__home_team__full_time_goals']
        + data['target__away_team__full_time_goals']
        < 2.5,
    ),
    (
        'output__under_3.5__full_time_goals',
        lambda data: data['target__home_team__full_time_goals']
        + data['target__away_team__full_time_goals']
        < 3.5,
    ),
    (
        'output__under_4.5__full_time_goals',
        lambda data: data['target__home_team__full_time_goals']
        + data['target__away_team__full_time_goals']
        < 4.5,
    ),
]


def _read_csv(url):
    """Read csv file from URL as a pandas dataframe."""
    names = pd.read_csv(url, nrows=0, encoding='ISO-8859-1').columns
    return pd.read_csv(
        url, names=names, skiprows=1, encoding='ISO-8859-1', on_bad_lines='skip'
    )
