"""Includes utilities for soccer data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import pandas as pd

OVER_UNDER = [1.5, 2.5, 3.5, 4.5]
OUTPUTS = [
    (
        'output__home_win__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] > data['target__away_team__full_time_goals'],
    ),
    (
        'output__away_win__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] < data['target__away_team__full_time_goals'],
    ),
    (
        'output__draw__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] == data['target__away_team__full_time_goals'],
    ),
    (
        f'output__over_{OVER_UNDER[0]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        > OVER_UNDER[0],
    ),
    (
        f'output__over_{OVER_UNDER[1]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        > OVER_UNDER[1],
    ),
    (
        f'output__over_{OVER_UNDER[2]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        > OVER_UNDER[2],
    ),
    (
        f'output__over_{OVER_UNDER[3]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        > OVER_UNDER[3],
    ),
    (
        f'output__under_{OVER_UNDER[0]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        < OVER_UNDER[0],
    ),
    (
        f'output__under_{OVER_UNDER[1]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        < OVER_UNDER[1],
    ),
    (
        f'output__under_{OVER_UNDER[2]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        < OVER_UNDER[2],
    ),
    (
        f'output__under_{OVER_UNDER[3]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        < OVER_UNDER[3],
    ),
]


def _read_csv(url: str) -> pd.DataFrame:
    """Read csv file from URL as a pandas dataframe."""
    names = pd.read_csv(url, nrows=0, encoding='ISO-8859-1').columns.to_list()
    return pd.read_csv(url, names=names, skiprows=1, encoding='ISO-8859-1', on_bad_lines='skip')
