"""Common parameters definitions for modules."""

from typing import Annotated

import pandas as pd

DATE_COLS = ['date']
GROUPS_COLS = ['league', 'division', 'year']
TEAMS_COLS = ['home_team', 'away_team']
MATCH_COLS = GROUPS_COLS + TEAMS_COLS
IDENTITY_COLS = DATE_COLS + GROUPS_COLS + TEAMS_COLS
INPUT_EVENT_STATUSES = ['preplay']
TARGET_EVENT_STATUSES = ['inplay', 'postplay']
STATUSES = INPUT_EVENT_STATUSES + TARGET_EVENT_STATUSES
EVENT_COLS = ['event_status', 'event_time']
IDENTITY_FIELDS = {
    'date': Annotated[pd.DatetimeTZDtype, 'ns', 'utc'],
    'league': str,
    'division': int,
    'year': int,
    'home_team': str,
    'away_team': str,
}
ALIASES: dict[str, str] = {
    'Olimpia Milano': 'EA7 Emporio Armani Milan',
}
