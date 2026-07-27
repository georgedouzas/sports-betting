"""Convert event times to and from the whole-minute tokens used in column names."""

import pandas as pd


def format_event_time(event_time: pd.Timedelta) -> str:
    """Render an event time as the whole-minute token used in column names.

    Args:
        event_time: A time delta, e.g. `pd.Timedelta('60min')`.

    Returns:
        The token, e.g. `60min`.
    """
    total_minutes = int(event_time.total_seconds() / 60)
    return f'{total_minutes}min'


def parse_event_time(token: str) -> pd.Timedelta:
    """Read the whole-minute token used in column names back into a time delta.

    Args:
        token: A whole-minute token, e.g. `60min`.

    Returns:
        The time delta the token names.
    """
    return pd.Timedelta(minutes=int(token[: -len('min')]))
