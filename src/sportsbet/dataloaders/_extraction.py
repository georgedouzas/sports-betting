"""Turn extraction arguments into keyword arguments for the training data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Any

import pandas as pd


def build_extraction_settings(
    odds_type: str | None = None,
    drop_na_thres: float | None = None,
    target_event_status: str | None = None,
    target_event_time: str | None = None,
    input_event_status: str | None = None,
    input_event_time: str | None = None,
) -> dict[str, Any]:
    """Turn the extraction arguments into keyword arguments for `extract_train_data`.

    Args:
        odds_type:
            The odds to extract, or `None` to leave it to the method.
        drop_na_thres:
            The threshold to drop missing columns, or `None` to leave it to the method.
        target_event_status:
            Where the targets are taken from, or `None` to leave it to the method.
        target_event_time:
            The moment of the targets when they are in-play, as `45min`, or `None`.
        input_event_status:
            The latest snapshot kept as a feature, or `None` to keep every one before the target.
        input_event_time:
            The moment of the input horizon, as `45min`, or `None`.

    Returns:
        settings:
            The keyword arguments, with the times as `Timedelta` and the `None` values dropped.

    Examples:
        >>> from sportsbet.dataloaders import build_extraction_settings
        >>> build_extraction_settings(odds_type='market_average', target_event_time='45min')
        {'odds_type': 'market_average', 'target_event_time': Timedelta('0 days 00:45:00')}
    """
    settings: dict[str, Any] = {
        'odds_type': odds_type,
        'drop_na_thres': drop_na_thres,
        'target_event_status': target_event_status,
        'input_event_status': input_event_status,
        'target_event_time': target_event_time,
        'input_event_time': input_event_time,
    }
    for name in ('target_event_time', 'input_event_time'):
        if settings.get(name) is not None:
            settings[name] = pd.Timedelta(settings[name])
    return {name: value for name, value in settings.items() if value is not None}
