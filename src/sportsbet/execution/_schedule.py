"""Find the moment a bet goes on for a match."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import pandas as pd

from ..core import ExecutionError
from ..dataloaders import BaseDataLoader

DEFAULT_LEAD = pd.Timedelta('5min')


def find_betting_moment(
    dataloader: BaseDataLoader,
    kickoff: pd.Timestamp,
    lead: pd.Timedelta = DEFAULT_LEAD,
) -> pd.Timestamp:
    """Return when the bet goes on for a match: its kickoff minus the lead time.

    A run supports preplay models only. Most bookmakers suspend a preplay market at kickoff, so the bet goes on a lead
    time before it.

    Args:
        dataloader:
            The dataloader the model was fitted on.
        kickoff:
            The kickoff of the match.
        lead:
            How long before the kickoff the bet goes on. It must not be negative.

    Returns:
        moment:
            When the bet goes on.

    Raises:
        ExecutionError: If the dataloader was fitted for an in-play moment, or the lead time is negative.

    Examples:
        >>> import pandas as pd
        >>> from sportsbet.dataloaders import build_dataloader
        >>> from sportsbet.execution import find_betting_moment
        >>> dataloader = build_dataloader(stats='football-data', leagues=['Italy'], years=[2024])
        >>> find_betting_moment(dataloader, pd.Timestamp('2026-11-15 15:00', tz='UTC'))
        Timestamp('2026-11-15 14:55:00+0000', tz='UTC')
    """
    if getattr(dataloader, 'target_event_status_', None) == 'inplay':
        msg = 'In-play execution is not supported. Fit the model for a preplay moment to run it.'
        raise ExecutionError(msg)
    if lead < pd.Timedelta(0):
        msg = f'The lead time must not be negative, and it is {lead}.'
        raise ExecutionError(msg)
    return kickoff - lead
