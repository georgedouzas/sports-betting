"""Find the moment a bet goes on for a match."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from sportsbet.dataloaders import BaseDataLoader


def find_betting_moment(dataloader: BaseDataLoader, kickoff: pd.Timestamp) -> pd.Timestamp:
    """Return when the bet goes on for a match.

    A live model bets at the kickoff plus the time into the match it was fitted for. Any other model bets at the
    kickoff.

    Args:
        dataloader:
            The dataloader the model was fitted on.
        kickoff:
            The kickoff of the match.

    Returns:
        moment:
            When the bet goes on.
    """
    if dataloader.target_event_status_ == 'inplay':
        return kickoff + dataloader.target_event_time_
    return kickoff
