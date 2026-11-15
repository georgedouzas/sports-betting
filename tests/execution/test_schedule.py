"""Test finding the moment a bet goes on for a match."""

import pandas as pd
import pytest

from sportsbet.core import ExecutionError
from sportsbet.execution import find_betting_moment

KICKOFF = pd.Timestamp('2026-11-15 15:00', tz='UTC')


class FakeDataLoader:
    """A dataloader that remembers the moment it was fitted for."""

    def __init__(self, status='preplay', minutes=0):
        """Keep the status and the time into the match."""
        self.target_event_status_ = status
        self.target_event_time_ = pd.Timedelta(minutes=minutes)


def test_find_betting_moment_default_lead_is_five_minutes():
    """Test a preplay bet with no lead time given goes on five minutes before a 15:00 kickoff."""
    assert find_betting_moment(FakeDataLoader(), KICKOFF) == pd.Timestamp('2026-11-15 14:55', tz='UTC')


def test_find_betting_moment_custom_lead():
    """Test the lead time sets how long before the kickoff the bet goes on."""
    moment = find_betting_moment(FakeDataLoader(), KICKOFF, pd.Timedelta('30min'))
    assert moment == pd.Timestamp('2026-11-15 14:30', tz='UTC')


def test_find_betting_moment_inplay_model_execution_error():
    """Test a model fitted for an in-play moment is refused, since a run supports preplay models only."""
    with pytest.raises(ExecutionError, match='In-play execution is not supported'):
        find_betting_moment(FakeDataLoader(status='inplay', minutes=60), KICKOFF)


def test_find_betting_moment_negative_lead_execution_error():
    """Test a negative lead time is refused."""
    with pytest.raises(ExecutionError, match='must not be negative'):
        find_betting_moment(FakeDataLoader(), KICKOFF, pd.Timedelta('-1min'))
