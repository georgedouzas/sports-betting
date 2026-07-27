"""Test finding the moment a bet goes on for a match."""

import pandas as pd

from sportsbet.execution import find_betting_moment


class FakeDataLoader:
    """A dataloader that remembers the moment it was fitted for."""

    def __init__(self, status='preplay', minutes=0):
        """Keep the status and the time into the match."""
        self.target_event_status_ = status
        self.target_event_time_ = pd.Timedelta(minutes=minutes)


def test_a_live_model_bets_at_the_kickoff_plus_the_time():
    """The moment of a live bet is the kickoff plus the minutes into the match."""
    loader = FakeDataLoader(status='inplay', minutes=60)
    assert find_betting_moment(loader, pd.Timestamp('2026-07-17 15:00', tz='UTC')) == pd.Timestamp(
        '2026-07-17 16:00',
        tz='UTC',
    )


def test_a_prematch_model_bets_at_the_kickoff():
    """The moment of a prematch bet is the kickoff."""
    loader = FakeDataLoader()
    assert find_betting_moment(loader, pd.Timestamp('2026-07-17 15:00', tz='UTC')) == pd.Timestamp(
        '2026-07-17 15:00',
        tz='UTC',
    )
