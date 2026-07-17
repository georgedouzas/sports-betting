"""Test placing the value bets of the upcoming matches one at a time."""

import asyncio
import logging

import numpy as np
import pandas as pd
import pytest

from sportsbet.evaluation import OddsComparisonBettor
from sportsbet.execution import betting_moment, execute, feasible

from .conftest import FakeVenue

NOW = pd.Timestamp('2026-07-17 12:00', tz='UTC')
STAKE = 10.0
BOTH = 2


class KeenBettor(OddsComparisonBettor):
    """A bettor that backs the home win of every match, so the schedule always has bets."""

    def bet(self, X, O):
        """Back the one market on every row."""
        return np.ones((len(X), len(self.betting_markets_)), dtype=bool)


class FakeDataLoader:
    """A dataloader that returns arranged fixtures and remembers the moment it was fitted for."""

    def __init__(self, fixtures, status='preplay', minutes=0):
        """Keep the fixtures and the moment."""
        self.fixtures = fixtures
        self.target_event_status_ = status
        self.target_event_time_ = pd.Timedelta(minutes=minutes)

    def extract_fixtures_data(self):
        """Return the arranged fixtures with home win odds."""
        X = self.fixtures
        O = pd.DataFrame({'home_win__odds': [2.5] * len(X)}, index=X.index)
        return X, None, O


def _fixtures(kickoffs, teams):
    """Return upcoming matches indexed by kickoff."""
    index = pd.DatetimeIndex([pd.Timestamp(k, tz='UTC') for k in kickoffs], name='date')
    return pd.DataFrame({'home_team': [t[0] for t in teams], 'away_team': [t[1] for t in teams]}, index=index)


def _bettor():
    """Return a fitted bettor that backs the home win."""
    bettor = KeenBettor(betting_markets=['home_win'])
    bettor.betting_markets_ = ['home_win']
    return bettor


def run(coroutine):
    """Run a coroutine."""
    return asyncio.run(coroutine)


def _clock():
    """Return a clock stuck at NOW, so a test never waits."""
    return NOW


async def _no_wait(_seconds):
    """Wait for nothing, so a test never sleeps."""
    return


def test_a_live_model_bets_at_the_kickoff_plus_the_time():
    """The moment of a live bet is the kickoff plus the minutes into the match."""
    loader = FakeDataLoader(_fixtures(['2026-07-17 15:00'], [('A', 'B')]), status='inplay', minutes=60)
    assert betting_moment(loader, pd.Timestamp('2026-07-17 15:00', tz='UTC')) == pd.Timestamp(
        '2026-07-17 16:00', tz='UTC',
    )


def test_a_prematch_model_bets_at_the_kickoff():
    """The moment of a prematch bet is the kickoff."""
    loader = FakeDataLoader(_fixtures(['2026-07-17 15:00'], [('A', 'B')]))
    assert betting_moment(loader, pd.Timestamp('2026-07-17 15:00', tz='UTC')) == pd.Timestamp(
        '2026-07-17 15:00', tz='UTC',
    )


def test_a_match_whose_moment_has_passed_is_not_feasible():
    """A live match already past its moment cannot be bet on."""
    fixtures = _fixtures(['2026-07-17 09:00', '2026-07-17 14:00'], [('Past', 'X'), ('Soon', 'Y')])
    loader = FakeDataLoader(fixtures, status='inplay', minutes=60)
    mask = feasible(loader, fixtures, NOW)
    assert list(mask) == [False, True]


def test_a_window_keeps_only_the_matches_inside_it():
    """A window drops the matches whose moment falls beyond it."""
    fixtures = _fixtures(['2026-07-17 12:30', '2026-07-17 20:00'], [('Inside', 'X'), ('Beyond', 'Y')])
    loader = FakeDataLoader(fixtures)
    mask = feasible(loader, fixtures, NOW, window=pd.Timedelta(hours=2))
    assert list(mask) == [True, False]


def test_only_the_feasible_value_bets_are_placed():
    """The passed one is dropped and the reachable one goes on."""
    fixtures = _fixtures(['2026-07-17 09:00', '2026-07-17 14:00'], [('Past', 'X'), ('Soon', 'Y')])
    loader = FakeDataLoader(fixtures, status='inplay', minutes=60)
    venue = FakeVenue(prices={('Soon vs Y', 'home_win', 'Soon'): 2.5})
    receipts = run(
        execute(venue, loader, _bettor(), stake=STAKE, confirm_total=STAKE, clock=_clock, wait=_no_wait),
    )
    assert list(receipts['match']) == ['Soon vs Y']
    assert receipts['status'].iloc[0] == 'matched_full'


def test_nothing_is_staked_without_the_confirmed_total():
    """The default stakes nothing and states the total to pass back."""
    fixtures = _fixtures(['2026-07-17 14:00'], [('Soon', 'Y')])
    loader = FakeDataLoader(fixtures)
    venue = FakeVenue(prices={('Soon vs Y', 'home_win', 'Soon'): 2.5})
    receipts = run(execute(venue, loader, _bettor(), stake=STAKE, clock=_clock, wait=_no_wait))
    assert (receipts['status'] == 'dry_run').all()
    assert receipts['stake'].sum() == 0.0
    assert venue.orders == {}
    assert str(STAKE) in receipts['detail'].iloc[0]


def test_a_wrong_total_stakes_nothing_and_states_the_real_one():
    """A total that does not match the quote refuses and names the real total."""
    fixtures = _fixtures(['2026-07-17 14:00'], [('Soon', 'Y')])
    loader = FakeDataLoader(fixtures)
    venue = FakeVenue(prices={('Soon vs Y', 'home_win', 'Soon'): 2.5})
    receipts = run(execute(venue, loader, _bettor(), stake=STAKE, confirm_total=999.0, clock=_clock, wait=_no_wait))
    assert (receipts['status'] == 'refused_unconfirmed').all()
    assert venue.orders == {}
    assert str(STAKE) in receipts['detail'].iloc[0]


def test_the_matches_go_on_one_at_a_time():
    """The venue is called once per match, in the schedule's order."""
    fixtures = _fixtures(['2026-07-17 13:00', '2026-07-17 14:00'], [('First', 'X'), ('Second', 'Y')])
    loader = FakeDataLoader(fixtures)
    venue = FakeVenue(
        prices={('First vs X', 'home_win', 'First'): 2.5, ('Second vs Y', 'home_win', 'Second'): 2.5},
    )
    receipts = run(execute(venue, loader, _bettor(), stake=STAKE, confirm_total=2 * STAKE, clock=_clock, wait=_no_wait))
    assert len(venue.placed_order) == BOTH
    assert venue.placed_order == list(receipts['ref'])
    assert set(receipts['match']) == {'First vs X', 'Second vs Y'}


def test_watching_one_at_a_time_can_miss_a_simultaneous_match():
    """Two live matches share a moment, so placing one makes the run late for the other and the window closes.

    This is the cost of watching one match at a time. Time passes as the run waits and as it places, so the clock
    advances by both.
    """
    fixtures = _fixtures(['2026-07-17 12:00', '2026-07-17 12:00'], [('First', 'X'), ('Second', 'Y')])
    loader = FakeDataLoader(fixtures, status='inplay', minutes=60)
    current = {'t': NOW}

    class SlowVenue(FakeVenue):
        async def place(self, intent):
            current['t'] += pd.Timedelta(minutes=5)
            return await super().place(intent)

    venue = SlowVenue(
        prices={('First vs X', 'home_win', 'First'): 2.5, ('Second vs Y', 'home_win', 'Second'): 2.5},
    )

    def clock():
        return current['t']

    async def wait(seconds):
        current['t'] += pd.Timedelta(seconds=seconds)

    run(
        execute(
            venue,
            loader,
            _bettor(),
            stake=STAKE,
            confirm_total=2 * STAKE,
            window=pd.Timedelta(hours=1),
            clock=clock,
            wait=wait,
        ),
    )
    assert len(venue.placed_order) == 1


def test_the_run_logs_what_it_places(caplog):
    """Each selection and placement is logged for the terminal to show."""
    fixtures = _fixtures(['2026-07-17 14:00'], [('Soon', 'Y')])
    loader = FakeDataLoader(fixtures)
    venue = FakeVenue(prices={('Soon vs Y', 'home_win', 'Soon'): 2.5})
    with caplog.at_level(logging.INFO, logger='sportsbet.execution'):
        run(execute(venue, loader, _bettor(), stake=STAKE, confirm_total=STAKE, clock=_clock, wait=_no_wait))
    messages = ' '.join(record.message for record in caplog.records)
    assert 'feasible value bets' in messages
    assert 'Placing Soon vs Y' in messages


def test_no_upcoming_matches_places_nothing():
    """An empty fixture set stakes nothing."""
    loader = FakeDataLoader(_fixtures([], []))
    venue = FakeVenue()
    receipts = run(execute(venue, loader, _bettor(), stake=STAKE, confirm_total=STAKE, clock=_clock, wait=_no_wait))
    assert receipts.empty


def test_a_future_live_match_waits_for_its_moment():
    """A match whose moment is ahead waits, and the wait is the seconds to that moment."""
    fixtures = _fixtures(['2026-07-17 12:30'], [('Later', 'Y')])
    loader = FakeDataLoader(fixtures, status='inplay', minutes=60)
    venue = FakeVenue(prices={('Later vs Y', 'home_win', 'Later'): 2.5})
    waited = []

    async def record_wait(seconds):
        waited.append(seconds)

    run(execute(venue, loader, _bettor(), stake=STAKE, confirm_total=STAKE, clock=_clock, wait=record_wait))
    assert waited == [pytest.approx(90 * 60)]
