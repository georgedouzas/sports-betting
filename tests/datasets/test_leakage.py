"""Tests that a bettor is never handed something it could not have had.

A bet is placed at the moment its odds are quoted. Anything a model is shown from a later moment is not an edge, it is
knowledge of the future, and it shows up as a backtest that makes money no bookmaker would allow.
"""

import pandas as pd
import pytest

from tests.conftest import SnapshotsDataLoader

IDENTITY = {'league': 'England', 'division': 1, 'year': 2025, 'home_team': 'A', 'away_team': 'B'}
LATER = {'league': 'England', 'division': 1, 'year': 2025, 'home_team': 'C', 'away_team': 'D'}
HALF_TIME = pd.Timedelta('45min')
KICK_OFF = pd.Timedelta('0min')


def _stats():
    """The statistics of two matches, each of them seen before, during and after."""
    rows = []
    for date, identity, goals in (('2024-08-16', IDENTITY, 2), ('2024-08-23', LATER, 1)):
        rows += [
            {'date': date, **identity, 'event_status': 'preplay', 'event_time': KICK_OFF, 'home_form': 1.0},
            {'date': date, **identity, 'event_status': 'inplay', 'event_time': HALF_TIME, 'home_goals': goals},
            {
                'date': date,
                **identity,
                'event_status': 'postplay',
                'event_time': KICK_OFF,
                'home_goals': goals + 1,
                'home_win': 1,
            },
        ]
    return pd.DataFrame(rows)


def _odds(moments):
    """The odds of the two matches, quoted at each of the given moments."""
    rows = [
        {'date': date, **identity, 'event_status': status, 'event_time': time, 'provider': 'book', 'home_win': 2.0}
        for date, identity in (('2024-08-16', IDENTITY), ('2024-08-23', LATER))
        for status, time in moments
    ]
    return pd.DataFrame(rows)


def test_a_feature_is_never_later_than_the_odds_it_is_bet_against():
    """Test the half-time score is not a feature of a bet placed before kick-off.

    The odds a free soccer feed carries are the ones quoted before the match, and the statistics run all the way to the
    whistle. Left alone, a model was handed the score at half time and asked to bet at the price offered before kick-
    off, which is not a strategy but a way of travelling in time. It backtested at a yield no bookmaker would survive.
    """
    dataloader = SnapshotsDataLoader(_stats(), _odds([('preplay', KICK_OFF)]))
    X, _, _ = dataloader.extract_train_data(odds_type='book')
    assert not [col for col in X.columns if 'inplay' in col]


def test_a_feature_from_the_moment_of_the_odds_is_kept():
    """Test in-play betting still works, when the odds are quoted in play.

    The rule is not that the half-time score is forbidden. It is that a bet cannot use what was not known when its price
    was quoted. Buy a price at half time and the half-time score is yours to use.
    """
    dataloader = SnapshotsDataLoader(_stats(), _odds([('preplay', KICK_OFF), ('inplay', HALF_TIME)]))
    X, _, _ = dataloader.extract_train_data(odds_type='book')
    assert [col for col in X.columns if 'inplay' in col]


def test_asking_for_a_feature_later_than_the_odds_is_refused():
    """Test asking for it on purpose is refused, and told why."""
    dataloader = SnapshotsDataLoader(_stats(), _odds([('preplay', KICK_OFF)]))
    with pytest.raises(ValueError, match='could not have had'):
        dataloader.extract_train_data(
            odds_type='book',
            input_event_status='inplay',
            input_event_time=HALF_TIME,
        )


def test_a_match_with_no_features_is_not_dropped():
    """Test the first match of a season survives, though nothing has happened yet to describe it.

    It has two teams, a date and a price, and it is perfectly bettable. A pivot drops a row it has nothing to put in, so
    the match used to vanish with nothing said, and only the in-play columns had been keeping it alive.
    """
    stats = _stats()
    formless = stats['event_status'].eq('preplay') & stats['date'].eq('2024-08-16')
    stats.loc[formless, 'home_form'] = None

    dataloader = SnapshotsDataLoader(stats, _odds([('preplay', KICK_OFF)]))
    X, Y, O = dataloader.extract_train_data(odds_type='book')

    matches = 2
    assert len(X) == matches
    assert len(Y) == matches
    assert len(O) == matches


def test_without_odds_there_is_nothing_to_predict_but_the_features_remain():
    """Test a dataloader with no odds gives the features, and refuses to invent a target.

    The markets a model learns are the ones its odds price, so with no odds there is no market and no target. The
    features are still there, and exploring them, or learning from them without a target of ours, is still worth doing.
    """
    dataloader = SnapshotsDataLoader(_stats())

    with pytest.raises(ValueError, match='no markets to predict'):
        dataloader.extract_train_data()

    X, Y, O = dataloader.extract_train_data(learning_type='unsupervised')
    matches = 2
    assert len(X) == matches
    assert Y is None
    assert O.empty
