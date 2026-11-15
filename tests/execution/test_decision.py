"""Test deciding the one bet a model backs on an event."""

import pandas as pd

from sportsbet.execution import decide_bet
from tests.execution.conftest import EVENT, StubBettor

MARKETS = ('home_win', 'draw', 'away_win', 'over_2.5')
PRICES = {'home_win': 1.8, 'draw': 3.5, 'away_win': 4.2, 'over_2.5': 1.9}
STAKE = 10.0


def _event(prices=PRICES):
    """Return the event's one-row features and odds."""
    X = pd.DataFrame({'home_team': ['Arsenal'], 'away_team': ['Chelsea']})
    O = pd.DataFrame({f'acme__{market}__preplay__0min': [price] for market, price in prices.items()})
    return X, O


def _decide(bettor, prices=PRICES):
    """Return the decision a bettor makes on the event."""
    X, O = _event(prices)
    return decide_bet(EVENT, bettor, X, O, venue_key='book', stake=STAKE)


def test_decide_bet_home_win_backs_the_home_team():
    """Test backing `home_win` backs the home team at the configured stake."""
    intent, _ = _decide(StubBettor(MARKETS, backed={'home_win'}))
    assert (intent.identity.market, intent.identity.selection, intent.stake) == ('home_win', 'Arsenal', STAKE)


def test_decide_bet_away_win_backs_the_away_team():
    """Test backing `away_win` backs the away team."""
    intent, _ = _decide(StubBettor(MARKETS, backed={'away_win'}))
    assert intent.identity.selection == 'Chelsea'


def test_decide_bet_draw_backs_draw():
    """Test backing `draw` backs `Draw`."""
    intent, _ = _decide(StubBettor(MARKETS, backed={'draw'}))
    assert intent.identity.selection == 'Draw'


def test_decide_bet_over_line_backs_the_line():
    """Test backing `over_2.5` backs `Over 2.5`."""
    intent, _ = _decide(StubBettor(MARKETS, backed={'over_2.5'}))
    assert intent.identity.selection == 'Over 2.5'


def test_decide_bet_two_markets_takes_the_first_in_market_order():
    """Test two backed markets give the first in the bettor's order, and the other is named as not placed."""
    intent, reason = _decide(StubBettor(MARKETS, backed={'away_win', 'over_2.5'}))
    assert intent.identity.market == 'away_win'
    assert 'Not placed: over_2.5' in reason


def test_decide_bet_minimum_price_is_the_source_price():
    """Test the lowest acceptable price is the source's price the model decided on."""
    intent, _ = _decide(StubBettor(MARKETS, backed={'away_win'}))
    assert intent.min_price == PRICES['away_win']


def test_decide_bet_no_source_price_places_nothing():
    """Test a backed market with no source price gives no bet and says no price was available."""
    intent, reason = _decide(StubBettor(MARKETS, backed={'draw'}), {**PRICES, 'draw': float('nan')})
    assert intent is None
    assert 'no price was available' in reason


def test_decide_bet_no_value_places_nothing():
    """Test a bettor that backs nothing gives no bet and says no value bet was found."""
    intent, reason = _decide(StubBettor(MARKETS, value=False))
    assert intent is None
    assert 'No value bet' in reason


def test_decide_bet_unknown_market_places_nothing():
    """Test a market a run does not know how to back gives no bet and says so."""
    intent, reason = _decide(StubBettor(('corners_over_9.5',)), {'corners_over_9.5': 2.0})
    assert intent is None
    assert 'not a market a run knows' in reason
