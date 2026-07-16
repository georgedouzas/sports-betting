"""Test the venue that has an official betting API.

Nothing here reaches Betfair. There is no sandbox to reach: the delayed application key runs against the live exchange
and places real bets, so the only safe proof is a recorded answer.
"""

import asyncio

import pytest

from sportsbet.execution import BetfairVenue, BetIdentity, PlacementIntent, PlacementStatus

MATCH = 'Arsenal v Chelsea'
SESSION = 'session-token'
ENV_NAMES = ('BETFAIR_APP_KEY', 'BETFAIR_USERNAME', 'BETFAIR_PASSWORD')
REF_CHARS = 32
ARSENAL_ID = 47972
ARSENAL_PRICE = 2.10
CHELSEA_PRICE = 3.40
LIMIT_PRICE = 2.05
STAKE = 10.0
BALANCE = 500.0
EXPOSURE = 25.0
CATALOGUE = [
    {
        'marketId': '1.234',
        'marketName': 'Match Odds',
        'event': {'name': MATCH},
        'runners': [
            {'selectionId': 47972, 'runnerName': 'Arsenal'},
            {'selectionId': 47973, 'runnerName': 'Chelsea'},
        ],
    },
]
BOOK = [
    {
        'marketId': '1.234',
        'runners': [
            {'selectionId': 47972, 'ex': {'availableToBack': [{'price': 2.10, 'size': 500.0}]}},
            {'selectionId': 47973, 'ex': {'availableToBack': [{'price': 3.40, 'size': 500.0}]}},
        ],
    },
]
PLACED = {
    'status': 'SUCCESS',
    'marketId': '1.234',
    'instructionReports': [
        {
            'status': 'SUCCESS',
            'betId': '298537625817',
            'orderStatus': 'EXECUTION_COMPLETE',
            'sizeMatched': 10.0,
            'averagePriceMatched': 2.10,
        },
    ],
}


class Recorder:
    """A stand-in for the API that records what it was asked and answers what it was told to."""

    def __init__(self, orders=None):
        """Keep the orders the venue is meant to already hold."""
        self.calls = []
        self.orders = orders or []

    async def __call__(self, url, method, params):
        """Record the call and answer it."""
        self.calls.append({'url': url, 'method': method, 'params': params})
        if method.endswith('listCurrentOrders'):
            return {'currentOrders': self.orders}
        if method.endswith('listMarketCatalogue'):
            return CATALOGUE
        if method.endswith('listMarketBook'):
            return BOOK
        if method.endswith('placeOrders'):
            return PLACED
        if method.endswith('getAccountFunds'):
            return {'availableToBetBalance': 500.0, 'exposure': -25.0}
        if method.endswith('cancelOrders'):
            return {'status': 'SUCCESS'}
        raise AssertionError(method)

    def of(self, name):
        """Return the calls made to a method."""
        return [call for call in self.calls if call['method'].endswith(name)]


@pytest.fixture
def venue(monkeypatch):
    """Return a venue whose API is a recording."""
    built = BetfairVenue(min_interval=0.0)
    built.token_ = SESSION
    recorder = Recorder()
    monkeypatch.setattr(built, 'call', recorder)
    built.recorder = recorder
    return built


def identity(selection='Arsenal'):
    """Return an identity."""
    return BetIdentity('betfair', MATCH, 'Match Odds', selection)


def intent(stake=10.0, min_price=2.0, selection='Arsenal'):
    """Return an intent."""
    return PlacementIntent(identity=identity(selection), stake=stake, min_price=min_price, value_bet='row-1')


def run(coroutine):
    """Run a coroutine."""
    return asyncio.run(coroutine)


def test_the_reference_goes_on_as_both_fields(venue):
    """The reference is sent as both, since neither field alone places once and only once.

    `customerRef` de-duplicates a resubmission for sixty seconds and never comes back. `customerOrderRef` persists and
    can be filtered on, and the venue polices nothing about it.
    """
    placed = intent()
    run(venue.place(placed))
    params = venue.recorder.of('placeOrders')[0]['params']
    assert params['customerRef'] == placed.identity.ref
    assert params['instructions'][0]['customerOrderRef'] == placed.identity.ref


def test_one_instruction_per_request(venue):
    """A request carries one bet, which is what makes the two references take the same value."""
    run(venue.place(intent()))
    params = venue.recorder.of('placeOrders')[0]['params']
    assert len(params['instructions']) == 1


def test_the_reference_fits_the_field(venue):
    """The reference fits Betfair's limit of thirty two characters and its charset."""
    ref = identity().ref
    assert len(ref) <= REF_CHARS
    assert all(character in '0123456789abcdef' for character in ref)


def test_an_already_placed_bet_is_read_back_by_its_order_reference(venue):
    """Recovery filters on the reference that persists, which is the only one that can be."""
    run(venue.place(intent()))
    listed = venue.recorder.of('listCurrentOrders')
    assert listed
    assert listed[0]['params'] == {'customerOrderRefs': [identity().ref]}


def test_a_bet_the_venue_already_holds_is_not_placed_again(monkeypatch):
    """A venue that already holds the bet is told nothing further."""
    built = BetfairVenue(min_interval=0.0)
    built.token_ = SESSION
    recorder = Recorder(
        orders=[
            {
                'betId': '298537625817',
                'marketId': '1.234',
                'customerOrderRef': identity().ref,
                'sizeMatched': 10.0,
                'averagePriceMatched': 2.10,
            },
        ],
    )
    monkeypatch.setattr(built, 'call', recorder)
    receipt = run(built.place(intent()))
    assert receipt.status is PlacementStatus.ALREADY_PLACED
    assert receipt.venue_bet_id == '298537625817'
    assert recorder.of('placeOrders') == []


def test_nothing_reads_the_request_reference_back(venue):
    """`customerRef` is treated as unreadable, since the docs and the interface definition disagree on whether it is."""
    receipt = run(venue.place(intent()))
    assert receipt.venue_bet_id == '298537625817'
    for call in venue.recorder.of('listCurrentOrders'):
        assert 'customerRefs' not in call['params']


def test_the_market_is_resolved_to_what_the_venue_calls_it(venue):
    """A model names a market and a venue names it something else, so the venue is asked."""
    run(venue.place(intent()))
    params = venue.recorder.of('placeOrders')[0]['params']
    assert params['marketId'] == '1.234'
    assert params['instructions'][0]['selectionId'] == ARSENAL_ID


def test_a_selection_the_venue_does_not_offer_is_rejected(venue):
    """A bet the venue has no market for does not go on."""
    receipt = run(venue.place(intent(selection='Tottenham')))
    assert receipt.status is PlacementStatus.REJECTED
    assert venue.recorder.of('placeOrders') == []


def test_the_minimum_price_is_the_limit_price(venue):
    """The bet goes on at the worst price the caller allowed, and no worse."""
    run(venue.place(intent(min_price=LIMIT_PRICE)))
    params = venue.recorder.of('placeOrders')[0]['params']
    assert params['instructions'][0]['limitOrder']['price'] == LIMIT_PRICE
    assert params['instructions'][0]['limitOrder']['size'] == STAKE


def test_the_markets_carry_their_prices(venue):
    """The markets on offer come back with what they are paying."""
    markets = run(venue.list_markets([MATCH]))
    assert list(markets['selection']) == ['Arsenal', 'Chelsea']
    assert list(markets['price']) == [ARSENAL_PRICE, CHELSEA_PRICE]


def test_the_balance_reports_the_exposure_as_a_positive_amount(venue):
    """The exposure is an amount at stake, and the venue reports it as a negative number."""
    balance, exposure = run(venue.read_balance())
    assert balance == BALANCE
    assert exposure == EXPOSURE


def test_a_venue_that_is_not_authenticated_says_so():
    """A call before a login says what is missing rather than failing obscurely."""
    with pytest.raises(Exception, match='not authenticated'):
        run(BetfairVenue().read_balance())


def test_the_defaults_are_variable_names_rather_than_secrets():
    """Nothing sensitive has a default, since the defaults are names."""
    built = BetfairVenue()
    assert (built.app_key_env, built.username_env, built.password_env) == ENV_NAMES
