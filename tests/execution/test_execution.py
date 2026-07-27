"""Test bet identities, credentials and venue behaviour."""

import asyncio

import pytest

from sportsbet.execution import (
    BetIdentity,
    CancellationUnsupportedError,
    CredentialError,
    CredentialRef,
    PlacementIntent,
    resolve,
)
from tests.conftest import FakeVenue

MATCH = 'Arsenal vs Chelsea'
OTHER = 'Liverpool vs Everton'
ARSENAL_PRICE = 2.10
PRICES = {
    (MATCH, 'home_win', 'Arsenal'): ARSENAL_PRICE,
    (OTHER, 'home_win', 'Liverpool'): 1.80,
}
REF_CHARS = 32
OPEN_EXPOSURE = 40.0
BALANCE = 250.0


def identity(match=MATCH, market='home_win', selection='Arsenal', venue='fake'):
    """Return an identity."""
    return BetIdentity(venue, match, market, selection)


def run(coroutine):
    """Run a coroutine."""
    return asyncio.run(coroutine)


def test_ref_is_thirty_two_hex_characters():
    """The reference fits the field a venue carries it in."""
    ref = identity().ref_
    assert len(ref) == REF_CHARS
    assert all(character in '0123456789abcdef' for character in ref)


def test_ref_is_derived_rather_than_remembered():
    """The same four fields give the same reference, so a fresh run recomputes it."""
    assert identity().ref_ == identity().ref_
    assert BetIdentity('fake', MATCH, 'home_win', 'Arsenal').ref_ == identity().ref_


@pytest.mark.parametrize(
    ('field', 'value'),
    [('venue', 'other'), ('match', OTHER), ('market', 'draw'), ('selection', 'Chelsea')],
)
def test_ref_changes_with_every_field(field, value):
    """Each of the four fields is part of what makes a bet that bet."""
    assert identity(**{field: value}).ref_ != identity().ref_


def test_same_four_fields_are_the_same_bet():
    """Two intents from different runs that share the four are one bet."""
    first = PlacementIntent(identity=identity(), stake=10.0, min_price=2.0, value_bet='run-1')
    second = PlacementIntent(identity=identity(), stake=25.0, min_price=1.5, value_bet='run-2')
    assert first.identity.ref_ == second.identity.ref_


def test_credential_is_read_from_the_named_variable(monkeypatch):
    """A credential is named, and the name is what travels."""
    monkeypatch.setenv('SPORTSBET_TEST_SECRET', 'sentinel-value')
    ref = CredentialRef('SPORTSBET_TEST_SECRET')
    assert str(ref) == 'SPORTSBET_TEST_SECRET'
    assert 'sentinel-value' not in repr(ref)


def test_missing_credential_names_the_variable(monkeypatch):
    """A missing credential says which variable was wanted."""
    monkeypatch.delenv('SPORTSBET_TEST_SECRET', raising=False)
    with pytest.raises(CredentialError, match='SPORTSBET_TEST_SECRET'):
        resolve(CredentialRef('SPORTSBET_TEST_SECRET'))


def test_the_secret_stays_out_of_what_is_shown(monkeypatch):
    """A secret is read where it is used and shown nowhere."""
    monkeypatch.setenv('SPORTSBET_TEST_SECRET', 'sentinel-value')
    ref = CredentialRef('SPORTSBET_TEST_SECRET')
    assert resolve(ref) == 'sentinel-value'
    assert 'sentinel-value' not in repr(ref)
    assert 'sentinel-value' not in str(ref)


def test_the_same_selection_at_two_venues_are_two_bets():
    """One selection at two venues is two bets, since the venue is part of what makes a bet that bet."""
    assert identity(venue='fake').ref_ != identity(venue='other').ref_


def test_the_venue_reports_nothing_for_a_bet_that_never_went_on():
    """A bet that was never placed is not at the venue, and the venue says so."""
    venue = FakeVenue(prices=PRICES)
    assert run(venue.read_status([identity()])).empty


def test_the_balance_and_the_exposure_are_both_reported():
    """The balance is what can be staked and the exposure is what already is."""
    venue = FakeVenue(prices=PRICES, balance=BALANCE, exposure=OPEN_EXPOSURE)
    balance, exposure = run(venue.read_balance())
    assert balance == BALANCE
    assert exposure == OPEN_EXPOSURE


def test_a_venue_that_cannot_cancel_says_so_rather_than_seeming_to():
    """A venue that cannot cancel raises rather than returning a receipt that implies it did."""
    venue = FakeVenue(prices=PRICES, cancels=False)
    with pytest.raises(CancellationUnsupportedError, match='cannot cancel'):
        run(venue.cancel(identity()))
