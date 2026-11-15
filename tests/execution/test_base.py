"""Test the venue contract, the reference API venue and the run record."""

import asyncio

import pandas as pd
import pandera.errors
import pytest

from sportsbet.core import CancellationUnsupportedError, ExecutionError
from sportsbet.execution import BaseVenue, PlacementIntent, PlacementStatus, build_steps_frame
from tests.execution.conftest import ReferenceVenue, identity_of

PRICE = 2.5
STAKE = 10.0
DRIFTED = 2.0
LIMIT = 5.0


class MinimalVenue(BaseVenue):
    """A venue that implements only what a run needs."""

    key = 'minimal'

    async def authenticate(self):
        """Accept the login."""

    async def read_price(self, identity):
        """Offer nothing."""

    async def place(self, intent):
        """Place nothing."""


def _intent(min_price=PRICE, stake=STAKE):
    """Return an intent to back Arsenal on the event."""
    return PlacementIntent(identity=identity_of(), stake=stake, min_price=min_price, value_bet='event|home_win')


def test_base_venue_without_read_price_type_error():
    """Test a venue that leaves out reading the price cannot be built, and Python names the missing method."""

    class Incomplete(BaseVenue):
        async def authenticate(self):
            """Accept the login."""

        async def place(self, intent):
            """Place nothing."""
            return

    with pytest.raises(TypeError, match='read_price'):
        Incomplete()


@pytest.mark.parametrize(
    ('operation', 'message'),
    [
        (lambda venue: venue.list_markets(['Arsenal vs Chelsea']), 'does not list its markets'),
        (lambda venue: venue.read_balance(), 'does not read its balance'),
        (lambda venue: venue.read_status([identity_of()]), 'does not read the status'),
    ],
)
def test_base_venue_unsupported_operation_execution_error(operation, message):
    """Test an optional operation a venue does not provide raises an execution error that says so."""
    with pytest.raises(ExecutionError, match=message):
        asyncio.run(operation(MinimalVenue()))


def test_base_venue_cancel_unsupported_error():
    """Test a venue that cannot cancel raises the cancellation-unsupported error."""
    with pytest.raises(CancellationUnsupportedError, match='cannot cancel'):
        asyncio.run(MinimalVenue().cancel(identity_of()))


def test_base_venue_close_does_nothing_by_default():
    """Test releasing a venue that holds nothing open is allowed and does nothing."""
    assert asyncio.run(MinimalVenue().close()) is None


def test_reference_venue_places_at_the_book_price():
    """Test the reference venue answers with the book's price and its own bet identifier."""
    venue = ReferenceVenue({('Arsenal vs Chelsea', 'home_win', 'Arsenal'): PRICE})
    receipt = asyncio.run(venue.place(_intent(min_price=PRICE - 0.1)))
    assert (receipt.status, receipt.price, receipt.venue_bet_id) == (PlacementStatus.MATCHED_FULL, PRICE, 'REF-1')


def test_reference_venue_drift_refuses_price():
    """Test a price that drifted below the minimum is refused, carries the venue's price, and stakes nothing."""
    venue = ReferenceVenue({('Arsenal vs Chelsea', 'home_win', 'Arsenal'): PRICE}, drift=DRIFTED)
    receipt = asyncio.run(venue.place(_intent(min_price=2.1)))
    assert (receipt.status, receipt.price) == (PlacementStatus.REFUSED_PRICE, DRIFTED)
    assert venue.placements == []


def test_reference_venue_limit_refuses_with_its_message():
    """Test a stake above the account's limit is refused with the venue's message."""
    venue = ReferenceVenue({('Arsenal vs Chelsea', 'home_win', 'Arsenal'): PRICE}, limit=LIMIT)
    receipt = asyncio.run(venue.place(_intent()))
    assert receipt.status is PlacementStatus.REFUSED_LIMIT
    assert 'limit' in receipt.detail


def test_reference_venue_without_confirmation_refuses_unconfirmed():
    """Test a bet the venue never confirms is reported as unconfirmed and tells the user to check the venue."""
    venue = ReferenceVenue({('Arsenal vs Chelsea', 'home_win', 'Arsenal'): PRICE}, confirms=False)
    receipt = asyncio.run(venue.place(_intent()))
    assert receipt.status is PlacementStatus.REFUSED_UNCONFIRMED
    assert 'Check the venue' in receipt.detail


def test_reference_venue_unknown_selection_has_no_price():
    """Test a selection the book does not hold has no price."""
    assert asyncio.run(ReferenceVenue().read_price(identity_of())) is None


def test_build_steps_frame_invalid_steps_fail_validation():
    """Test a step without a time fails the validation of the steps frame."""
    with pytest.raises(pandera.errors.SchemaError):
        build_steps_frame([(pd.NaT, 'start', 'Arsenal vs Chelsea')])


@pytest.mark.parametrize('status', ['pending', 'refused_killed', 'blocked'])
def test_placement_status_unknown_value_error(status):
    """Test a status no run or venue produces cannot be built."""
    with pytest.raises(ValueError, match=status):
        PlacementStatus(status)
