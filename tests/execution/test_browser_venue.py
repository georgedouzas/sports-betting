"""Test the browser venue against a fake bookmaker served over loopback."""

import pytest

from sportsbet.core import ExecutionError, VenueBlockedError
from sportsbet.execution import BetIdentity, PlacementIntent, PlacementStatus
from tests.execution.conftest import EVENT

HOME_PRICE = 2.5
STAKE = 10.0


def _intent(selection='Arsenal', market='home_win', min_price=HOME_PRICE, match=EVENT):
    """Return an intent to back a selection on the event."""
    identity = BetIdentity('loopback', match, market, selection)
    return PlacementIntent(identity=identity, stake=STAKE, min_price=min_price, value_bet=f'{match}|{market}')


@pytest.fixture
def opened(book, run):
    """Return the browser venue, closing its browser after the test."""
    yield book
    run(book.close())


def test_served_site_event_page_is_reachable(opened, served_site, run):
    """Test the served bookmaker lists the event and its page shows the selections."""
    shot = run(opened.session.navigate(f'{served_site.url}/event/arsenal-chelsea'))
    assert all(selection in shot.yaml for selection in ('Arsenal', 'Draw', 'Chelsea'))


def test_authenticate_logged_in_profile_passes(opened, run):
    """Test a profile that holds a login passes the logged-in check."""
    run(opened.authenticate())


def test_authenticate_without_login_names_the_profile(opened, served_site, run):
    """Test a profile with no login fails with a message saying how to log in once with a visible browser."""
    served_site.state['logged_in'] = False
    with pytest.raises(ExecutionError, match='headless=False'):
        run(opened.authenticate())


def test_authenticate_blocked_site_venue_blocked_error(opened, served_site, run):
    """Test a site that answers with a blocking status raises the venue-blocked error."""
    served_site.state['status'] = 403
    with pytest.raises(VenueBlockedError):
        run(opened.authenticate())


def test_place_unknown_event_rejected_before_stake(opened, run):
    """Test an event the site does not list gives a rejected receipt, and no stake is typed."""
    receipt = run(opened.place(_intent(match='Liverpool vs Everton')))
    assert receipt.status is PlacementStatus.REJECTED
    assert 'not found' in receipt.detail
    assert 'enter_stake' not in opened.taken


def test_place_mismatched_slip_rejected_before_stake(opened, served_site, run):
    """Test a slip that holds another selection gives a rejected receipt, and no stake is typed."""
    served_site.state['slip_label'] = 'Chelsea'
    receipt = run(opened.place(_intent()))
    assert receipt.status is PlacementStatus.REJECTED
    assert 'did not match' in receipt.detail
    assert 'enter_stake' not in opened.taken


def test_place_price_below_minimum_refused_price(opened, run):
    """Test a slip price below the minimum is refused with the slip's price, and no stake is typed."""
    receipt = run(opened.place(_intent(min_price=HOME_PRICE + 0.4)))
    assert (receipt.status, receipt.price) == (PlacementStatus.REFUSED_PRICE, HOME_PRICE)
    assert 'enter_stake' not in opened.taken


def test_place_confirmation_carries_the_site_bet_id_and_price(opened, run):
    """Test a confirmed bet carries the site's own bet identifier and price."""
    receipt = run(opened.place(_intent()))
    assert (receipt.status, receipt.venue_bet_id, receipt.price) == (PlacementStatus.ACCEPTED, 'B-1042', HOME_PRICE)
    assert receipt.stake == STAKE
    assert opened.taken == ['add_selection', 'enter_stake', 'confirm_bet']


def test_place_refusal_carries_the_site_message(opened, served_site, run):
    """Test a bet the site refuses over the limit carries the site's message and no stake."""
    served_site.state['confirmation'] = 'refuse'
    receipt = run(opened.place(_intent()))
    assert receipt.status is PlacementStatus.REFUSED_LIMIT
    assert 'limit' in receipt.detail
    assert receipt.stake == 0.0


def test_place_missing_confirmation_refused_unconfirmed(opened, served_site, run):
    """Test a confirm the site never answers is unconfirmed and tells the user to check the venue."""
    served_site.state['confirmation'] = 'never'
    receipt = run(opened.place(_intent()))
    assert receipt.status is PlacementStatus.REFUSED_UNCONFIRMED
    assert 'Check the venue' in receipt.detail


def test_read_price_offered_selection_without_stake(opened, run):
    """Test reading the price returns the slip's price and enters no stake."""
    assert run(opened.read_price(_intent().identity)) == HOME_PRICE
    assert 'enter_stake' not in opened.taken


def test_read_price_selection_not_offered_is_none(opened, run):
    """Test a selection the event page does not offer has no price."""
    assert run(opened.read_price(_intent(selection='Over 2.5', market='over_2.5').identity)) is None
