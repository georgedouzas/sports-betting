"""Test the placing of bets."""

import asyncio

import pytest

from sportsbet.dataloaders import DataLoader
from sportsbet.evaluation import OddsComparisonBettor
from sportsbet.execution import (
    BetIdentity,
    CancellationUnsupportedError,
    CredentialError,
    CredentialRef,
    ExposureLimits,
    PlacementIntent,
    PlacementStatus,
    place,
    quote,
    resolve,
    value_bet_intents,
)
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

from .conftest import FakeVenue

MATCH = 'Arsenal vs Chelsea'
OTHER = 'Liverpool vs Everton'
ARSENAL_PRICE = 2.10
CHELSEA_PRICE = 3.40
PRICES = {
    (MATCH, 'home_win', 'Arsenal'): ARSENAL_PRICE,
    (OTHER, 'home_win', 'Liverpool'): 1.80,
}
REF_CHARS = 32
STAKE = 10.0
OTHER_STAKE = 15.0
BATCH_STAKE = 25.0
OVER_STAKE = 50.0
FILLED = 4.0
OPEN_EXPOSURE = 40.0
QUOTED_EXPOSURE = 50.0
BALANCE = 250.0
BOTH = 2
FALLBACK_PRICE = 1.01


def identity(match=MATCH, market='home_win', selection='Arsenal', venue='fake'):
    """Return an identity."""
    return BetIdentity(venue, match, market, selection)


def intent(stake=10.0, min_price=2.0, **rest):
    """Return an intent."""
    return PlacementIntent(identity=identity(**rest), stake=stake, min_price=min_price, value_bet='row-1')


def limits(max_stake=100.0, max_exposure=1000.0, killed=False):
    """Return limits."""
    return ExposureLimits(max_stake_per_bet=max_stake, max_total_exposure=max_exposure, killed=killed)


def run(coroutine):
    """Run a coroutine."""
    return asyncio.run(coroutine)


def quoted(venue, intents):
    """Return a quote."""
    return run(quote(venue, intents, limits()))


def test_ref_is_thirty_two_hex_characters():
    """The reference fits the field a venue carries it in."""
    ref = identity().ref
    assert len(ref) == REF_CHARS
    assert all(character in '0123456789abcdef' for character in ref)


def test_ref_is_derived_rather_than_remembered():
    """The same four fields give the same reference, so a fresh run recomputes it."""
    assert identity().ref == identity().ref
    assert BetIdentity('fake', MATCH, 'home_win', 'Arsenal').ref == identity().ref


@pytest.mark.parametrize(
    ('field', 'value'),
    [('venue', 'other'), ('match', OTHER), ('market', 'draw'), ('selection', 'Chelsea')],
)
def test_ref_changes_with_every_field(field, value):
    """Each of the four fields is part of what makes a bet that bet."""
    assert identity(**{field: value}).ref != identity().ref


def test_same_four_fields_are_the_same_bet():
    """Two intents from different runs that share the four are one bet."""
    first = PlacementIntent(identity=identity(), stake=10.0, min_price=2.0, value_bet='run-1')
    second = PlacementIntent(identity=identity(), stake=25.0, min_price=1.5, value_bet='run-2')
    assert first.identity.ref == second.identity.ref


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


def test_nothing_is_staked_without_a_confirmation():
    """The default refuses, so a forgotten argument costs a run rather than a balance."""
    venue = FakeVenue(prices=PRICES)
    intents = [intent(), intent(match=OTHER, selection='Liverpool', min_price=1.5)]
    receipts = run(place(venue, quoted(venue, intents), limits()))
    assert (receipts['status'] == PlacementStatus.DRY_RUN.value).all()
    assert receipts['stake'].sum() == 0.0
    assert venue.orders == {}
    assert venue.attempts == []


def test_the_quote_is_stated_when_nothing_is_confirmed():
    """A caller that stakes nothing still learns what it would have staked."""
    venue = FakeVenue(prices=PRICES)
    batch = quoted(venue, [intent(stake=10.0), intent(stake=15.0, match=OTHER, selection='Liverpool', min_price=1.5)])
    receipts = run(place(venue, batch, limits()))
    assert batch.total_stake == BATCH_STAKE
    assert str(BATCH_STAKE) in receipts['detail'].iloc[0]


@pytest.mark.parametrize(
    ('confirm_stake', 'confirm_exposure'),
    [(999.0, 10.0), (10.0, 999.0), (999.0, 999.0), (None, 10.0), (10.0, None)],
)
def test_a_mismatched_confirmation_stakes_nothing(confirm_stake, confirm_exposure):
    """Only the figures that were quoted place anything."""
    venue = FakeVenue(prices=PRICES)
    batch = quoted(venue, [intent(stake=10.0)])
    receipts = run(place(venue, batch, limits(), confirm_stake, confirm_exposure))
    assert (receipts['status'] == PlacementStatus.REFUSED_UNCONFIRMED.value).all()
    assert receipts['stake'].sum() == 0.0
    assert venue.orders == {}


def test_a_mismatched_confirmation_states_the_real_figures():
    """A refusal says what the figures actually are."""
    venue = FakeVenue(prices=PRICES)
    batch = quoted(venue, [intent(stake=10.0)])
    receipts = run(place(venue, batch, limits(), 999.0, 999.0))
    detail = receipts['detail'].iloc[0]
    assert str(STAKE) in detail
    assert '999.0' in detail


def test_the_exact_confirmation_places_each_bet_once():
    """The figures that were quoted place the bets."""
    venue = FakeVenue(prices=PRICES)
    intents = [intent(stake=10.0), intent(stake=15.0, match=OTHER, selection='Liverpool', min_price=1.5)]
    batch = quoted(venue, intents)
    receipts = run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    assert (receipts['status'] == PlacementStatus.MATCHED_FULL.value).all()
    assert receipts['stake'].sum() == BATCH_STAKE
    assert len(venue.orders) == BOTH
    assert sorted(venue.attempts) == sorted(receipts['ref'])


def test_a_stake_over_the_ceiling_is_refused_and_the_ceiling_named():
    """A bet over the stake ceiling is refused, and the ceiling is named."""
    venue = FakeVenue(prices=PRICES)
    batch = quoted(venue, [intent(stake=OVER_STAKE)])
    receipts = run(place(venue, batch, limits(max_stake=10.0), batch.total_stake, batch.total_exposure))
    assert receipts['status'].iloc[0] == PlacementStatus.REFUSED_LIMIT.value
    assert 'maximum stake per bet' in receipts['detail'].iloc[0]
    assert str(STAKE) in receipts['detail'].iloc[0]
    assert venue.orders == {}


def test_a_batch_over_the_exposure_ceiling_is_refused_and_the_ceiling_named():
    """A batch over the exposure ceiling is refused, and the ceiling is named."""
    venue = FakeVenue(prices=PRICES)
    batch = quoted(venue, [intent(stake=30.0)])
    receipts = run(place(venue, batch, limits(max_exposure=20.0), batch.total_stake, batch.total_exposure))
    assert receipts['status'].iloc[0] == PlacementStatus.REFUSED_LIMIT.value
    assert 'maximum total exposure' in receipts['detail'].iloc[0]
    assert venue.orders == {}


def test_the_exposure_ceiling_reached_partway_keeps_what_went_on():
    """The bets that fit go on, and the ones that do not are refused."""
    venue = FakeVenue(prices=PRICES)
    intents = [intent(stake=15.0), intent(stake=15.0, match=OTHER, selection='Liverpool', min_price=1.5)]
    batch = quoted(venue, intents)
    receipts = run(place(venue, batch, limits(max_exposure=20.0), batch.total_stake, batch.total_exposure))
    assert receipts['status'].iloc[0] == PlacementStatus.MATCHED_FULL.value
    assert receipts['status'].iloc[1] == PlacementStatus.REFUSED_LIMIT.value
    assert len(venue.orders) == 1


def test_the_kill_switch_stakes_nothing():
    """The kill switch stops placing."""
    venue = FakeVenue(prices=PRICES)
    batch = quoted(venue, [intent(), intent(match=OTHER, selection='Liverpool', min_price=1.5)])
    receipts = run(place(venue, batch, limits(killed=True), batch.total_stake, batch.total_exposure))
    assert (receipts['status'] == PlacementStatus.REFUSED_KILLED.value).all()
    assert venue.orders == {}


def test_the_kill_switch_engaged_partway_keeps_what_went_on():
    """A batch stopped partway leaves the bets that already went on alone."""
    venue = FakeVenue(prices=PRICES)
    engaged = limits()
    intents = [intent(stake=10.0), intent(stake=10.0, match=OTHER, selection='Liverpool', min_price=1.5)]
    batch = quoted(venue, intents)

    original = venue.place

    async def kill_after_first(placed):
        engaged.killed = True
        return await original(placed)

    venue.place = kill_after_first
    receipts = run(place(venue, batch, engaged, batch.total_stake, batch.total_exposure))
    assert receipts['status'].iloc[0] == PlacementStatus.MATCHED_FULL.value
    assert receipts['status'].iloc[1] == PlacementStatus.REFUSED_KILLED.value
    assert len(venue.orders) == 1


def test_a_price_below_the_minimum_is_refused():
    """A bet below its minimum price is no longer a value bet, so it does not go on."""
    venue = FakeVenue(prices={(MATCH, 'home_win', 'Arsenal'): 1.50})
    batch = quoted(venue, [intent(min_price=2.0)])
    receipts = run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    assert receipts['status'].iloc[0] == PlacementStatus.REFUSED_PRICE.value
    assert '1.5' in receipts['detail'].iloc[0]
    assert venue.orders == {}


def test_a_price_at_the_minimum_goes_on():
    """The minimum is what is acceptable, not what is refused."""
    venue = FakeVenue(prices={(MATCH, 'home_win', 'Arsenal'): 2.00})
    batch = quoted(venue, [intent(min_price=2.0)])
    receipts = run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    assert receipts['status'].iloc[0] == PlacementStatus.MATCHED_FULL.value


def test_a_retry_after_a_timeout_stakes_once():
    """A bet the venue took before the connection dropped is not staked again."""
    venue = FakeVenue(prices=PRICES, fail_after_accept=1)
    batch = quoted(venue, [intent(stake=10.0)])
    with pytest.raises(TimeoutError):
        run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    assert len(venue.orders) == 1

    venue.fail_after_accept = None
    receipts = run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    assert receipts['status'].iloc[0] == PlacementStatus.ALREADY_PLACED.value
    assert len(venue.orders) == 1


def test_a_fresh_run_that_lost_its_state_stakes_once():
    """A run that remembers nothing recomputes the reference and finds its own bet."""
    venue = FakeVenue(prices=PRICES)
    first = quoted(venue, [intent(stake=10.0)])
    run(place(venue, first, limits(), first.total_stake, first.total_exposure))
    assert len(venue.orders) == 1

    second = quoted(venue, [intent(stake=10.0)])
    receipts = run(place(venue, second, limits(), second.total_stake, second.total_exposure))
    assert receipts['status'].iloc[0] == PlacementStatus.ALREADY_PLACED.value
    assert len(venue.orders) == 1


def test_two_selections_on_one_market_are_two_bets():
    """Backing both sides of a market is two bets, so both go on.

    The once-only promise must not become an over-merging one: an identity that dropped the selection would report the
    second of these as already placed and quietly stake nothing.
    """
    venue = FakeVenue(
        prices={(MATCH, 'home_win', 'Arsenal'): 2.10, (MATCH, 'home_win', 'Chelsea'): CHELSEA_PRICE},
    )
    intents = [intent(stake=10.0), intent(stake=10.0, selection='Chelsea', min_price=3.0)]
    batch = quoted(venue, intents)
    receipts = run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    assert (receipts['status'] == PlacementStatus.MATCHED_FULL.value).all()
    assert len(venue.orders) == BOTH
    assert receipts['ref'].nunique() == BOTH


def test_the_same_selection_at_two_venues_are_two_bets():
    """One selection at two venues is two bets, since the venue is part of what makes a bet that bet."""
    assert identity(venue='fake').ref != identity(venue='other').ref


def test_placing_keeps_no_state_of_its_own(tmp_path, monkeypatch):
    """Nothing is written down, since the venue is the record and a second copy drifts."""
    monkeypatch.chdir(tmp_path)
    venue = FakeVenue(prices=PRICES)
    batch = quoted(venue, [intent(stake=10.0)])
    run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    assert list(tmp_path.iterdir()) == []


def test_the_bets_go_on_one_at_a_time():
    """The bets are placed in order, one after another."""
    venue = FakeVenue(prices=PRICES)
    intents = [intent(stake=10.0), intent(stake=10.0, match=OTHER, selection='Liverpool', min_price=1.5)]
    batch = quoted(venue, intents)
    receipts = run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    assert venue.placed_order == list(receipts['ref'])


def test_a_blocked_venue_is_reported_and_placing_stops():
    """A venue that blocks automation is reported, and nothing gets around it."""
    venue = FakeVenue(prices=PRICES, blocked=True)
    intents = [intent(stake=10.0), intent(stake=10.0, match=OTHER, selection='Liverpool', min_price=1.5)]
    batch = quoted(venue, intents)
    receipts = run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    assert (receipts['status'] == PlacementStatus.BLOCKED.value).all()
    assert venue.orders == {}
    assert len(venue.attempts) == 1


def test_a_partial_match_says_so():
    """An exchange that fills part of a stake reports what it filled."""
    venue = FakeVenue(prices=PRICES, matched=FILLED)
    batch = quoted(venue, [intent(stake=10.0)])
    receipts = run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    assert receipts['status'].iloc[0] == PlacementStatus.MATCHED_PARTIAL.value
    assert receipts['stake'].iloc[0] == FILLED


def test_the_receipt_traces_back_to_the_value_bet():
    """Every receipt says which bet it came from and what it got."""
    venue = FakeVenue(prices=PRICES)
    batch = quoted(venue, [intent(stake=10.0)])
    receipts = run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    row = receipts.iloc[0]
    assert row['venue'] == 'fake'
    assert row['match'] == MATCH
    assert row['market'] == 'home_win'
    assert row['selection'] == 'Arsenal'
    assert row['value_bet'] == 'row-1'
    assert row['price'] == ARSENAL_PRICE


def test_the_quote_counts_the_exposure_already_open():
    """The exposure is what is at stake, including what was already there."""
    venue = FakeVenue(prices=PRICES, exposure=OPEN_EXPOSURE)
    batch = quoted(venue, [intent(stake=10.0)])
    assert batch.total_stake == STAKE
    assert batch.total_exposure == QUOTED_EXPOSURE


def test_the_venue_reports_what_it_holds():
    """A bet that went on can be read back from the venue."""
    venue = FakeVenue(prices=PRICES)
    batch = quoted(venue, [intent(stake=STAKE)])
    run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    held = run(venue.read_status([identity()]))
    assert list(held['ref']) == [identity().ref]
    assert list(held['stake']) == [STAKE]


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


def test_a_venue_that_can_cancel_cancels():
    """A cancelled bet is no longer at the venue."""
    venue = FakeVenue(prices=PRICES)
    batch = quoted(venue, [intent(stake=STAKE)])
    run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    assert len(venue.orders) == 1
    run(venue.cancel(identity()))
    assert venue.orders == {}


def test_a_cancelled_bet_can_be_placed_again():
    """A bet that was cancelled is no longer at the venue, so it is no longer already placed."""
    venue = FakeVenue(prices=PRICES)
    batch = quoted(venue, [intent(stake=STAKE)])
    run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    run(venue.cancel(identity()))
    receipts = run(place(venue, batch, limits(), batch.total_stake, batch.total_exposure))
    assert receipts['status'].iloc[0] == PlacementStatus.MATCHED_FULL.value


def test_value_bet_intents_handles_a_shared_kickoff_and_real_odds():
    """Two matches at the same kickoff are two intents with their own prices.

    Football fills a Saturday afternoon with matches at the same time, so the fixtures index repeats, and the odds
    column is `{provider}__{market}__{status}__{time}` rather than `{market}__odds`. An earlier version indexed the
    fixtures by the shared kickoff, which returned every match at once and garbled the name, and looked for an odds
    column that never existed, which floored every price at the fallback.
    """
    loader = DataLoader(param_grid={'league': ['England']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds())
    X, Y, O = loader.extract_train_data(odds_type='market_maximum')
    assert X.index.duplicated().any()
    bettor = OddsComparisonBettor(alpha=0.03, betting_markets=['home_win', 'draw', 'away_win']).fit(X, Y, O)

    intents = value_bet_intents('demo', bettor, X.head(40), O.head(40), stake=10.0)
    assert intents
    assert all('\n' not in intent.identity.match for intent in intents)
    assert all(' vs ' in intent.identity.match for intent in intents)
    assert any(intent.min_price != FALLBACK_PRICE for intent in intents)
    assert len({intent.identity.ref for intent in intents}) == len(intents)
