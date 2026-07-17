"""
Placing value bets safely
=========================

This example places the value bets a bettor found, at a venue you write yourself. The library ships no bookmaker, so a
venue with an API is a `BaseVenue` you implement. Nothing here reaches a real bookmaker or stakes real money: the venue
below keeps its bets in memory.

The point of the example is the safety model. Placing stakes nothing until you pass back the total it quotes, and it
places each bet once, so a repeat run does not stake twice.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import asyncio

import matplotlib.pyplot as plt
import pandas as pd

from sportsbet.dataloaders import DataLoader
from sportsbet.evaluation import OddsComparisonBettor
from sportsbet.execution import (
    BaseVenue,
    ExposureLimits,
    PlacementReceipt,
    PlacementStatus,
    place,
    quote,
    value_bet_intents,
)
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

# %%
# A venue you write
# -----------------
#
# A venue with an API implements the contract. Yours would call that API. This one keeps its bets in a dictionary, so
# the example runs without a network or an account. It places a bet once for an identity: asked again for the same
# selection, it reports the bet it already holds rather than staking a second time.


class DemoVenue(BaseVenue):
    """An in-memory venue, for the example.

    A real one would call a bookmaker's API.
    """

    key = 'demo'
    can_cancel = True

    def __init__(self, prices):
        """Keep the prices on offer and an empty book of placed bets."""
        self.prices = prices
        self.orders = {}

    async def authenticate(self):
        """A demo venue guards nothing."""

    async def list_markets(self, matches):
        """Return the prices on offer for the given matches."""
        records = [
            {'match': match, 'market': market, 'selection': selection, 'price': price}
            for (match, market, selection), price in self.prices.items()
            if match in matches
        ]
        return pd.DataFrame.from_records(records, columns=['match', 'market', 'selection', 'price'])

    async def read_balance(self):
        """Return a balance and the exposure of the bets already placed."""
        return 10000.0, sum(order['stake'] for order in self.orders.values())

    async def place(self, intent):
        """Record a bet once for its identity, reporting the one already held on a repeat."""
        ref = intent.identity.ref
        if ref in self.orders:
            order = self.orders[ref]
            return PlacementReceipt(
                identity=intent.identity,
                status=PlacementStatus.ALREADY_PLACED,
                stake=order['stake'],
                price=order['price'],
                value_bet=intent.value_bet,
                detail='This selection already has a bet.',
            )
        price = self.prices[(intent.identity.match, intent.identity.market, intent.identity.selection)]
        self.orders[ref] = {'stake': intent.stake, 'price': price}
        return PlacementReceipt(
            identity=intent.identity,
            status=PlacementStatus.MATCHED_FULL,
            stake=intent.stake,
            price=price,
            value_bet=intent.value_bet,
        )

    async def read_status(self, identities):
        """Return the bets the venue holds for these identities."""
        records = [{'ref': i.ref, **self.orders[i.ref]} for i in identities if i.ref in self.orders]
        return pd.DataFrame.from_records(records)

    async def cancel(self, identity):
        """Cancel a bet."""
        self.orders.pop(identity.ref, None)
        return PlacementReceipt(identity=identity, status=PlacementStatus.REJECTED, detail='Cancelled.')


# %%
# The value bets to place
# -----------------------
#
# A bettor finds the value bets the ordinary way, and `value_bet_intents` turns them into intents. Each intent carries
# a minimum price, the price the value bet was computed at, since below it the bet is no longer a value bet.

dataloader = DataLoader(param_grid={'league': ['England']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds())
X, Y, O = dataloader.extract_train_data(odds_type='market_maximum')
bettor = OddsComparisonBettor(alpha=0.05, betting_markets=['home_win', 'draw', 'away_win']).fit(X, Y, O)

intents = value_bet_intents('demo', bettor, X.head(30), O.head(30), stake=10.0)[:8]
pd.DataFrame(
    {
        'match': [intent.identity.match for intent in intents],
        'market': [intent.identity.market for intent in intents],
        'min_price': [round(intent.min_price, 2) for intent in intents],
    },
)

# %%
# The venue offers each selection at its value-bet price, so nothing is refused on price.

prices = {(i.identity.match, i.identity.market, i.identity.selection): i.min_price for i in intents}
venue = DemoVenue(prices)
limits = ExposureLimits(max_stake_per_bet=10.0, max_total_exposure=1000.0)
quoted = asyncio.run(quote(venue, intents, limits))
quoted.total_stake

# %%
# Refusal is the default
# ----------------------
#
# Placing without the quoted total stakes nothing. It is a dry run, not because a flag was set, but because the
# confirmation is missing.

receipts = asyncio.run(place(venue, quoted, limits))
receipts[['match', 'market', 'status', 'stake']]

# %%
# The wrong total stakes nothing either, and says what the total really is.

receipts = asyncio.run(place(venue, quoted, limits, confirm_stake=1.0, confirm_exposure=1.0))
receipts['detail'].iloc[0]

# %%
# The quoted total places the bets, each with a receipt and a price.

receipts = asyncio.run(
    place(venue, quoted, limits, confirm_stake=quoted.total_stake, confirm_exposure=quoted.total_exposure),
)
receipts[['match', 'market', 'status', 'stake', 'price']]

# %%
# Placed once and only once
# -------------------------
#
# Run the same batch again. A retry, a reconnection or a fresh run recomputes the same identity and finds the bet
# already at the venue, so nothing is staked twice.

again = asyncio.run(
    place(venue, quoted, limits, confirm_stake=quoted.total_stake, confirm_exposure=quoted.total_exposure),
)
again[['match', 'status', 'stake']]

# %%
# The whole safety model in one view. Only newly staked money is counted, since an already placed bet reports the stake
# it already holds. The default and the wrong total stake nothing, the quoted total stakes the batch, and the repeat
# stakes nothing again.

scenarios = {
    'no confirmation': asyncio.run(place(DemoVenue(venue.prices), quoted, limits)),
    'wrong total': asyncio.run(place(DemoVenue(venue.prices), quoted, limits, confirm_stake=1.0, confirm_exposure=1.0)),
    'quoted total': receipts,
    'repeat run': again,
}
placed = {PlacementStatus.MATCHED_FULL.value, PlacementStatus.MATCHED_PARTIAL.value, PlacementStatus.ACCEPTED.value}
staked = {name: float(frame.loc[frame['status'].isin(placed), 'stake'].sum()) for name, frame in scenarios.items()}

fig, ax = plt.subplots()
ax.bar(staked.keys(), staked.values(), color=['#4c72b0', '#4c72b0', '#c44e52', '#4c72b0'])
ax.set_title('Nothing is staked until the quoted total is passed back')
ax.set_ylabel('money newly staked')
ax.axhline(quoted.total_stake, linestyle='--', color='grey', linewidth=1)
fig.autofmt_xdate(rotation=20)
