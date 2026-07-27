"""
Placing value bets
===================

This example places the value bets a bettor found, at a venue you write yourself. The library ships no bookmaker, so a
venue with an API is a `BaseVenue` you implement. Nothing here reaches a real bookmaker or stakes real money: the venue
below keeps its bets in memory.

The point of the example is that a bet is placed once for its identity, so a repeat run does not stake a second time.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import asyncio

import matplotlib.pyplot as plt
import pandas as pd

from sportsbet.dataloaders import DataLoader
from sportsbet.evaluation import OddsComparisonBettor, find_latest_odds_column
from sportsbet.execution import (
    BaseVenue,
    BetIdentity,
    PlacementIntent,
    PlacementReceipt,
    PlacementStatus,
    build_receipts_frame,
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
        ref = intent.identity.ref_
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
        records = [{'ref': i.ref_, **self.orders[i.ref_]} for i in identities if i.ref_ in self.orders]
        return pd.DataFrame.from_records(records)

    async def cancel(self, identity):
        """Cancel a bet."""
        self.orders.pop(identity.ref_, None)
        return PlacementReceipt(identity=identity, status=PlacementStatus.REJECTED, detail='Cancelled.')


# %%
# The value bets to place
# -----------------------
#
# A bettor finds the value bets the ordinary way. Each value bet becomes a `PlacementIntent`: the bet to place, its
# stake, and a minimum price, the price the value bet was computed at, below which it is no longer value.

STAKE = 10.0
FALLBACK_PRICE = 1.01


def build_intents(bettor, X_fix, O_fix, stake):
    """Return an intent for each value bet the bettor found."""
    markets = list(bettor.betting_markets_)
    odds_columns = {market: find_latest_odds_column(list(O_fix.columns), market) for market in markets}
    value_bets = pd.DataFrame(bettor.bet(X_fix, O_fix), columns=markets)
    built = []
    for position in range(len(value_bets)):
        game = X_fix.iloc[position]
        match = f'{game["home_team"]} vs {game["away_team"]}'
        selection = str(game['home_team'])
        for market in markets:
            if not value_bets.iloc[position][market]:
                continue
            column = odds_columns[market]
            price = O_fix.iloc[position][column] if column is not None else None
            built.append(
                PlacementIntent(
                    identity=BetIdentity('demo', match, market, selection),
                    stake=stake,
                    min_price=float(price) if price is not None and not pd.isna(price) else FALLBACK_PRICE,
                    value_bet=f'{match}|{market}',
                ),
            )
    return built


dataloader = DataLoader(param_grid={'league': ['England']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds())
X, Y, O = dataloader.extract_train_data(odds_type='market_maximum')
bettor = OddsComparisonBettor(alpha=0.05, betting_markets=['home_win', 'draw', 'away_win']).fit(X, Y, O)

intents = build_intents(bettor, X.head(30), O.head(30), STAKE)[:8]
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

# %%
# Placing each bet
# ----------------
#
# Each intent is placed on its own by calling `place` on the venue. The venue returns a receipt saying what became of
# the bet, and `build_receipts_frame` gathers the receipts into one validated table.


async def place_bets(placed):
    """Place each intent in turn and return its receipt."""
    return [await venue.place(intent) for intent in placed]


receipts = build_receipts_frame(asyncio.run(place_bets(intents)))
receipts[['match', 'market', 'status', 'stake', 'price']]

# %%
# Placed once and only once
# -------------------------
#
# Place the same bets again. A retry, a reconnection or a fresh run recomputes the same identity and finds the bet
# already at the venue, so nothing is staked twice.

again = build_receipts_frame(asyncio.run(place_bets(intents)))
again[['match', 'status', 'stake']]

# %%
# What was staked, in one view. Only newly staked money is counted, since an already placed bet reports the stake it
# already holds rather than adding to it. The first run stakes the bets, and the repeat stakes nothing.

placed = {PlacementStatus.MATCHED_FULL.value, PlacementStatus.MATCHED_PARTIAL.value, PlacementStatus.ACCEPTED.value}
staked = {
    'first run': float(receipts.loc[receipts['status'].isin(placed), 'stake'].sum()),
    'repeat run': float(again.loc[again['status'].isin(placed), 'stake'].sum()),
}

fig, ax = plt.subplots()
ax.bar(staked.keys(), staked.values(), color=['#4c72b0', '#c44e52'])
ax.set_title('A bet is placed once, so a repeat run stakes nothing')
ax.set_ylabel('money newly staked')
