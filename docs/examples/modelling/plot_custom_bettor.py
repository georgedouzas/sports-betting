"""
A bettor of your own
====================

This example illustrates [`BaseBettor`][sportsbet.evaluation.BaseBettor] and
[`complementary_events`][sportsbet.evaluation.complementary_events].
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from sportsbet.dataloaders import DataLoader
from sportsbet.evaluation import BaseBettor, backtest, complementary_events
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

# %%
# Two methods to implement
# ------------------------
#
# A bettor turns probabilities into bets. Implement `_fit` and `_predict_proba`, and the value bets, the backtest
# and the bankroll all follow: a bet is placed when the probability your model gives an outcome is higher than the one
# the price implies.


class BaseRateBettor(BaseBettor):
    """A bettor that knows only how often each outcome has happened."""

    def _fit(self, X, Y, O):
        # `Y` carries the markets it was told to bet, in the order it was told them.
        self.rates_ = Y.mean().to_numpy()
        return self

    def _predict_proba(self, X):
        rates = np.tile(self.rates_, (len(X), 1))
        return rates / rates.sum(axis=1, keepdims=True)


# %%
# Extracting the data
# -------------------

dataloader = DataLoader(
    param_grid={'league': ['England']},
    stats=SampleSoccerStats(),
    odds=SampleSoccerOdds(),
)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average', download=True)

# %%
# Betting with it
# ---------------

bettor = BaseRateBettor(betting_markets=['home_win', 'draw', 'away_win'])
_ = bettor.fit(X_train, Y_train, O_train)
bettor.predict_proba(X_train)

# %%
backtest(bettor, X_train, Y_train, O_train, cv=TimeSeriesSplit(3))

# %%
# Which markets are mutually exclusive
# ------------------------------------
#
# The probabilities of a group of complementary markets must sum to one, and the groups come from the **data** rather
# than from a list somebody wrote down. A sport that cannot be drawn simply has two outcomes instead of three, and
# nothing had to be told which sport this is.

complementary_events(['home_win', 'draw', 'away_win', 'over_2.5', 'under_2.5'])

# %%
complementary_events(['home_win', 'away_win'])

# %%
# A picture of it
# ---------------

markets = bettor.betting_markets_.tolist()

fig, ax = plt.subplots()
ax.bar(markets, bettor.predict_proba(X_train)[0])
ax.set_title('The base rates of the Premier League season')
ax.set_ylabel('probability')
