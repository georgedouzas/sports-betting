"""
Odds comparison bettor
======================

This example illustrates [`OddsComparisonBettor`][sportsbet.evaluation.OddsComparisonBettor], which bets by comparing
the odds of different providers rather than by learning from the features.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from sportsbet.dataloaders import DataLoader
from sportsbet.evaluation import OddsComparisonBettor, backtest
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

# %%
# Extracting the data
# -------------------

dataloader = DataLoader(
    param_grid={'league': ['England']},
    stats=SampleSoccerStats(),
    odds=SampleSoccerOdds(),
)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum', download=True)

# %%
# A model that needs no features
# ------------------------------
#
# It compares what one provider offers with what another does. When the best price in the market is far enough above the
# average, the outcome is priced more generously than the market as a whole believes, and that is the bet.
#
# `alpha` is how far above is far enough.

bettor = OddsComparisonBettor(alpha=0.03, betting_markets=['home_win', 'draw', 'away_win'])
_ = bettor.fit(X_train, Y_train, O_train)

# %%
# The value bets, one boolean column per market. This bettor models the odds themselves, so it needs them at prediction
# time as well as at training time — unlike a classifier bettor, which learns from the features alone.

bettor.bet(X_train, O_train)

# %%
# Backtesting
# -----------

results = backtest(bettor, X_train, Y_train, O_train, cv=TimeSeriesSplit(3))
results

# %%
# A picture of it
# ---------------

yields = results['Yield percentage per bet'].to_numpy()

fig, ax = plt.subplots()
ax.bar(range(1, len(yields) + 1), yields)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Odds comparison bettor: yield per bet, by fold')
ax.set_xlabel('fold')
ax.set_ylabel('yield %')
