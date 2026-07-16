"""
Odds comparison bettor
======================

This example illustrates OddsComparisonBettor, which bets by comparing
the odds of different providers rather than by learning from the features.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt
import numpy as np
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
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum')

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
# time as well as at training time, unlike a classifier bettor, which learns from the features alone.

bettor.bet(X_train, O_train)

# %%
# Backtesting
# -----------

results = backtest(bettor, X_train, Y_train, O_train, cv=TimeSeriesSplit(3))
results

# %%
# Turning the one knob
# --------------------
#
# `alpha` is the whole model. Small, and it bets on almost every match, taking any price a shade above the average.
# Large, and it waits for the rare, glaring mispricing. Sweep it and you can watch the bettor go from greedy to picky.

alphas = np.linspace(0.0, 0.12, 13)
placed = []
for alpha in alphas:
    picky = OddsComparisonBettor(alpha=alpha, betting_markets=['home_win', 'draw', 'away_win'])
    picky.fit(X_train, Y_train, O_train)
    placed.append(int(picky.bet(X_train, O_train).sum()))

fig, ax = plt.subplots()
ax.plot(alphas, placed, marker='o')
ax.set_title('The bettor gets pickier as alpha rises')
ax.set_xlabel('alpha (how far above average a price must be)')
ax.set_ylabel('value bets found')
