"""
Saving and loading a bettor
===========================

This example illustrates [`save_bettor`][sportsbet.evaluation.save_bettor] and
[`load_bettor`][sportsbet.evaluation.load_bettor].
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sportsbet.dataloaders import DataLoader
from sportsbet.evaluation import OddsComparisonBettor, load_bettor, save_bettor
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

# %%
# A fitted bettor comes back fitted
# ---------------------------------
#
# The model you backtested is the model that should place the bets. Saving it is what makes those the same object rather
# than two that were fitted separately and are hopefully the same.

dataloader = DataLoader(
    param_grid={'league': ['England']},
    stats=SampleSoccerStats(),
    odds=SampleSoccerOdds(),
)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average', download=True)

bettor = OddsComparisonBettor(alpha=0.03, betting_markets=['home_win', 'draw', 'away_win'])
_ = bettor.fit(X_train, Y_train, O_train)

path = str(Path(tempfile.mkdtemp()) / 'bettor.pkl')
save_bettor(bettor, path)

# %%
# It is ready to bet without being fitted again.

reloaded = load_bettor(path)
reloaded.betting_markets_.tolist()

# %%
reloaded.bet(X_train, O_train)

# %%
# A picture of it
# ---------------

markets = reloaded.betting_markets_.tolist()
counts = pd.DataFrame(reloaded.bet(X_train, O_train), columns=markets).sum()

fig, ax = plt.subplots()
ax.bar(counts.index, counts.to_numpy())
ax.set_title('Value bets found, by market')
ax.set_ylabel('bets')
