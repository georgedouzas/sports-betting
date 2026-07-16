"""
EuroLeague basketball
=====================

This example illustrates EuroLeagueStats, the free statistics of the
EuroLeague's own API.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import EuroLeagueStats

# %%
# A different sport, the same dataloader
# --------------------------------------
#
# The sport is a property of the source, not of the dataloader. There is one dataloader, and it is basketball because
# this source is.

stats = EuroLeagueStats()
stats.name, stats.kind, stats.sport

# %%
# The seasons it publishes. A whole season arrives in a single request, and it needs no key.

params = stats.available_params()
sorted({param['year'] for param in params})[-5:]

# %%
# Extracting the features
# -----------------------
#
# There is no free basketball odds feed anywhere, so the odds are yours to buy. See
# [`OddsApi`](plot_odds_api.md). Without them there are no markets and so nothing to predict, so `extract_train_data`
# would stop and say so. The features are still worth having, so `extract_exploration_data` returns them on their own.

dataloader = DataLoader(param_grid={'league': ['Euroleague'], 'year': [2024]}, stats=stats)
X = dataloader.extract_exploration_data()

# %%
# The features, with no targets and no odds:
X

# %%
# Two things fall out of the data rather than being configured. There is no draw, since a tie goes to overtime, so
# the outcome is two-way. And there is no totals market, because a bookmaker sets a different line for every game,
# and a market whose line moves is not a column.

# %%
# A picture of it
# ---------------
#
# The form the two sides bring, side by side. The home teams sit a little to the right of the away teams, which is the
# home advantage falling out of the data rather than being put there.

fig, ax = plt.subplots()
ax.hist(X['home_points_for_avg'].dropna(), bins=25, alpha=0.6, label='home team')
ax.hist(X['away_points_for_avg'].dropna(), bins=25, alpha=0.6, label='away team')
ax.set_title('EuroLeague: scoring form the two sides bring, before tip-off')
ax.set_xlabel('points per game')
ax.set_ylabel('games')
ax.legend()
