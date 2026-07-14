"""
EuroLeague basketball
=====================

This example illustrates [`EuroLeagueStats`][sportsbet.sources.EuroLeagueStats], the free statistics of the
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
# Extracting the data without odds
# --------------------------------
#
# There is no free basketball odds feed anywhere, so the odds are yours to buy — see
# [`OddsApi`](plot_odds_api.md). Without them there are no markets and so nothing to predict, and the dataloader says
# exactly that rather than inventing a target. The features are still worth having, so ask for them.

dataloader = DataLoader(param_grid={'league': ['Euroleague'], 'year': [2024]}, stats=stats)
X_train, Y_train, O_train = dataloader.extract_train_data(learning_type='unsupervised', download=True)

# %%
# The input data:
X_train

# %%
# There is no target and there are no odds, which is the honest answer rather than a fabricated one.

Y_train is None, O_train.empty

# %%
# Two things fall out of the data rather than being configured. There is **no draw**, since a tie goes to overtime, so
# the outcome is two-way. And there is **no totals market**, because a bookmaker sets a different line for every game,
# and a market whose line moves is not a column.

# %%
# A picture of it
# ---------------

fig, ax = plt.subplots()
ax.hist(X_train['home_points_for_avg'].dropna(), bins=30)
ax.set_title('EuroLeague: points the home team scores, on average, before the game')
ax.set_xlabel('points per game')
ax.set_ylabel('games')
