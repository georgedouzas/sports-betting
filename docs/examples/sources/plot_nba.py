"""
NBA basketball
==============

This example illustrates [`NBAStats`][sportsbet.sources.NBAStats], the free NBA statistics that ESPN publishes.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import EuroLeagueStats, NBAStats

# %%
# A league is a source, not a dataloader
# --------------------------------------
#
# The NBA is the same sport as the EuroLeague, so it is the same dataloader with different statistics.

stats = NBAStats()
stats.name, stats.kind, stats.sport

# %%
{'same sport as the EuroLeague': NBAStats().sport == EuroLeagueStats().sport}

# %%
# A season is named by the year it **ends** in, so 2026 is the 2025-26 season. It carries the regular season, the
# play-in and the play-offs, but never the pre-season and never the all-star weekend, whose teams are not clubs.

params = stats.available_params()
sorted({param['year'] for param in params})[-5:]

# %%
# It is **live**: the games played this week carry their scores this week, which is what makes the current season
# bettable rather than merely reviewable. The NBA's own official archive publishes a season's results only months after
# it has ended, so a source built on that could backtest the league and never bet on it.

# %%
# Extracting the data
# -------------------
#
# It is free and needs no key. The odds are another source, and there is no free one for basketball.

dataloader = DataLoader(param_grid={'league': ['NBA'], 'year': [2024]}, stats=stats)
X_train, _, _ = dataloader.extract_train_data(learning_type='unsupervised')
X_train

# %%
# A picture of it
# ---------------

fig, ax = plt.subplots()
ax.scatter(X_train['home_points_for_avg'], X_train['away_points_for_avg'], s=8, alpha=0.4)
ax.set_title('NBA: the form of the two teams, before tip-off')
ax.set_xlabel('home points per game')
ax.set_ylabel('away points per game')
