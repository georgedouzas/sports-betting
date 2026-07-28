"""
Dataloader
==========

A DataLoader turns the data from your sources into training and fixtures data. This example builds one, extracts the
training data and the fixtures, and plots the odds.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

# %%
# One dataloader for every sport
# -------------------------------
#
# The same `DataLoader` works for every sport. The sport comes from the sources you give it, not from anything you
# configure, and `sport_` reports what it found.

dataloader = DataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2024]},
    stats=SampleSoccerStats(),
    odds=SampleSoccerOdds(),
)
dataloader.sport_

# %%
# Selecting the data
# ------------------
#
# `param_grid` selects what to train on. Any dimension you leave out takes all of its available values. The dataloader
# never requests a combination the sources do not publish.

dataloader.sources_

# %%
# The available odds types are the providers the data carries. The dataloader reads them from the data rather than from
# a registry.

dataloader.get_odds_types()

# %%
# Extracting the training data
# ----------------------------
#
# This is where the download happens. It pulls the selected seasons and returns three frames. The dataloader keeps
# them, so once you have extracted, `save` carries the data with it and nothing has to be fetched twice.

X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')

# %%
# The input data:
X_train

# %%
# The multi-output targets:
Y_train

# %%
# The odds:
O_train

# %%
# Extracting the fixtures data
# ----------------------------
#
# A fixture is a match that has not been played. `param_grid` picks the leagues and seasons you train on. The fixtures
# are the upcoming matches in those leagues, so they never overlap the seasons you trained on. The two frames share
# their columns, not their contents.
#
# The sample is a finished season, so it has no fixtures. Use a live source to get some.

X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
{'fixtures': len(X_fix), 'same columns': list(X_fix.columns) == list(X_train.columns)}

# %%
# There is no target for a match that has not been played:

{'no target for an unplayed match': Y_fix is None}

# %%
# A picture of it
# ---------------

home_win = next(col for col in O_train.columns if '__home_win__' in col)

fig, ax = plt.subplots()
ax.hist(O_train[home_win].dropna(), bins=40)
ax.set_title('Market average odds on the home team')
ax.set_xlabel('decimal odds')
ax.set_ylabel('matches')
