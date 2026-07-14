"""
Sample soccer data
==================

This example illustrates [`SampleSoccerStats`][sportsbet.sources.SampleSoccerStats] and
[`SampleSoccerOdds`][sportsbet.sources.SampleSoccerOdds], the sample data that ships with the library.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

# %%
# What the sample carries
# -----------------------
#
# It is a real season of the English and Spanish first divisions, taken from football-data.co.uk and frozen. It needs
# no key and it reaches no network, which is what makes it the data of the examples and the tests.

stats = SampleSoccerStats()
stats.name, stats.kind, stats.sport

# %%
# It ships with the library, so it knows what it publishes without reading anything.

stats.available_params()

# %%
# Extracting the data
# -------------------
#
# It is an ordinary source, so it is used like any other: give it to a dataloader beside an odds source. The
# `download=True` copies the bundled files into the store, which costs nothing and touches no network.

dataloader = DataLoader(param_grid={'league': ['England']}, stats=stats, odds=SampleSoccerOdds())
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average', download=True)

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
# It has no fixtures
# ------------------
#
# The season is finished, so every match in it has been played. A fixture is a match that has **not** been played, so
# the sample has none, and `extract_fixtures_data` returns an empty frame with the training columns.
#
# To bet on something you need a source that is still publishing matches — see
# [Football-Data](plot_football_data.md). The sample is for learning the interface, not for betting.

X_fix, _, O_fix = dataloader.extract_fixtures_data()
len(X_fix)

# %%
# A picture of it
# ---------------

outcomes = Y_train.sum()
outcomes.index = outcomes.index.str.split('__').str[0]

fig, ax = plt.subplots()
ax.bar(outcomes.index, outcomes.to_numpy())
ax.set_title('Outcomes of the Premier League sample season')
ax.set_ylabel('matches')
fig.autofmt_xdate(rotation=30)
