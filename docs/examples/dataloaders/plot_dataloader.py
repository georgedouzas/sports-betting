"""
Dataloader
==========

This example illustrates [`DataLoader`][sportsbet.dataloaders.DataLoader], the object that turns the data of your
sources into training and fixtures data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

# %%
# There is one dataloader
# -----------------------
#
# Not one per sport. The sport is a property of the sources, so a dataloader never has to be told what it is looking at.
# You choose the sources, so you always know where your data came from.

dataloader = DataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2024]},
    stats=SampleSoccerStats(),
    odds=SampleSoccerOdds(),
)
dataloader.sport

# %%
# Selecting the data
# ------------------
#
# `param_grid` selects what to **train** on. Any dimension left out takes all of its available values, and a combination
# the sources do not publish is never requested.

dataloader.sources

# %%
# The available odds types are the providers the data carries, and they are read from it rather than registered.

dataloader.get_odds_types()

# %%
# Extracting the training data
# ----------------------------
#
# `download` is the only thing that reaches the network. Leave it out and nothing is fetched: you are told how many
# requests it would take, and you decide.

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
# A fixture is a match that has **not been played**. It is *not* restricted by `param_grid`: that selects what to train
# on, and a match you could have trained on has by definition already been played. So you may train on England and bet
# on Italy. What the two frames share is their **columns**, not their contents.
#
# The sample is a finished season, so it has no fixtures at all. Use a live source to get some.

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
