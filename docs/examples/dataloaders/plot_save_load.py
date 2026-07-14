"""
Saving and loading a dataloader
===============================

This example illustrates `save` and [`load_dataloader`][sportsbet.dataloaders.load_dataloader].
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

from sportsbet.dataloaders import DataLoader, load_dataloader
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

# %%
# A dataloader remembers what it was told
# ---------------------------------------
#
# The columns of the fixtures data are the columns of the training data, so a dataloader that has extracted once carries
# the shape with it. Saving it keeps that, which is what lets you extract the training data on one machine and predict
# on another.

dataloader = DataLoader(
    param_grid={'league': ['England']},
    stats=SampleSoccerStats(),
    odds=SampleSoccerOdds(),
)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average', download=True)

path = str(Path(tempfile.mkdtemp()) / 'dataloader.pkl')
_ = dataloader.save(path)

# %%
# It comes back knowing its selection and its columns.

loaded = load_dataloader(path)
{'same selection': loaded.param_grid_ == dataloader.param_grid_}

# %%
X_fix, _, O_fix = loaded.extract_fixtures_data()
{'same columns': list(X_fix.columns) == list(X_train.columns)}

# %%
# A picture of it
# ---------------

fig, ax = plt.subplots()
ax.bar(['X train', 'X fixtures'], [len(X_train.columns), len(X_fix.columns)])
ax.set_title('The fixtures carry the columns of the training data')
ax.set_ylabel('columns')
