"""
Soccer data
===========

This example illustrates the usage of the [`DataLoader`][sportsbet.dataloaders.DataLoader].
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import FootballDataOdds, FootballDataStats

# %%
# Getting the available parameters
# --------------------------------
#
# A parameter grid cannot be written before it is known what exists, so we ask the data
# source rather than a dataloader. What a source publishes depends on how it is
# configured, since a credential may only cover part of what it offers.

FootballDataStats().available_params()

# %%
# We select to extract training data only for the year 2021 of the first
# division Spanish and Italian leagues.

param_grid = {'league': ['Spain', 'Italy'], 'division': [1], 'year': [2021]}
dataloader = DataLoader(param_grid=param_grid, stats=FootballDataStats(), odds=FootballDataOdds())

# %%
# Preparing the data
# ------------------
#
# The data is downloaded onto your own machine by the `prepare` method, and never as a
# side effect of asking for it. Extracting from a dataloader that was not prepared raises
# rather than quietly downloading, so no data request can cost time or money by surprise.
# It is incremental, so re-running it only fetches what changed upstream.

dataloader.prepare()

# %%
# Extracting the training data
# ----------------------------
#
# We can get the available odds types using the `get_odds_types` class method.

dataloader.get_odds_types()

# %%
# We select the odds types to be the market average:

odds_type = 'market_average'

# %%
# We keep columns with non missing values for the training data by setting the
# `drop_na_thres` parameter equal to `1.0`.

drop_na_thres = 1.0

# %%
# We extract the training data:

X_train, Y_train, O_train = dataloader.extract_train_data(drop_na_thres=drop_na_thres, odds_type=odds_type)

# %%
# The input data:
X_train

# %%
# The output data:
Y_train

# %%
# The market average odds:
O_train

# %%
# Extracting the fixtures data
# ----------------------------
#
# The fixtures data are extracted with columns that match the columns of the
# training data:

X_fix, _, O_fix = dataloader.extract_fixtures_data()
