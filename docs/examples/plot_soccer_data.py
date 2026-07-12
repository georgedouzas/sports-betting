"""
Soccer data
===========

This example illustrates the usage of the [`SoccerDataLoader`][sportsbet.datasets.SoccerDataLoader].
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from sportsbet.datasets import SoccerDataLoader

# %%
# Getting the available parameters
# --------------------------------
#
# We can get the available parameters in order to select the training data
# to be extracted, using the `get_all_params` method. What is available depends on the
# data source, so it is a property of the configured dataloader rather than of its class.

SoccerDataLoader().get_all_params()

# %%
# We select to extract training data only for the year 2021 of the first
# division Spanish and Italian leagues.

param_grid = {'league': ['Spain', 'Italy'], 'division': [1], 'year': [2021]}
dataloader = SoccerDataLoader(param_grid=param_grid)

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
