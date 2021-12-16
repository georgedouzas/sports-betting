"""
===========================
FiveThirtyEight soccer data
===========================

This example illustrates the usage of FiveThirtyEight soccer dataloader.

"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import pandas as pd
from sportsbet.datasets import FTESoccerDataLoader

###############################################################################
# Getting the available parameters
###############################################################################

###############################################################################
# We can get the available parameters in order to select the training data
# to be extracted, using the :meth:`get_all_params` class method.

params = FTESoccerDataLoader.get_all_params()

###############################################################################
# The available parameters can be presented as a DataFrame.

params_df = pd.DataFrame(params).sort_values(
    ['league', 'year', 'division'], ignore_index=True
)
params_df

###############################################################################
# We select to extract training data only for the year 2021 of all the
# divisions of English league.

param_grid = {'league': ['England'], 'year': [2021]}

###############################################################################
# Getting the available odds types
###############################################################################

###############################################################################
# We can get the available odds types in order to match the output of the
# training data, using the :meth:`get_odds_types` class method.

FTESoccerDataLoader.get_odds_types()

###############################################################################
# Therefore no odds data are available.

###############################################################################
# Extracting the training data
###############################################################################

###############################################################################
# We extract the training data using the default values for the parameters
# `odds_type` and `drop_na_thres`.

dataloader = FTESoccerDataLoader(param_grid=param_grid)
X_train, Y_train, _ = dataloader.extract_train_data()

###############################################################################
# The input data:
X_train

###############################################################################
# The targets:
Y_train

###############################################################################
# Extracting the fixtures data
###############################################################################

###############################################################################
# We extract the fixtures data with columns that match the columns of the
# training data. On the other hand, the fixtures data are not affected by
# the `param_grid` selection.

X_fix, *_ = dataloader.extract_fixtures_data()

###############################################################################
# The input data:
X_fix
