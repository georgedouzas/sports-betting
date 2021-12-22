"""
=================
Soccer value bets
=================

This example illustrates how to estimate value bets for soccer fixtures by 
training a machine learning multi-output classifier.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import numpy as np
import pandas as pd
from sportsbet.datasets import SoccerDataLoader
from sklearn.neighbors import KNeighborsClassifier

###############################################################################
# Extracting the training data
###############################################################################

###############################################################################
# We extract the training data for the spanish league. We also remove any
# missing values and select the market average odds.

dataloader = SoccerDataLoader(param_grid={'league': ['Spain']})
X_train, Y_train, _ = dataloader.extract_train_data(
    drop_na_thres=1.0, odds_type='market_average'
)

###############################################################################
# The input data:
X_train

###############################################################################
# The targets:
Y_train

###############################################################################
# Training a multi-output classifier
###############################################################################

###############################################################################
# We train a :class:`~sklearn.neighbors.KNeighborsClassifier` using only numerical
# features from the input data. We also use the extracted targets.

num_features = [
    col
    for col in X_train.columns
    if X_train[col].dtype in (np.dtype(int), np.dtype(float))
]
clf = KNeighborsClassifier()
clf.fit(X_train[num_features], Y_train)

###############################################################################
# Extracting the fixtures data
###############################################################################

###############################################################################
# We extract the fixtures data. The columns by default match the columns of the
# training data.

X_fix, _, Odds_fix = dataloader.extract_fixtures_data()

###############################################################################
# The input data:
X_fix

###############################################################################
# The market average odds:
Odds_fix

###############################################################################
# Estimating the value bets
###############################################################################

###############################################################################
# We can estimate the value bets by using the fitted classifier.

Y_pred_prob = np.concatenate(
    [prob[:, 1].reshape(-1, 1) for prob in clf.predict_proba(X_fix[num_features])],
    axis=1,
)
X_fix_info = X_fix[['home_team', 'away_team']].reset_index()
value_bets = pd.concat([X_fix_info, Y_pred_prob * Odds_fix > 1], axis=1).set_index(
    'date'
)
value_bets.rename(
    columns={
        col: col.split('__')[1] for col in value_bets.columns if col.endswith('odds')
    }
)
