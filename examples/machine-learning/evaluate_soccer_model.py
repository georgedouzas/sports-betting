"""
================
Model evaluation
================

This example illustrates how to evaluate a model's performance 
on soccer historical data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import numpy as np
from sportsbet.datasets import SoccerDataLoader
from sklearn.neighbors import KNeighborsClassifier

###############################################################################
# Extracting the training data
###############################################################################

###############################################################################
# We extract the training data for the spanish league. We also remove any
# missing values and select the market maximum odds.

dataloader = SoccerDataLoader(param_grid={'league': ['Spain']})
X_train, Y_train, Odds_train = dataloader.extract_train_data(
    drop_na_thres=1.0, odds_type='market_maximum'
)

###############################################################################
# The input data:
X_train

###############################################################################
# The targets:
Y_train

###############################################################################
# Splitting the data
###############################################################################

###############################################################################
# We split the training data into training and testing data by keeping the
# first 80% of observations as training data, since the data are already
# sorted by date.

ind = int(len(X_train) * 0.80)
X_test, Y_test, Odds_test = X_train[ind:], Y_train[ind:], Odds_train[ind:]
X_train, Y_train = X_train[:ind], Y_train[:ind]

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
# Estimating the value bets
###############################################################################

###############################################################################
# We can estimate the value bets by using the fitted classifier.

Y_pred_prob = np.concatenate(
    [prob[:, 1].reshape(-1, 1) for prob in clf.predict_proba(X_test[num_features])],
    axis=1,
)
value_bets = Y_pred_prob * Odds_test > 1

###############################################################################
# We assume that we bet an amount of +1 in every value bet. Then we have the
# following mean profit per bet:

profit = np.nanmean((Y_test.values * Odds_test.values - 1) * value_bets.values)
profit
