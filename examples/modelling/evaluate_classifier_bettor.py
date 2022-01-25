"""
=================
Classifier bettor
=================

This example illustrates how to use a classifier-based bettor
and evaluate its performance on soccer historical data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import numpy as np
from sportsbet.datasets import SoccerDataLoader
from sportsbet.evaluation import ClassifierBettor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

###############################################################################
# Extracting the training data
###############################################################################

###############################################################################
# We extract the training data for the Spanish league.
# We also remove columns that contain missing values and select the market
# maximum odds.

dataloader = SoccerDataLoader(param_grid={'league': ['Spain']})
X_train, Y_train, O_train = dataloader.extract_train_data(
    drop_na_thres=1.0, odds_type='market_maximum'
)

###############################################################################
# The input data:
X_train

###############################################################################
# The multi-output targets:
Y_train

###############################################################################
# The odds:
O_train

###############################################################################
# The simplify the training process we keep only the numerical features of the
# input data:

num_features = [
    col
    for col in X_train.columns
    if X_train[col].dtype in (np.dtype(int), np.dtype(float))
]
X_train = X_train[num_features]

###############################################################################
# Classifier bettor
###############################################################################

###############################################################################
# We can use :class:`~sportsbet.evaluation.ClassifierBettor` class to create
# a classifier-based bettor. A :class:`~sklearn.neighbors.KNeighborsClassifier`
# is selected.

bettor = ClassifierBettor(KNeighborsClassifier())

###############################################################################
# Any bettor is a classifier, therefore we can fit it on the training data.

bettor.fit(X_train, Y_train)

###############################################################################
# We can predict probabilities for the positive class.

bettor.predict_proba(X_train)

###############################################################################
# We can predict the class label.

bettor.predict(X_train)

###############################################################################
# Finally, we can evaluate its cross-validation accuracy.

cross_val_score(bettor, X_train, Y_train, cv=TimeSeriesSplit()).mean()

###############################################################################
# Backtesting the bettor
###############################################################################

###############################################################################
# We can backtest the bettor using the historical data.

bettor.backtest(X_train, Y_train, O_train)

###############################################################################
# Various backtesting statistics are calculated.

bettor.backtest_results_

###############################################################################
# We can also plot the portfolio value for any testing period from the above
# backtesting results.

testing_period = 2
bettor.backtest_plot_value_(testing_period)

###############################################################################
# Estimating the value bets
###############################################################################

###############################################################################
# We extract the fixtures data to estimate the value bets.

X_fix, _, Odds_fix = dataloader.extract_fixtures_data()
X_fix = X_fix[num_features]

###############################################################################
# We can estimate the value bets by using the fitted classifier.

bettor.bet(X_fix, Odds_fix)
