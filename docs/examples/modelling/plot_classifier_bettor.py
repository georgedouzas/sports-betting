"""
Classifier bettor
=================

This example illustrates how to use [`ClassfierBettor`][sportsbet.evaluation.ClassifierBettor]
and evaluate its performance on soccer historical data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from sportsbet.dataloaders import DataLoader
from sportsbet.evaluation import ClassifierBettor, backtest
from sportsbet.sources import FootballDataOdds, FootballDataStats

# %%
# Extracting the training data
# ----------------------------
#
# We extract the training data for the Spanish soccer league.
# We also remove columns that contain missing values and select the market
# maximum odds.

dataloader = DataLoader(
    param_grid={'league': ['Spain'], 'year': [2020, 2021, 2022]},
    stats=FootballDataStats(),
    odds=FootballDataOdds(),
)
X_train, Y_train, O_train = dataloader.extract_train_data(drop_na_thres=0.5, odds_type='market_maximum', download=True)

# %%
# The input data:
X_train

# %%
# The multi-output targets:
Y_train

# %%
# The odds data:
O_train

# %%
# In order to simplify the selected classifier, we keep only numerical features of the input data:
num_cols = X_train.columns[['float' in col_type.name for col_type in X_train.dtypes]]
X_train = X_train[num_cols]

# %%
# Classifier bettor
# -----------------
#
# We can use [`ClassifierBettor`][sportsbet.evaluation.ClassifierBettor] class to create
# a classifier-based bettor. We use a pipeline of an imputer to handle missing values
# and a KNN classifier.

clf = make_pipeline(SimpleImputer(), KNeighborsClassifier())
bettor = ClassifierBettor(clf)

# %%
# Any bettor is a classifier, therefore we can fit it on the training data.

_ = bettor.fit(X_train, Y_train)

# %%
# We can predict probabilities for the positive class.

bettor.predict_proba(X_train)

# %%
# We can also predict the class label.

bettor.predict(X_train)

# %%
# Finally, we can evaluate its cross-validation accuracy.

cross_val_score(bettor, X_train, Y_train, cv=TimeSeriesSplit(), scoring='accuracy').mean()

# %%
# Backtesting the bettor
# ----------------------
#
# We can backtest the bettor using the historical data.

backtesting_results = backtest(bettor, X_train, Y_train, O_train)

# %%
# Various backtesting statistics are calculated.

backtesting_results

# %%
# Estimating the value bets
# -------------------------
#
# We extract the fixtures data to estimate the value bets.

X_fix, _, Odds_fix = dataloader.extract_fixtures_data()
X_fix = X_fix[num_cols]
assert Odds_fix is not None

# %%
# We can estimate the value bets by using the fitted classifier.

_ = bettor.bet(X_fix, Odds_fix)

# %%
# A picture of it
# ---------------

yields = backtesting_results['Yield percentage per bet'].to_numpy()

fig, ax = plt.subplots()
ax.bar(range(1, len(yields) + 1), yields)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Classifier bettor: yield per bet, by fold')
ax.set_xlabel('fold')
ax.set_ylabel('yield %')
