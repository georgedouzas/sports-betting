"""
Searching over betting markets
==============================

This example illustrates [`BettorGridSearchCV`][sportsbet.evaluation.BettorGridSearchCV], and the point that is easy to
miss: **the markets to bet on are a hyperparameter**, so they can be searched over like any other.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from sportsbet.dataloaders import DataLoader
from sportsbet.evaluation import BettorGridSearchCV, ClassifierBettor, backtest
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

# %%
# Extracting the data
# -------------------

dataloader = DataLoader(
    param_grid={'league': ['England', 'Spain']},
    stats=SampleSoccerStats(),
    odds=SampleSoccerOdds(),
)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum')

# %%
# The model
# ---------

classifier = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']),
        remainder='passthrough',
    ),
    SimpleImputer(),
    MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=7, class_weight='balanced')),
)

# %%
# The markets are a hyperparameter
# --------------------------------
#
# Which market to bet is a choice, exactly as the regularisation of the classifier is a choice, and there is no
# reason to make one by hand and search the other. Put both in the grid and let the search decide.

bettor = BettorGridSearchCV(
    estimator=ClassifierBettor(classifier, init_cash=10000.0, stake=50.0),
    param_grid={
        'classifier__multioutputclassifier__estimator__C': [0.1, 1.0],
        'betting_markets': [['home_win'], ['draw'], ['away_win'], ['home_win', 'draw', 'away_win']],
    },
    cv=TimeSeriesSplit(2),
)
_ = bettor.fit(X_train, Y_train, O_train)

# %%
# What it chose:

bettor.best_params_

# %%
# Backtesting the search
# ----------------------
#
# The search is re-run inside every fold, so the markets are chosen on the training part of the fold and judged on the
# part it has not seen. Choosing them once, on everything, would be choosing them on the answers.

results = backtest(bettor, X_train, Y_train, O_train, cv=TimeSeriesSplit(3))
results

# %%
# A picture of it
# ---------------

yields = results['Yield percentage per bet'].to_numpy()

fig, ax = plt.subplots()
ax.bar(range(1, len(yields) + 1), yields)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Searched bettor: yield per bet, by fold')
ax.set_xlabel('fold')
ax.set_ylabel('yield %')
