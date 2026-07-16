"""
Classifier bettor
=================

This example illustrates ClassifierBettor, which wraps any scikit-learn
classifier and turns its probabilities into bets.
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
# The data
# --------
#
# Three seasons of the Spanish league. Dropping the columns that are more than half empty keeps the classifier simple,
# and the market maximum odds are the most generous price on offer for each outcome.

dataloader = DataLoader(
    param_grid={'league': ['Spain'], 'year': [2020, 2021, 2022]},
    stats=FootballDataStats(),
    odds=FootballDataOdds(),
)
X_train, Y_train, O_train = dataloader.extract_train_data(drop_na_thres=0.5, odds_type='market_maximum')

# %%
# The targets are the three outcomes, one boolean column each:
Y_train

# %%
# A KNN classifier only understands numbers, so keep the numerical columns and let an imputer fill the gaps.

num_cols = X_train.columns[['float' in col_type.name for col_type in X_train.dtypes]]
X_train = X_train[num_cols]

# %%
# The bettor
# ----------
#
# A bettor is a classifier. It has `fit`, `predict` and `predict_proba`, so the pipeline goes straight in, and the
# usual scikit-learn tooling works on it. Its cross-validated accuracy, for instance:

bettor = ClassifierBettor(make_pipeline(SimpleImputer(), KNeighborsClassifier()))
cross_val_score(bettor, X_train, Y_train, cv=TimeSeriesSplit(), scoring='accuracy').mean()

# %%
# Backtesting
# -----------
#
# Accuracy is not money. The backtest walks the seasons in order, bets only where the model sees value, and reports
# what the bankroll actually did.

bettor.fit(X_train, Y_train)
backtesting_results = backtest(bettor, X_train, Y_train, O_train)
backtesting_results

# %%
# The value bets for the upcoming matches come from the same fitted model:

X_fix, _, Odds_fix = dataloader.extract_fixtures_data()
X_fix = X_fix[num_cols]
assert Odds_fix is not None
_ = bettor.bet(X_fix, Odds_fix)

# %%
# Where the bets come from
# ------------------------
#
# A bet is placed when the model thinks an outcome is likelier than its price implies. Below, each match's home-win
# probability from the model is set against the probability the odds imply. Everything above the diagonal is where the
# model disagrees with the market in your favour, those are the bets, and whether they were wisdom or noise is exactly
# what the backtest above is there to tell you.

home_win_col = next(col for col in O_train.columns if '__home_win__' in col)
markets = bettor.betting_markets_.tolist()
model_prob = bettor.predict_proba(X_train)[:, markets.index('home_win')]
implied_prob = 1 / O_train[home_win_col].to_numpy()

fig, ax = plt.subplots()
ax.scatter(implied_prob, model_prob, s=10, alpha=0.4)
ax.plot([0, 1], [0, 1], color='black', linewidth=0.8)
ax.set_title('The model against the market, on the home win')
ax.set_xlabel('probability the price implies')
ax.set_ylabel('probability the model gives')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
