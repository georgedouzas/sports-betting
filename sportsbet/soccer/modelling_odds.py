from os.path import join
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from category_encoders import OrdinalEncoder
from sportsbet.soccer import (
    RESULTS_MAPPING,
    BETTING_INTERVAL,
    TEST_SEASON,
    ODDS_FEATURES,
    GOALS_ODDS_FEATURES,
    ASIAN_ODDS_FEATURES,
    CLOSING_ODDS_FEATURES
)

BASELINE_NUM_FEATURES = ODDS_FEATURES + GOALS_ODDS_FEATURES + ASIAN_ODDS_FEATURES + CLOSING_ODDS_FEATURES


def select_odd(clf, X, agent):
    """Select odds based on predictions."""
    y_pred = pd.Series(clf.predict(X), name='ypred', dtype=int)
    odds = X.loc[:, [agent + result for result in RESULTS_MAPPING.keys()]]
    return pd.concat([odds, y_pred], axis=1).apply(lambda row: row[int(row[3])], axis=1)


def calculate_profit(y_true, y_pred, odd, select):
    """Calculate mean profit."""
    profit = (y_true == y_pred) * (odd - 1)
    profit[profit == 0] = -1
    profit = profit[select]
    return 100 * profit


class BaselineEstimator(BaseEstimator, ClassifierMixin):
    """Predict the result based on the odds given by betting agents."""

    def __init__(self, agent='B365', starting_index=None):
        self.agent = agent
        self.starting_index = starting_index

    def fit(self, X, y, sample_weight=None):
        """No actual fitting occurs."""
        starting_index = self.starting_index
        if starting_index is None:
            starting_index = X.shape[1] - len(BASELINE_NUM_FEATURES) + ODDS_FEATURES.index(self.agent + 'H')
        self.features_indices_ = list(range(starting_index, starting_index + 3))
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted class label per sample.
        """
        predictions = X[:, self.features_indices_].argmin(axis=1)
        return predictions


training_odds_data = pd.read_csv(join('data', 'training_odds_data.csv'))
weeks = pd.cut(training_odds_data.TimeIndex, range(0, max(training_odds_data.TimeIndex) + BETTING_INTERVAL, BETTING_INTERVAL), False)
weeks_test_season = weeks[training_odds_data.Season == TEST_SEASON].cat.remove_unused_categories()
profits = pd.Series()
for week_test in weeks_test_season.cat.categories:

    training = training_odds_data[weeks < week_test].reset_index(drop=True)
    X_train, y_train = training.iloc[:, :-1], training.iloc[:, -1]

    testing = training_odds_data[weeks == week_test].reset_index(drop=True)
    X_test, y_test = testing.iloc[:, :-1], testing.iloc[:, -1]

    agent = 'IW'

    clf = XGBClassifier()

    pipeline = make_pipeline(OrdinalEncoder(), Imputer(), clf)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    profit = calculate_profit(y_test, y_pred, select_odd(pipeline, X_test, agent), pipeline.predict_proba(X_test).max(axis=1) > 0.6)
    profits = profits.append(pd.Series(profit))
    print(profits.mean())