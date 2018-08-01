from os.path import join
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from category_encoders import OrdinalEncoder
from . import RESULTS_MAPPING
from .classification import OddsEstimator

TEST_SEASON = '17-18'
BETTING_INTERVAL = 7


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

    clf = OddsEstimator()

    pipeline = make_pipeline(OrdinalEncoder(), Imputer(), clf)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    profit = calculate_profit(y_test, y_pred, select_odd(pipeline, X_test, agent), pipeline.predict_proba(X_test).max(axis=1) > 0.6)
    profits = profits.append(pd.Series(profit))
    print(profits.mean())