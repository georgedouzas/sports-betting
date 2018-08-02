
import pandas as pd
from . import RESULTS_MAPPING


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
