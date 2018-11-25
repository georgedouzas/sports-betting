"""
Includes classes and functions to test and select the optimal 
betting strategy on historical and current data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from os.path import join
from pickle import dump, load
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_score
from sklearn.utils import Parallel, delayed
from tqdm import tqdm

from .. import PATH
from ..utils import (
    check_classifier,
    fit_predict,
    generate_month_indices,
    yield_score
)
from .data import load_data


class BettingAgent:

    def backtest(self, classifier, param_grid, predicted_result, max_odds, test_season, random_state):
        """Apply backtesting to betting agent."""

        # Load and prepare data
        X, y, odds, matches = load_data('historical', max_odds, predicted_result=predicted_result)

        # Check classifier
        classifier = check_classifier(classifier, param_grid, random_state)

        # Define train and test indices of each month
        indices = generate_month_indices(matches, test_season)

        # Run cross-validation
        results = Parallel(n_jobs=-1)(delayed(fit_predict)(classifier,
                                                           X, y, odds, matches,
                                                           train_indices, test_indices)
                           for train_indices, test_indices in tqdm(indices, desc='Backtesting: '))

        # Combine results
        self.backtest_results_ = list(zip(*results))
        for ind, result in enumerate(self.backtest_results_):
            self.backtest_results_[ind] = reduce(lambda x, y: np.append(x, y, axis=0) if ind < 4 else x.append(y), result)
        self.backtest_results_[-1] = self.backtest_results_[-1].reset_index(drop=True)

    def calculate_backtest_results(self, n_bets_threshold):
        """Calculate the results of backtesting."""

        # Get backtesting results
        y_test, y_pred, y_pred_proba, odds, matches = self.backtest_results_

        # Calculate results
        n_bets = y_pred.size
        mean_odds = odds.mean()
        yld = yield_score(y_test, (y_pred, y_pred_proba, odds))
        profit = yld * n_bets 
        precision = precision_score(y_test, y_pred, labels=list(set(y_test).difference({'-'})), average='micro')
        results = [(n_bets, mean_odds, yld, profit, precision)]

        # Format results
        results = pd.DataFrame(results, columns=['Bets', 'Odds', 'Yield', 'Profit', 'Precision'])

        return results

    def fit_dump_classifier(self, classifier, param_grid, clf_name, predicted_result, random_state):
        """Fit and dump a classifier."""

        # Load modelling data
        X, y, odds, _ = load_data('historical', ['pinnacle', 'bet365', 'bwin'], predicted_result=predicted_result)

        # Modify input data
        X = np.hstack((X, odds))

        # Check classifier
        classifier = check_classifier(classifier, param_grid, random_state)

        # Fit classifier
        classifier.fit(X, y)

        # Dump classifier
        with open(join(PATH, '%s.pkl' % clf_name) , 'wb') as file:
            dump(classifier, file)
    
    def predict(self, clf_name):
        """Generate predictions using a fitted classifier."""

        # Load predictions data
        X, _, odds, matches  = load_data('predictions', ['pinnacle', 'bet365', 'bwin'])

        # Modify input data
        X = np.hstack((X, odds))

        # Load classifier
        with open(join(PATH, '%s.pkl' % clf_name) , 'rb') as file:
            classifier = load(file)

        # Generate predictions
        y_pred, _, odds = classifier.predict(X)
        y_pred = pd.DataFrame(y_pred, columns=['Prediction'])
        odds = pd.DataFrame(odds, columns=['Maximum Odds'])
        
        # Combine data
        predictions = pd.concat([matches, y_pred, odds], axis=1)
        predictions = predictions[predictions['Prediction'] != '-']

        return predictions
