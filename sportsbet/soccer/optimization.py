"""
Includes classes and functions to test and select the optimal 
betting strategy on historical and current data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from os.path import join
from pickle import dump, load

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.utils import Parallel, delayed
from tqdm import tqdm

from .. import PATH
from ..utils import BetEstimator, yield_score, fit_predict
from .data import load_data
from ..config import DEFAULT_CLASSIFIERS


class BettingAgent:

    @staticmethod
    def _check_classifier(classifier, fit_params):
        """Use profit estimator and set default values."""

        # Check classifier and its fitting parameters
        classifier = BetEstimator(classifier) if classifier is not None else BetEstimator(DEFAULT_CLASSIFIERS['random'][0])
        fit_params = fit_params.copy() if fit_params is not None else DEFAULT_CLASSIFIERS['random'][1]

        return classifier, fit_params

    def backtest(self, classifier, fit_params, predicted_result, input_odds, max_odds, test_season):
        """Apply backtesting to betting agent."""

        # Load and prepare data
        X, y, max_odds_data, match_data = load_data('historical', input_odds, max_odds, predicted_result=predicted_result)

        # Prepare data
        X = np.hstack((X, max_odds_data))

        # Check classifier and its fitting parameters
        classifier, fit_params = self._check_classifier(classifier, fit_params)

        # Define train and test indices
        test_indices = match_data.loc[match_data['Season'] == test_season].groupby('Month', sort=False).apply(lambda row: np.array(row.index)).values
        train_indices = [np.arange(0, test_indices[0][0])]
        train_indices += [np.arange(0, test_ind[-1] + 1) for test_ind in test_indices]
        indices = list(zip(train_indices, test_indices))

        # Run cross-validation
        self.backtest_results_ = Parallel(n_jobs=-1)(delayed(fit_predict)(classifier, 
                                                                           X, y, 
                                                                           train_indices, test_indices, 
                                                                           **fit_params)
                                          for train_indices, test_indices in tqdm(indices, desc='Backtesting: '))

    def calculate_backtest_results(self, bet_factor=2, credit_exponent=3):
        """Calculate the results of backtesting."""

        # Initialize parameters
        statistics, precisions = [], []
        capital, bet_amount = 1.0, 1.0
        y_test_all, y_pred_all, y_pred_proba_all, odds_all = np.array([]), np.array([]), np.array([]), np.array([])
        

        for y_test, ((y_pred, y_pred_proba), odds) in self.backtest_results_:

            # Append results, predictions and odds
            y_test_all, y_pred_all = np.hstack((y_test, y_test_all)), np.hstack((y_pred, y_pred_all))
            y_pred_proba_all = np.vstack((y_pred_proba, y_pred_proba_all.reshape(-1, y_pred_proba.shape[1])))
            odds_all = np.vstack((odds, odds_all.reshape(-1, odds.shape[1])))

            # Calculate number of bets and matches
            mask = (y_pred != '-')
            n_bets = mask.sum()
            n_matches = y_pred.size

            # Calculate precision
            precision = precision_score(y_test[mask], y_pred[mask], average='micro') if n_bets > 0 else np.nan
            precisions.append(precision)

            # Calculate profit
            profit = bet_amount * yield_score(y_test, ((y_pred, y_pred_proba), odds))
            
            # Calculate capital
            capital += profit

            # Adjust bet amount
            if profit < 0.0:
                bet_amount *= bet_factor
            elif profit > 0.0:
                bet_amount = 1.0

            # Calculate credit
            max_credit = capital + bet_factor ** credit_exponent
            if bet_amount > max_credit:
                bet_amount = max_credit
                
            # Generate statistic
            statistic = (capital, profit, bet_amount, n_bets, n_matches, precision)

            # Append statistic
            statistics.append(statistic)

            if bet_amount == 0:
                break

        # Define statistics dataframe
        statistics = pd.DataFrame(statistics, columns=['Capital', 'Profit', 'Bet amount', 'Bets', 'Matches', 'Precision'])

        # Define attributes
        mean_precision = np.nanmean(precisions)
        profit_per_bet = yield_score(y_test_all, ((y_pred_all, y_pred_proba_all), odds_all))

        return statistics, mean_precision, profit_per_bet

    def fit_dump_classifier(self, classifier, fit_params, predicted_result, input_odds, max_odds, clf_name):
        """Fit and dump a classifier."""

        # Load modelling data
        X, y, *_ = load_data('historical', input_odds, max_odds)

        # Fit classifier
        classifier.fit(X, y, **fit_params)

        # Dump classifier
        with open(join(PATH, '%s.pkl' % clf_name) , 'wb') as file:
            dump(classifier, file)
    
    def predict(self, clf_name):
        """Generate predictions using a fitted classifier."""

        # Load predictions data
        X, y, max_odds_data, match_data  = load_data('predictions', input_odds, max_odds)

        # Load classifier
        with open(join(PATH, '%s.pkl' % clf_name) , 'rb') as file:
            classifier = load(file)

        # Generate predictions
        y_pred = pd.DataFrame(classifier.predict(X), columns=['Prediction'])
        minimum_odds = pd.DataFrame(1 / classifier.predict_proba(X).max(axis=1), columns=['Minimum Odds'])

        # Stack predictions
        predictions = pd.concat([match_data, max_odds_data, y_pred, minimum_odds], axis=1)

        return predictions