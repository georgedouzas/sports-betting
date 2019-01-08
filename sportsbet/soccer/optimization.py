"""
Includes classes and functions to test and select the optimal 
betting strategy on historical and current data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from argparse import ArgumentParser
from os import listdir
from os.path import join
from pickle import dump, load
from pathlib import Path
from itertools import product, groupby
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.utils import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from tqdm import tqdm

from sportsbet.soccer.data import SoccerDataLoader

from .. import PATH

RESULTS_PATH = join(PATH, 'results')


class BettingAgent:
    """Class that is used for model evaluation, training and predictions 
    on upcoming matches."""

    def __init__(self, classifiers, betting_types):
        self.estimators = classifiers
        self.betting_types = betting_types
    
    def _extract_target(self, data, betting_type):
        """Extract target."""

        # Check betting type
        target_type_error_msg = 'Wrong target type.'
        if not isinstance(target_type, str):
            raise TypeError(target_type_error_msg)
        if target_type not in ('H', 'D', 'A', 'both_score', 'final_score') and 'over' not in target_type and 'under' not in target_type:
            raise ValueError(target_type_error_msg)
        self.target_type_ = target_type

        if self.target_type_ == 'H':
            y = (data['FTHG'] > data['FTAG']).astype(int)
        elif self.target_type_ == 'D':
            y = (data['FTHG'] == data['FTAG']).astype(int)
        elif self.target_type_ == 'A':
            y = (data['FTHG'] < data['FTAG']).astype(int)
        elif 'over' in self.target_type_:
            y = (data['FTHG'] + data['FTAG'] > float(self.target_type_[-2:])).astype(int)
        elif 'under' in self.target_type_:
            y = (data['FTHG'] + data['FTAG'] < float(self.target_type_[-2:])).astype(int)
        elif self.target_type_ == 'both_score':
            y = (data['FTHG'] * data['FTAG'] > 0).astype(int)
        elif self.target_type_ == 'final_score':
            y = data[['FTHG', 'FTAG']]
        return y

    def fit_predict(self, X, y, estimator, train_indices, test_indices, random_state):
        """Fit estimator and predict for a set of train and test indices."""
    
        # Set random state
        for param in estimator.get_params():
            if 'random_state' in param:
                estimator.set_params(**{param: random_state})

        # Filter test samples
        X_test = X[test_indices]

        # Fit classifier
        estimator.fit(X[train_indices], y[train_indices])
    
        # Get test set predictions
        y_pred = estimator.predict_proba(X_test)

        return y_pred
        
    def backtest(self, X, y, odds, test_season, random_states):
        """Apply backtesting to betting agent."""

        # Extract data
        X, y, odds, matches = training_data

        # Train and test indices
        self.test_indices_ = matches.loc[matches['Season'] == test_season].groupby('Month', sort=False).apply(lambda row: np.array(row.index)).values 
        self.train_indices_ = [np.arange(0, test_ind[0]) for test_ind in self.test_indices_]

        # Clone 

        # Run cross-validation
        cv_data = Parallel(n_jobs=-1)(delayed(self.fit_predict)(X, y, index, classifier, train_indices, test_indices, random_state) 
        for (train_indices, test_indices), (index, classifier), random_state in tqdm(list(product(zip(self.train_indices_, self.test_indices_), enumerate(self.classifiers_), random_states)), desc='Backtesting: '))
        self.cv_data = dict(cv_data)

        # Generate probabilities data
        indices_combinations = list(product(*[[index for index, _ in results] for _, results in groupby(enumerate(self.target_labels_), key=itemgetter(1))]))
        probabilities_data = {}
        for random_state in random_states:
            for indexH, indexA, indexD in indices_combinations:
                data = []
                for indices in self.test_indices_:
                    data.append(pd.concat([
                        self.cv_data[(random_state, indexH, indices[0])], 
                        self.cv_data[(random_state, indexA, indices[0])],
                        self.cv_data[(random_state, indexD, indices[0])]], axis=1)
                    )
                data = pd.concat(data).reset_index(drop=True)
                probabilities_data[(random_state, indexH, indexA, indexD)] = data
        
        # Generate months, odds and results data 
        months_data, odds_data, results_data = [], [], []
        results = pd.DataFrame(y, columns=['Result'])
        for indices in self.test_indices_:
            for data, df in zip([months_data, odds_data, results_data], [matches.Month, odds, results]):
                data.append(df.iloc[indices])

        # Backtest data
        self.backtesting_data_ = (
            pd.concat(months_data).reset_index(drop=True),
            pd.concat(odds_data).reset_index(drop=True),
            pd.concat(results_data).reset_index(drop=True),
            probabilities_data
        )

    def extract_backtesting_results(self, betting_results):
        """Extract backtesting results."""        

        # Unpack data
        months, odds, results, probabilities = self.backtesting_data_

        # Backtesting results placeholder
        self.backtesting_results_ = {}

        # Extract odds
        if self.betting_type == '12X':
            back_bets_profits = odds.applymap(lambda odd: [odd - 1.0, -1.0])
            lay_bets_profits = odds.applymap(lambda odd: [1.0, 1.0 - odd])
            profits = pd.concat([back_bets_profits, lay_bets_profits], axis=1)
            profits['-'] = [[0.0, 0.0]] * len(profits)
            profits_mapping = profits.apply(lambda profits: dict(zip(['H', 'A', 'D', '-H', '-A', '-D', '-'], profits.values.tolist())), axis=1).rename('Mapping')

        for key, probs in probabilities.items():

            # Extract probabilities
            probs = probs[betting_results]

            # Define boolean mask
            mask = (probs > 0.5).sum(axis=1).astype(bool)

            # Extract predictions
            predictions = probs.idxmax(axis=1).rename('Prediction')
            predictions.loc[~mask] = '-'

            # Extract profits
            profits = pd.concat([results, predictions, profits_mapping], axis=1)
            profits = profits.apply(lambda row: row['Mapping'][row['Prediction']][0 if row['Prediction'] == row['Result'] else 1], axis=1).rename('Profit')

            # Generate results
            self.backtesting_results_[key] = pd.DataFrame({
                'Bets percentage': [mask.sum() / len(mask)],
                'Precision': [precision_score(results, predictions, average='micro')],
                'Total yield': [profits.sum()],
                'Mean yield': [profits.mean()]
            })

    def fit_dump_classifiers(self, backtest_index):
        """Fit and dump classifiers."""

        # Create directory
        path = join(RESULTS_PATH, backtest_index)
        Path(path).mkdir(exist_ok=True)

        # Load training data
        X, y, _, _ = load_data('historical', ['pinnacle', 'bet365', 'bwin'])

        # Load backtesting results
        predicted_result, params_indices = pd.read_csv(join(RESULTS_PATH, 'backtesting.csv'), usecols=['Predicted results', 'Indices']).values[int(backtest_index)]
        
        # Extract predicted results and parameters indices
        predicted_results = list(predicted_result)
        params_indices = dict(zip(RESULTS_HAD, eval(params_indices)[0]))

        # Fit and dump classifiers
        for label in predicted_results:

            # Select classifier and parameters
            classifier = self.classifiers[label]
            params = self.param_grids[label][params_indices[label]]

            # Binarize labels
            y_bin = y.copy()
            y_bin[y_bin != label] = '-'

            # Fit classifier
            classifier.set_params(**params).fit(X, y_bin)
            
            # Dump classifier
            with open(join(path, '%s.pkl' % label) , 'wb') as file:
                dump(classifier, file)
        
    def predict(self, backtest_index):
        """Generate predictions using fitted classifiers."""

        # Load predictions data
        X, _, odds, matches  = load_data('predictions', ['pinnacle', 'bet365', 'bwin'])

        # Define file path and pickled classifiers
        path = join(RESULTS_PATH, backtest_index)
        pickled_classifiers = [file for file in listdir(path) if 'pkl' in file]

        # Iterate through results
        y_pred = pd.DataFrame()
        for pickled_classifier in pickled_classifiers:

            # Load classifier
            with open(join(path, pickled_classifier) , 'rb') as file:
                classifier = load(file)

            # Populate predictions
            y_pred = pd.concat([y_pred, pd.DataFrame(classifier.predict_proba(X)[:, -1], columns=[pickled_classifier[0]])], axis=1)
        
        # Extract predictions
        y_pred['Bet'] = y_pred.idxmax(axis=1)
        y_pred.loc[~(y_pred[ y_pred.columns[:-1]] > 0.5).sum(axis=1).astype(bool), 'Bet'] = '-'

        # Extract maximum odds
        y_pred['Maximum Odds'] = odds.apply(lambda odds: dict(zip(RESULTS_HAD, odds.values)), axis=1)
        y_pred['Maximum Odds'] = y_pred.apply(lambda row: row['Maximum Odds'][row['Bet']] if row['Bet'] != '-' else np.nan, axis=1)
        
        # Combine data
        predictions = pd.concat([matches.drop(columns=['Month', 'Day']), y_pred[['Bet', 'Maximum Odds']]], axis=1)
        predictions = predictions[predictions['Bet'] != '-'].reset_index(drop=True)

        return predictions


if __name__ == '__main__':



    from importlib import import_module

    from sklearn.dummy import DummyClassifier

    from ..soccer.optimization import BookmakerEstimator

    DEFAULT_ESTIMATORS = {
        '12X': {
            'random': [(DummyClassifier(), {}), (DummyClassifier(), {}), (DummyClassifier(), {})],
            'bookmaker': [(BookmakerEstimator('12X'), {}), (BookmakerEstimator('12X'), {}),(BookmakerEstimator('12X'), {})]
        },
        'OU2.5': {
            'random': [(DummyClassifier(), {})],
            'bookmaker': [(BookmakerEstimator('OU2.5'), {})]
        }
    }


    def import_estimators(clfs_name):

        # Import classifiers
        try:
            import config
            clfs_param_grids = getattr(config, clfs_name)
        except AttributeError:
            clfs_param_grids = DEFAULT_ESTIMATORS[clfs_name]
    
        # Extract classifiers and parameters grids
        classifiers, param_grids = zip(*clfs_param_grids)

        return classifiers, param_grids
    
    # Create parser
    parser = ArgumentParser('Models evaluation using backtesting.')
    
    # Add arguments
    parser.add_argument('--clfs-name', default='random', help='The name of classifiers to predict the results.')
    parser.add_argument('--max-odds', default=['pinnacle', 'bet365', 'bwin'], nargs='*', help='Maximum odds to use for evaluation.')
    parser.add_argument('--test-season', default=1819, type=int, help='Test season.')
    parser.add_argument('--random-states', default=[0, 1, 2], type=int, nargs='*', help='The random states of estimators.')
    parser.add_argument('--save-results', default=True, type=bool, help='Save backtesting results to csv.')

