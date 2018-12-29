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
from sklearn.utils.metaestimators import _BaseComposition
from tqdm import tqdm

from .. import PATH

RESULTS_PATH = join(PATH, 'results')


@staticmethod
def fit_binary_classifier(X, y, classifier, label):
    """Binarize label and fit a classifier."""

    # Binarize labels
    y_bin = y.copy()
    y_bin[y_bin != label] = '-%s' % label
        
    # Fit classifier
    classifier.fit(X, y_bin)
        
    return classifier


class BaseBetClassifier(_BaseComposition, ClassifierMixin):

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):
        """Parallel fit of classifiers."""
        self.classes_ = np.unique(y)
        self.classifiers_ = Parallel(n_jobs=-1)(delayed(fit_binary_classifier)(X, y, clone(classifier), label) 
                                                for classifier, label in zip(self.classifiers, self.classes_))

    def _generate_probs(self, X):
        """Generate probabilities for each classifier."""
        probs = np.concatenate([clf.predict_proba(X)[:, 1] for clf in self.classifiers_])
        return probs

    def predict(self, X):
        """Predict the results."""
        classes = np.array([self.classes_[ind] for ind in self._generate_probs(X).argmax(axis=1)])
        return classes
    
    def predict_proba(self, X):
        """Predict the probabilities of results."""
        probs = self._generate_probs(X)
        normalized_probs = probs / probs.sum(axis=1)[:, None]
        return normalized_probs


class MatchOddsClassifier(BaseBetClassifier):

    RESULTS = ['A', 'D', 'H']

    def __init__(self, classifiers):
        self.classifiers = classifiers
    
    def _bet(self, X, odds):
        
        # Columns mapping
        columns_mapping = {result: '-%s' % result for result in self.RESULTS}

        # Calculate back and lay probabilities
        back_probs = self._generate_probs(X)
        lay_probs = (1 - back_probs).rename(columns=columns_mapping)
        probs = pd.concat([back_probs, lay_probs], axis=1)

        # Calculate back and lay values
        odds = odds[self.RESULTS]
        back_values = back_probs * odds.values - 1
        lay_values = -back_values.rename(columns=columns_mapping)
        values = pd.concat([back_values, lay_values], axis=1)

        return probs, values

    @staticmethod
    def _value_bets(values, betting_results):

        # Get probabilities
        values = values[betting_results]

        # Extract predictions
        predictions = values.idxmax(axis=1).rename('Prediction')
        predictions.loc[~(values > 0.0).sum(axis=1).astype(bool)] = '-'
        
        return predictions


class BookmakerEstimator(BaseEstimator, ClassifierMixin):
    """Estimator that uses the average odds to generate predictions."""

    def __init__(self, betting_type):
        self.betting_type = betting_type

    def fit(self, X, y):

        # Define predicted labels
        self.labels_ = [label if label in np.unique(y) else '-' for label in RESULTS[self.betting_type]]
        
        return self

    def predict(self, X):
        
        # Generate indices from minimum odds
        min_odds_indices = np.argmin(X[:, 0:3], axis=1)

        # Get predictions
        y_pred = np.array(self.labels_)[min_odds_indices]

        return y_pred

    def predict_proba(self, X):

        # Generate predicted probabilities
        y_pred_proba = 1 / X[:, 0:3]
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1)[:, None]

        # Sort predicted labels
        y_pred_proba = y_pred_proba[:, np.argsort(self.labels_)]
        
        # Add probabilities
        if len(np.unique(self.labels_)) < 3:
            y_pred_proba = np.apply_along_axis(arr=y_pred_proba, func1d=lambda probs: [probs[0:2].sum(), probs[2]], axis=1)

        return y_pred_proba


class BettingAgent:
    """Class that is used for model evaluation, training and predictions 
    on upcoming matches."""

    def __init__(self, classifiers, param_grids, betting_type):
        self.classifiers = classifiers
        self.param_grids = param_grids
        self.betting_type = betting_type

    def fit_predict(self, X, y, index, classifier, train_indices, test_indices, random_state):
        """Fit estimator and predict for a set of train and test indices."""
    
        # Set random state
        for param in classifier.get_params():
            if 'random_state' in param:
                classifier.set_params(**{param: random_state})

        # Generate target label
        label = self.target_labels_[index]

        # Binarize labels
        y_bin = y.copy()
        y_bin[y_bin != label] = '-%s' % label

        # Filter test samples
        X_test = X[test_indices]

        # Fit classifier
        classifier.fit(X[train_indices], y_bin[train_indices])
    
        # Get test set predictions
        probabilities = pd.DataFrame(classifier.predict_proba(X_test), columns=['-%s' % label, label])

        return (random_state, index,  test_indices[0]), probabilities

    def initialize_parameters(self, matches, test_season, random_states):

        # Train and test indices
        self.test_indices_ = matches.loc[matches['Season'] == test_season].groupby('Month', sort=False).apply(lambda row: np.array(row.index)).values 
        self.train_indices_ = [np.arange(0, test_ind[0]) for test_ind in self.test_indices_]

        # Generate list of classifiers
        self.classifiers_ = [clone(classifier).set_params(**params) for classifier, param_grid in zip(self.classifiers, self.param_grids) for params in ParameterGrid(param_grid)]

        # Generate indices for param_grids
        self.param_grids_indices_ = [(ind, clf.get_params()) for ind, clf in enumerate(self.classifiers_)]

        # Generate target labels
        self.target_labels_ = [result for result, param_grid in zip(RESULTS[self.betting_type], self.param_grids) for _ in ParameterGrid(param_grid)]
        
    def backtest(self, max_odds, test_season, random_states):
        """Apply backtesting to betting agent."""

        # Load and prepare data
        X, y, odds, matches = load_data('historical', max_odds)

        # Initialize parameters
        self.initialize_parameters(matches, test_season, random_states)

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
      
    return parser.parse_args()
