"""
Includes classes and functions to test and select the optimal 
betting strategy on historical and current data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from argparse import ArgumentParser
from ast import literal_eval
from itertools import product
from os.path import join
from sqlite3 import connect
from abc import abstractmethod
from importlib import import_module
from joblib import delayed, Parallel

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state, check_X_y, check_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sportsbet import SOCCER_PATH
from sportsbet.externals import TimeSeriesSplit
from sportsbet.soccer import TARGETS
from sportsbet.soccer.config import CONFIG

DB_CONNECTION = connect(join(SOCCER_PATH, 'soccer.db'))


def extract_multi_labels(score1, score2, targets):
    """Extract multi-labels matrix for multi-output classification."""
    
    # Check input data
    score1 = check_array(score1, dtype=int, ensure_2d=False)
    score2 = check_array(score2, dtype=int, ensure_2d=False)
    targets = check_array(targets, dtype=object, ensure_2d=False)

    # Generate multi-labels
    multi_labels = np.column_stack([TARGETS[target](score1, score2) for target in targets]).astype(int)
    
    return multi_labels


def extract_class_labels(score1, score2, odds, targets):
    """Extract class labels for multi-class classification."""
    
    # Check input data
    odds = check_array(odds)

    # Generate class labels
    multi_labels = extract_multi_labels(score1, score2, targets)
    indices = (multi_labels * odds).argmax(axis=1)
    class_labels = np.array([targets[ind] for ind in indices])
    class_labels[multi_labels.sum(axis=1) == 0] = '-'
    
    return class_labels


def calculate_yields(score1, score2, bets, odds, targets):
    """Calculate the yields."""

    # Check odds
    odds = check_array(odds)
    targets = check_array(targets, dtype=object, ensure_2d=False)

    # Generate yields
    bets = MultiLabelBinarizer(classes=['-'] + targets.tolist()).fit_transform([[bet] for bet in bets])[:, 1:]
    yields = ((extract_multi_labels(score1, score2, targets) * odds - 1.0) * bets).sum(axis=1)

    return yields


def extract_yields_stats(yields):
    """Extract coverage, mean and std of yields."""
    coverage_mask = (yields != 0.0)
    return coverage_mask.mean(), yields[coverage_mask].mean(), yields[coverage_mask].std()
    

def check_random_states(random_state, repetitions):
    """Create random states for experiments."""
    random_state = check_random_state(random_state)
    return [random_state.randint(0, 2 ** 32 - 1, dtype='uint32') for _ in range(repetitions)]


def fit_bet(bettor, params, risk_factors, random_state, X, scores, odds, train_indices, test_indices):
    """Parallel fit and bet"""

    # Unpack scores
    avg_score1, avg_score2, score1, score2 = scores

    # Set random state
    for param_name in bettor.get_params():
        if 'random_state' in param_name:
            bettor.set_params(**{param_name: random_state})

    # Fit better
    bettor.set_params(**params).fit(X[train_indices], avg_score1[train_indices], avg_score2[train_indices], odds[train_indices])

    # Generate data
    data = []
    for risk_factor in risk_factors:
        bets = bettor.bet(X[test_indices], risk_factor)
        yields = calculate_yields(score1[test_indices], score2[test_indices], bets, odds[test_indices], bettor.targets_)
        data.append((str(params), random_state, risk_factor, yields))
    data = pd.DataFrame(data, columns=['parameters', 'experiment', 'risk_factor', 'yields'])
    
    return data


def apply_backtesting(bettor, param_grid, risk_factors, X, scores, odds, cv, random_state, n_runs, n_jobs):
    """Apply backtesting to evaluate bettor."""
    
    # Check random states
    random_states = check_random_states(random_state, n_runs)

    # Check arrays
    X = check_array(X, dtype=None, force_all_finite=False)
    normalized_scores = []
    for score in scores:
        normalized_scores.append(check_array(score, dtype=None, ensure_2d=False))
    odds = check_array(odds, dtype=None)

    # Extract parameters
    parameters = ParameterGrid(param_grid)

    # Run backtesting
    data = Parallel(n_jobs=n_jobs)(delayed(fit_bet)(bettor, params, risk_factors, random_state, X, normalized_scores, odds, train_indices, test_indices) 
           for params, random_state, (train_indices, test_indices) in tqdm(list(product(parameters, random_states, cv.split(X))), desc='Tasks'))
    
    # Combine data
    data = pd.concat(data, ignore_index=True)
    data = data.groupby(['parameters', 'risk_factor', 'experiment']).apply(lambda df: np.concatenate(df.yields.values)).reset_index()
    data[['coverage', 'mean_yield', 'std_yield']] = pd.DataFrame(data[0].apply(lambda yields: extract_yields_stats(yields)).values.tolist())
    
    # Calculate results
    results = data.drop(columns=['experiment', 0]).groupby(['parameters', 'risk_factor']).mean().reset_index()
    results['std_mean_yield'] = data.groupby(['parameters', 'risk_factor'])['mean_yield'].std().values
    results = results.sort_values('mean_yield', ascending=False).reset_index(drop=True)

    return results


class BettorMixin:
    """Mixin class for all bettors."""

    _estimator_type = 'bettor'

    def __init__(self, targets):
        self.targets = targets

    @abstractmethod
    def predict(self, X):
        """Predict class labels."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Predict probabilities."""
        pass

    def fit(self):
        """Fit base bettor."""
        
        # Check targets
        if self.targets is None:
            self.targets_ = np.array(list(TARGETS.keys()))
        else:
            if not set(self.targets).issubset(TARGETS.keys()):
                raise ValueError(f'Targets should be any of {", ".join(self.targets)}')
            else:
                self.targets_ = check_array(self.targets, dtype=None, ensure_2d=False)
        
        return self
    
    def bet(self, X, risk_factor):
        """Generate bets."""

        # Check risk factor
        if not isinstance(risk_factor, float) or risk_factor > 1.0 or risk_factor < 0.0:
            raise ValueError('Risk factor should be a float in the [0.0, 1.0] interval.')
        
        # Generate bets
        bets = self.predict(X)

        # Apply no bets
        bets[self.predict_proba(X).max(axis=1) <= risk_factor] = '-'

        return bets


class Bettor(BaseEstimator, BettorMixin):
    """Bettor class that uses a multi-class classifier."""

    def __init__(self, classifier, targets=None):
        super(Bettor, self).__init__(targets)
        self.classifier = classifier

    def fit(self, X, score1, score2, odds):
        """Fit the classifier."""

        super(Bettor, self).fit()

        # Extract targets
        y = extract_class_labels(score1, score2, odds, self.targets_)

        # Fit classifier
        self.classifier_ = clone(self.classifier).fit(X, y)

        return self

    def predict(self, X):
        """Predict class labels."""
        return self.classifier_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.classifier_.predict_proba(X)


class MultiBettor(BaseEstimator, BettorMixin):
    """Bettor class that uses a multi-output classifier."""

    def __init__(self, multi_classifier, meta_classifier, test_size=0.5, random_state=None, targets=None):
        super(MultiBettor, self).__init__(targets)
        self.multi_classifier = multi_classifier
        self.meta_classifier = meta_classifier
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, score1, score2, odds):
        """Fit the multi-output classifier."""

        super(MultiBettor, self).fit()

        # Split data
        X_multi, X_meta, score1_multi, score1_meta, score2_multi, score2_meta, _, odds_meta = train_test_split(
            X, score1, score2, odds, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Extract targets
        Y_multi = extract_multi_labels(score1_multi, score2_multi, self.targets_)
        y_meta = extract_class_labels(score1_meta, score2_meta, odds_meta, self.targets_)

        # Fit multi-classifier
        self.multi_classifier_ = clone(self.multi_classifier).fit(X_multi, Y_multi)

        # Fit meta-classifier
        X_meta = np.column_stack([probs[:, 0] for probs in self.multi_classifier_.predict_proba(X_meta)])
        self.meta_classifier_ = clone(self.meta_classifier).fit(X_meta, y_meta)
        
        return self

    def predict(self, X):
        """Predict class labels."""
        X_meta = np.column_stack([probs[:, 0] for probs in self.multi_classifier_.predict_proba(X)])
        return self.meta_classifier_.predict(X_meta)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X_meta = np.column_stack([probs[:, 0] for probs in self.multi_classifier_.predict_proba(X)])
        return self.meta_classifier_.predict_proba(X_meta)


def extract_bettor():
    """Extract bettor from configuration file."""
    bettor_name = CONFIG['bettor']['type']
    bettor_params = CONFIG['bettor']['parameters']
    bettor_class = getattr(import_module(__name__), bettor_name)
    if bettor_name == 'Bettor':
        bettor = bettor_class(bettor_params['classifier'], bettor_params['targets'])
    elif bettor_name == 'MultiBettor':
        bettor = bettor_class(bettor_params['multi_classifier'], bettor_params['meta_classifier'], bettor_params['test_size'], targets=bettor_params['targets'])
    return bettor


def load_X(training=True):
    """Load input data."""
    tbl = "X" if training else "X_test"
    X_cols = [f'"{col}"' for col in pd.read_sql(f'PRAGMA table_info({tbl})', DB_CONNECTION)['name'] if col not in CONFIG['excluded_features']]
    X = pd.read_sql(f'select {", ".join(X_cols)} from {tbl}', DB_CONNECTION)
    return X


def load_odds(bettor, training=True):
    """Load odds data."""
    tbl = "odds" if training else "odds_test"
    odds_cols = [f'"{col}"' for col in (TARGETS.keys() if bettor.targets is None else bettor.targets)]
    odds = pd.read_sql(f'select {", ".join(odds_cols)} from {tbl}', DB_CONNECTION)
    return odds


def load_scores():
    """Load scores data."""
    y = pd.read_sql('select * from y', DB_CONNECTION)
    scores = []
    for ind in (1, 2):
        scores.append(np.round(y[f'{CONFIG["score_type"]}{ind}'], 0).astype(int))
    scores += [y['score1'], y['score2']]
    return scores


def backtest():
    """Command line function to backtest models.""" 

    # Command line parser
    parser = ArgumentParser('Models evaluation using backtesting.')
    parser.parse_args()
    
    # Extract backtesting parameters
    bettor = extract_bettor()
    cv = TimeSeriesSplit(CONFIG['n_splits'], CONFIG['min_train_size'])
    
    # Load data
    X, scores, odds = load_X(), load_scores(), load_odds(bettor)

    # Backtesting
    results = apply_backtesting(bettor, CONFIG['param_grid'], CONFIG['risk_factors'], X, scores, odds, cv, CONFIG['random_state'], CONFIG['n_runs'], CONFIG['n_jobs'])
    
    # Save backtesting results
    results.to_sql('backtesting_results', DB_CONNECTION, index=False, if_exists='replace')


def predict():
    """Command line function to predict new fixtures.""" 

    # Command line parser
    parser = ArgumentParser('Predict new fixtures.')
    parser.add_argument('--rank', default=0, type=int, help='The rank of the model to use for predictions.')
    args = parser.parse_args()

    # Extract backtesting parameters
    bettor = extract_bettor()

    # Load data
    X, X_test, scores, odds, odds_test = load_X(), load_X(training=False), load_scores(), load_odds(bettor), load_odds(bettor, training=False)
    parameters, risk_factor = pd.read_sql('select parameters, risk_factor from backtesting_results', DB_CONNECTION).values[args.rank]
    matches = pd.read_sql('select date, league, team1, team2 from X_test', DB_CONNECTION, parse_dates='date')
    
    # Fit bettor
    bettor = bettor.set_params(**literal_eval(parameters)).fit(X, scores[0], scores[1], odds)

    # Get predictions
    bets = bettor.bet(X_test, risk_factor)
    
    # Filter predictions
    mask = bets != '-'
    bets, odds_test, matches = bets[mask], odds_test[mask], matches[mask].reset_index(drop=True)
    odds_test = odds_test.values[range(len(bets)), [odds_test.columns.tolist().index(bet) for bet in bets]]

    # Format predictions
    predictions = pd.concat([matches, pd.DataFrame({'bet': bets, 'odds': odds_test})], axis=1)

    # Save predictions
    predictions.to_csv(join(SOCCER_PATH, 'predictions.csv'), index=False)
