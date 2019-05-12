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
from tqdm import tqdm

from sportsbet import SOCCER_PATH
from sportsbet.externals import TimeSeriesSplit
from sportsbet.soccer import TARGETS
from sportsbet.soccer.config import PORTOFOLIOS

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


def fit_bet(bettor, params, risk_factors, random_state, X, score1, score2, odds, train_indices, test_indices):
    """Parallel fit and bet"""

    # Set random state
    for param_name in bettor.get_params():
        if 'random_state' in param_name:
            bettor.set_params(**{param_name: random_state})

    # Fit better
    bettor.set_params(**params).fit(X[train_indices], score1[train_indices], score2[train_indices], odds[train_indices])

    # Generate data
    data = []
    for risk_factor in risk_factors:
        bets = bettor.bet(X[test_indices], risk_factor)
        yields = calculate_yields(score1[test_indices], score2[test_indices], bets, odds[test_indices], bettor.targets_)
        data.append((str(params), random_state, risk_factor, yields))
    data = pd.DataFrame(data, columns=['parameters', 'experiment', 'risk_factor', 'yields'])
    
    return data


def apply_backtesting(bettor, param_grid, risk_factors, X, score1, score2, odds, cv, random_state, n_runs):
    """Apply backtesting to evaluate bettor."""
    
    # Check random states
    random_states = check_random_states(random_state, n_runs)

    # Extract parameters
    parameters = ParameterGrid(param_grid)

    # Run backtesting
    data = Parallel(n_jobs=-1)(delayed(fit_bet)(bettor, params, risk_factors, random_state, X, score1, score2, odds, train_indices, test_indices) 
           for params, random_state, (train_indices, test_indices) in tqdm(list(product(parameters, random_states, cv.split(X))), desc='Tasks'))
    
    # Combine data
    data = pd.concat(data, ignore_index=True)
    data = data.groupby(['parameters', 'risk_factor', 'experiment']).apply(lambda df: np.concatenate(df.yields.values)).reset_index()
    data[['coverage', 'mean_yield', 'std_yield']] = pd.DataFrame(data[0].apply(lambda yields: extract_yields_stats(yields)).values.tolist())
    
    # Calculate results
    results = data.drop(columns=['experiment', 0]).groupby(['parameters', 'risk_factor']).mean().reset_index()
    results['std_mean_yield'] = data.groupby(['parameters', 'risk_factor'])['mean_yield'].std().values
    results = results.sort_values('mean_yield', ascending=False).reset_index(drop=True)

    print(results)

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
            self.targets_ = np.array(TARGETS.keys())
        else:
            if not set(self.targets).issubset(TARGETS.keys()):
                raise ValueError(f'Targets should be any of {", ".join(self.targets)}')
            else:
                self.targets_ = np.array(self.targets)
        
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

    def __init__(self, base_classifier, multi_classifier, targets=None):
        super(MultiBettor, self).__init__(targets)
        self.base_classifier = base_classifier
        self.multi_classifier = multi_classifier
    
    def fit(self, X, score1, score2, odds):
        """Fit the multi-output classifier."""

        super(MultiBettor, self).fit()

        # Extract targets
        y = extract_multi_labels(score1, score2, self.targets_)

        # Fit multi-classifier
        self.multi_classifier_ = clone(self.multi_classifier).fit(X, y)
        
        return self

    def predict(self, X):
        """Predict class probabilities."""
        y_pred = np.concatenate([probs[:, 1] for probs in self.multi_classifier_.predict_proba(X)], axis=1)
        return self.targets_[y_pred.argmax(axis=1)]


def backtest():
    """Command line function to backtest models.""" 

    # Create parser
    parser = ArgumentParser('Models evaluation using backtesting.')
        
    # Add arguments
    parser.add_argument('portofolio', help='The name of portofolio to evaluate.')
    parser.add_argument('--test-season', default='1819', type=str, help='The test season.')
    parser.add_argument('--n-splits', default=5, type=int, help='Number of cross-validation splits.')
    parser.add_argument('--random-state', default=0, type=int, help='The random seed.')
    parser.add_argument('--n-runs', default=5, type=int, help='Number of evaluation runs.')

    # Parse arguments
    args = parser.parse_args()
    
    # Load data
    X = pd.read_sql('select * from X', DB_CONNECTION)
    y = pd.read_sql('select * from y', DB_CONNECTION)
    odds = pd.read_sql('select * from odds', DB_CONNECTION)

    # Unpack configuration
    param_grid =  PORTOFOLIOS[args.portofolio]
    better_class, scores_type, risk_factors = param_grid.pop('better_class'), param_grid.pop('scores_type'), param_grid.pop('risk_factors')
    better = getattr(import_module(__name__), better_class)(param_grid['classifier'])
    scores_cols = ['score1', 'score2'] if scores_type == 'real' else ['avg_score1', 'avg_score2']

    # Create cross-validator
    cv = SeasonSplit(args.n_splits, X['season'].values, args.test_season)

    # Backtesting
    results = apply_backtesting(better, param_grid, risk_factors, X, y[scores_cols], odds, cv, args.random_state, args.n_runs)
    results['id'] = [(args.portofolio, better_class, scores_type)] * len(results)

    # Save backtesting results
    try:
        backtesting_results = pd.read_sql('select * from backtesting_results', DB_CONNECTION)
        backtesting_results = backtesting_results[backtesting_results['portofolio'] != args.portofolio]
    except pd.io.sql.DatabaseError:
        backtesting_results = pd.DataFrame([])
    backtesting_results = backtesting_results.append(results, ignore_index=True).sort_values('mean_yield', ascending=False)
    backtesting_results.to_sql('backtesting_results', DB_CONNECTION, index=False, if_exists='replace')


def predict():
    """Command line function to predict new fixtures.""" 

    # Create parser
    parser = ArgumentParser('Predict new fixtures.')
        
    # Add arguments
    parser.add_argument('portofolio', help='The name of portofolio to evaluate.')
    parser.add_argument('--rank', default=0, type=int, help='The rank of the model to use for predictions.')

    # Parse arguments
    args = parser.parse_args()

    # Load data
    X = pd.read_sql('select * from X', DB_CONNECTION)
    y = pd.read_sql('select * from y', DB_CONNECTION)
    odds = pd.read_sql('select * from odds', DB_CONNECTION)
    X_test = pd.read_sql('select * from X_test', DB_CONNECTION)
    odds_test = pd.read_sql('select * from odds_test', DB_CONNECTION)
    parameters, calibration = pd.read_sql('select parameters, calibration from backtesting_results where portofolio == "{}"'.format(args.portofolio), DB_CONNECTION).values[args.rank]

    # Stack input and odds data
    target_types = [target_type for target_type, *_, in PORTOFOLIOS[args.portofolio]]
    X = pd.concat([X, odds[target_types]], axis=1)
    X_test = pd.concat([X_test, odds_test[target_types]], axis=1)

    # Fit betting classifiers
    betting_classifiers = [(target_type, features, BettingClassifier(clf.set_params(**params))) for params, (target_type, clf, *_, features) in zip(literal_eval(parameters), PORTOFOLIOS[args.portofolio])]
    mbclf = _MetaBettingClassifier(betting_classifiers).fit(X, y)
    
    # Get predictions
    y_pred = mbclf.predict_proba(X_test)[0]
    bets_indices = y_pred.argmax(axis=1)
    mask = (y_pred > literal_eval(calibration))[range(len(y_pred)), bets_indices]

    # Format predictions
    predictions = X_test.loc[:, ['date', 'league', 'team1', 'team2']]
    predictions['odd'] = odds_test[target_types].values[range(len(y_pred)), bets_indices]
    predictions['bet'] = np.array(target_types)[bets_indices]
    predictions = predictions[mask]

    # Save predictions
    predictions.to_csv(join(SOCCER_PATH, 'predictions.csv'), index=False)
