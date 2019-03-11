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
from pickle import dump, load
from sqlite3 import connect
from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.model_selection import BaseCrossValidator, ParameterGrid, train_test_split
from sklearn.multiclass import check_classification_targets
from sklearn.multioutput import MultiOutputClassifier, _fit_estimator
from sklearn.utils import Parallel, check_random_state, check_X_y, delayed, check_array
from sklearn.utils.fixes import parallel_helper
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from sportsbet import SOCCER_PATH
from sportsbet.soccer import TARGETS, OFFSETS, SCORES_MAPPING, FEATURES
from sportsbet.soccer.config import PORTOFOLIOS

DB_CONNECTION = connect(join(SOCCER_PATH, 'soccer.db'))


def calculate_yields(scores, bets, odds):
    """Calculate the yields."""

    # Check scores
    scores = check_array(scores, dtype=int)
    
    # Convert bets to indicator matrix
    mlb = MultiLabelBinarizer()
    y_bets = mlb.fit_transform([[bet] for bet in bets])

    # Extract starting index and targets
    start_ind = 1 if '-' in mlb.classes_ else 0
    targets = mlb.classes_[start_ind:]

    # Calculate yields
    masks = np.concatenate([SCORES_MAPPING[target](scores[:, 0], scores[:, 1], 0.0).reshape(-1, 1) for target in targets], axis=1)
    yields = (y_bets[:, start_ind:] * masks * odds[targets].values - 1.0).max(axis=1)
    yields[bets == '-'] = 0.0

    return yields


def _extract_yields_stats(yields):
    """Extract coverage, mean and std of yields."""
    coverage_mask = (yields != 0.0)
    return coverage_mask.mean(), yields[coverage_mask].mean(), yields[coverage_mask].std()
    

def check_random_states(random_state, repetitions):
    """Create random states for experiments."""
    random_state = check_random_state(random_state)
    return [random_state.randint(0, 2 ** 32 - 1, dtype='uint32') for _ in range(repetitions)]


def _fit_bet(better, params, risk_factors, random_state, X, y, odds, train_indices, test_indices):
    """Parallel fit and bet"""

    # Set random state
    for param_name in better.get_params():
        if 'random_state' in param_name:
            better.set_params(**{param_name: random_state})

    # Fit better
    better.set_params(**params).fit(X.iloc[train_indices], y.iloc[train_indices], odds.iloc[train_indices])

    # Generate data
    data = []
    for risk_factor in risk_factors:
        bets = better.bet(X.iloc[test_indices], risk_factor)
        yields = calculate_yields(y.iloc[test_indices], bets, odds.iloc[test_indices])
        data.append((str(params), random_state, risk_factor, yields))
    data = pd.DataFrame(data, columns=['parameters', 'experiment', 'risk_factor', 'yields'])
    
    return data


def apply_backtesting(better, param_grid, risk_factors, X, y, odds, cv, random_state, n_runs):
    """Apply backtesting to evaluate better."""
    
    # Check random states
    random_states = check_random_states(random_state, n_runs)
    
    # Extract test indices
    test_indices = np.concatenate([indices for _, indices in cv.split()])

    # Extract parameters
    parameters = ParameterGrid(param_grid)

    # Run backtesting
    data = Parallel(n_jobs=-1)(delayed(_fit_bet)(better, params, risk_factors, random_state, X, y, odds, train_indices, test_indices) 
           for params, random_state, (train_indices, test_indices) in tqdm(list(product(parameters, random_states, cv.split())), desc='Tasks'))
    
    # Combine data
    data = pd.concat(data, ignore_index=True)
    data = data.groupby(['parameters', 'risk_factor', 'experiment']).apply(lambda df: np.concatenate(df.yields.values)).reset_index()
    data[['coverage', 'mean_yield', 'std_yield']] = pd.DataFrame(data[0].apply(lambda yields: _extract_yields_stats(yields)).values.tolist())
    
    # Calculate results
    results = data.drop(columns=['experiment', 0]).groupby(['parameters', 'risk_factor']).mean().reset_index()
    results['std_mean_yield'] = data.groupby(['parameters', 'risk_factor'])['mean_yield'].std().values
    results = mean_results.sort_values('mean_yield', ascending=False).reset_index(drop=True)

    return results
    

class SeasonSplit(BaseCrossValidator):
    """Split time-series data based on a test season."""

    def __init__(self, n_splits, seasons, test_season):
        self.n_splits = n_splits
        self.seasons = seasons
        self.test_season = test_season

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        start_index, end_index = (self.seasons != self.test_season).sum(), len(self.seasons)
        step = (end_index - start_index) // self.n_splits
        breakpoints = list(range(start_index, end_index, step)) + [end_index]
        for start, end in zip(breakpoints[:-1], breakpoints[1:]):
            yield np.arange(0, start), np.arange(start, end)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


class MultiOutputClassifiers(_BaseComposition, MultiOutputClassifier):

    def __init__(self, classifiers, features_container, n_jobs=None):
        self.classifiers = classifiers
        self.features_container = features_container
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit a separate classifier for each output variable."""

        for _, clf in self.classifiers:
            if not hasattr(clf, 'fit'):
                raise ValueError('Every base classifier should implement a fit method.')

        data = [check_X_y(X[features], y, multi_output=True, accept_sparse=True) for features in self.features_container]

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError('Output y must have at least two dimensions for multi-output classification but has only one.')

        if sample_weight is not None and any([not has_fit_parameter(clf, 'sample_weight') for _, clf in self.classifiers]):
            raise ValueError('One of base classifiers does not support sample weights.')

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_estimator)(clf, X, y[:, i], sample_weight) 
                                                        for i, (_, clf), (X, y) in zip(range(y.shape[1]), self.classifiers, data))
        
        return self

    def predict(self, X):
        """Predict multi-output target."""
        
        check_is_fitted(self, 'estimators_')

        for _, clf in self.classifiers:
            if not hasattr(clf, 'predict'):
                raise ValueError('Every base classifier should implement a predict method')

        data = [check_array(X[features], accept_sparse=True) for features in self.features_container]

        y_pred = Parallel(n_jobs=self.n_jobs)(delayed(parallel_helper)(e, 'predict', X) for e, X in zip(self.estimators_, data))

        return np.asarray(y_pred).T

    def predict_proba(self, X):
        """Predict multi-output probabilities."""
        
        check_is_fitted(self, 'estimators_')

        for _, clf in self.classifiers:
            if not hasattr(clf, 'predict_proba'):
                raise ValueError('Every base should implement predict_proba method')

        data = [check_array(X[features], accept_sparse=True) for features in self.features_container]

        y_pred = Parallel(n_jobs=self.n_jobs)(delayed(parallel_helper)(e, 'predict', X) for e, X in zip(self.estimators_, data))

        return y_pred


class BaseBetter(BaseEstimator):
    """Base class for all betters."""

    def __init__(self, targets, offsets, features):
        self.targets = targets
        self.offsets = offsets
        self.features = features

    @abstractmethod
    def predict(self, X):
        """Predict class labels."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Predict probabilities."""
        pass  

    def _check_targets_offsets(self):
        """Check the targets."""
    
        # Check parameter targets is a subset of predifined targets
        if not set(self.targets).issubset(TARGETS):
            raise ValueError('Selected targets are not supported.')
    
        # Sort targets
        indices = np.argsort(self.targets)
        
        self.targets_ = np.array(self.targets)[indices].astype(object)
        self.offsets_ = np.array(self.offsets)[indices]

    def _check_features(self, X):
        """Check the input matrix features."""
        if self.features != 'all' and not set(self.features).issubset(X.columns):
            raise ValueError('Selected features are not a subset of all features.')
        self.features_ = FEATURES if self.features == 'all' else self.features
    
    def _extract_multi_labels(self, y):
        """Extract multi-labels matrix for multi-output classification."""
    
        # Check targets
        y = check_array(y, dtype=int)
    
        # Generate boolean masks
        masks = np.concatenate([SCORES_MAPPING[target](y[:, 0], y[:, 1], offset).reshape(-1, 1) for target, offset in zip(self.targets_, self.offsets_)], axis=1)
    
        # Binarize targets
        y = MultiLabelBinarizer().fit_transform([self.targets_[mask] for mask in masks])
    
        return y

    def _extract_class_labels(self, y, odds):
        """Extract class labels for multi-class classification."""
    
        # Extract multi-label matrix
        y = self._extract_multi_labels(y)
    
        # Select label with highest yiled 
        y = np.array(self.targets_)[(y * odds[self.targets_.tolist()].values).argmax(axis=1)].astype(object)
    
        # Identify no bets
        mask = (y.reshape(-1, 1).sum(axis=1) == 0)
        y[mask] = '-'
    
        return y
    
    def bet(self, X, risk_factor):
        """Generate bets."""

        # Check risk factor
        if not isinstance(risk_factor, float) or risk_factor >= 1.0 or risk_factor <= 0.0:
            raise ValueError('Risk factor should be a float in the (0.0, 1.0) interval.')
        
        # Generate bets
        bets = self.predict(X)

        # Apply no bets
        bets[self.predict_proba(X).max(axis=1) <= risk_factor] = '-'

        return bets


class Better(BaseBetter):
    """Better class that uses a multi-class classifier."""

    def __init__(self, classifier, targets=TARGETS, offsets=OFFSETS, features='all'):
        super(Better, self).__init__(targets, offsets, features)
        self.classifier = classifier

    def fit(self, X, y, odds):
        """Fit the classifier."""

        # Checks
        self._check_targets_offsets()
        self._check_features(X)

        # Extract targets
        y = self._extract_class_labels(y, odds)

        # Fit classifier
        self.classifier_ = clone(self.classifier).fit(X[self.features_], y)
        return self

    def predict(self, X):
        """Predict class labels."""
        return self.classifier_.predict(X[self.features_])

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.classifier_.predict_proba(X[self.features_])


class MultiBetter(BaseBetter):
    """Better class that uses a multi-output classifier."""

    def __init__(self, multi_classifier, targets=TARGETS, offsets=OFFSETS, features='all'):
        super(MultiBetter, self).__init__(targets, offsets)
        self.multi_classifier = multi_classifier
    
    def fit(self, X, y):
        """Fit the multi-output classifier."""
        
        # Checks
        self._check_targets_offsets()
        self._check_features(X)

        # Extract targets
        y = self._extract_multi_labels(y)

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
    (better_type, scores_type, risk_factors), param_grid =  PORTOFOLIOS[args.portofolio]
    if better_type == 'better':
        better = Better(param_grid['classifier'])
    scores_cols = ['score1', 'score2'] if scores_type == 'real' else ['avg_score1', 'avg_score2']

    # Create cross-validator
    cv = SeasonSplit(args.n_splits, X['season'].values, args.test_season)

    # Backtesting
    results = apply_backtesting(better, param_grid, risk_factors, X, y[scores_cols], odds, cv, args.random_state, args.n_runs)
    results['id'] = args.portofolio, better_type, scores_type 

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
