"""
Defines helper functions and classes.
"""

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

##############
#Optimization#
##############

class OddsEstimator(BaseEstimator):
    """Estimator that appends the odds to the predictions."""

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y):

        # Get the input data
        X_input = X[:, :-3]

        # Fit the classifier
        self.classifier_ = clone(self.classifier).fit(X_input, y)

        return self

    def predict(self, X):

        # Get the input data and odds
        X_input, odds = X[:, :-3], X[:, -3:]

        # Get predictions
        y_pred = self.classifier_.predict(X_input)
        y_pred_proba = self.classifier_.predict_proba(X_input)

        # Select odds
        indices = [['H', 'A', 'D', '-'].index(result) for result in y_pred]
        odds = np.hstack((odds, np.zeros((y_pred.size, 1))))
        odds = odds[np.arange(len(odds)), indices]

        return y_pred, y_pred_proba, odds


def check_classifier(classifier, param_grid, random_state):
    """Set default values."""

    # Adjust parameters grid
    param_grid = {'classifier__%s' % param: value for param, value in param_grid.items()}

    # Check classifier
    classifier = GridSearchCV(OddsEstimator(classifier), param_grid, scoring=make_scorer(yield_score), cv=5, n_jobs=-1, iid=False)
    
    # Set random state
    for param in classifier.get_params():
        if 'random_state' in param:
            classifier.set_params(**{param: random_state})

    return classifier


def fit_predict(classifier, X, y, odds, matches, train_indices, test_indices):
    """Fit estimator and predict for a set of train and test indices."""

    # Modify input data
    X = np.hstack((X, odds))

    # Fit classifier
    classifier.fit(X[train_indices], y[train_indices])

    # Filter test samples
    X_test, y_test, matches = X[test_indices], y[test_indices], matches.iloc[test_indices]
    
    # Get test set predictions
    y_pred, y_pred_proba, odds = classifier.predict(X_test)
    
    # Filter placed bets
    mask = y_pred != '-'
    y_test, y_pred, y_pred_proba, odds, matches = y_test[mask], y_pred[mask], y_pred_proba[mask], odds[mask], matches[mask]
        
    return y_test, y_pred, y_pred_proba, odds, matches


def generate_month_indices(matches, test_season):
    """Generate train and test monthly indices for a test season."""
    
    # Test indices
    test_indices = matches.loc[matches['Season'] == test_season].groupby('Month', sort=False).apply(lambda row: np.array(row.index)).values
    
    # Train indices
    train_indices = [np.arange(0, test_indices[0][0])]
    train_indices += [np.arange(0, test_ind[-1] + 1) for test_ind in test_indices]
    
    # Combine indices
    indices = list(zip(train_indices, test_indices))
    
    return indices


def yield_score(y_true, y_pred_proba_odds):
    """Calculate yield for a set of bets."""

    # Get predictions and odds
    y_pred, _, odds = y_pred_proba_odds

    # Filter placed bets
    mask = y_pred != '-'
    y_true, y_pred, odds = y_true[mask], y_pred[mask], odds[mask]

    if odds.size == 0:
        return 0.0

    # Calculate wrong predictions
    mask = (y_true != y_pred)

    # Calculate yield
    yld = odds - 1
    yld[mask] = -1.0

    return yld.mean()

###########################
#Classifiers configuration#
###########################

def import_custom_classifiers(default_classifiers):
    """Try to import custom classifiers."""
    try:
        from extra_config import CLASSIFIERS
        CLASSIFIERS.update(default_classifiers)
    except ImportError:
        CLASSIFIERS = default_classifiers
    return CLASSIFIERS


DEFAULT_CLASSIFIERS = {
    'random': (DummyClassifier(random_state=0), {}),
    'baseline': (make_pipeline(SMOTE(), LogisticRegression(solver='lbfgs', max_iter=2000)), {})
}
CLASSIFIERS = import_custom_classifiers(DEFAULT_CLASSIFIERS)