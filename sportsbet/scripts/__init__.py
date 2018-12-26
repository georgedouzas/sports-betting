"""
Defines constants and helper functions/classes.
"""

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
        clfs_param_grids = DEFAULT_CLASSIFIERS[clfs_name]
    
    # Extract classifiers and parameters grids
    classifiers, param_grids = zip(*clfs_param_grids)

    return classifiers, param_grids