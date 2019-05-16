"""
Configuration file for models evaluation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sportsbet.externals import MultiOutputClassifiers

CONFIG = {
    'bettor': {
        'type': 'MultiBettor',
        'parameters': {
            'multi_classifier': make_pipeline(SimpleImputer(), MultiOutputClassifiers(
                [
                    ('over_2.5', LogisticRegression(max_iter=20000, solver='lbfgs', multi_class='auto'))
                ]
                )
            ),
            'meta_classifier': make_pipeline(SMOTE(), GradientBoostingClassifier()),
            'test_size': 0.5,
            'targets': ['over_2.5']
        }
    },
    'param_grid': {
        'multi_classifier__multioutputclassifiers__over_2.5__C': [1e5, 1e4],
        'meta_classifier__smote__k_neighbors': [2, 3],
        'meta_classifier__gradientboostingclassifier__max_depth': [3, 4]
    },
    'risk_factors': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'score_type': 'avg_score',
    'n_splits': 5,
    'min_train_size': 0.5,
    'random_state': 0,
    'n_runs': 3,
    'n_jobs': -1,
    'excluded_features': ['season', 'date', 'league', 'team1', 'team2']
}
