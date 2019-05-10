"""
Configuration file for models evaluation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from itertools import product

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

PORTOFOLIOS = {
    'all_markets': {
        'targets': ['H', 'A', 'D', 'over_2.5', 'under_2.5'],
        'scores_type': 'real',
        'offsets': [0.0, 0.0, 0.0, 0.0, 0.0],
        'better_class': 'Better', 
        'risk_factors': [0.3, 0.4, 0.5],
        'classifier': [make_pipeline(SimpleImputer(), SMOTE(), LogisticRegression(max_iter=20000, solver='lbfgs', multi_class='auto'))],
        'classifier__smote__k_neighbors': [2, 3],
        'classifier__logisticregression__C': [5e4]
    }
}
