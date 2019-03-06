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
    'match_odds': [
        (
            'H', 
            make_pipeline(SimpleImputer(), SMOTE(), LogisticRegression(max_iter=5000, solver='lbfgs')), 
            {'smote__k_neighbors': [3], 'smote__sampling_strategy': [0.95], 'logisticregression__C': [5e4]},
            np.arange(0.65, 0.75, 0.01),
            ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'quality', 'importance', 'rating']
        ),
        (
            'A', 
            make_pipeline(SimpleImputer(), SMOTE(), LogisticRegression(max_iter=5000, solver='lbfgs')), 
            {'smote__k_neighbors': [2], 'smote__sampling_strategy': [0.9], 'logisticregression__C': [5e4]},
            np.arange(0.75, 0.85, 0.01),
            ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'quality', 'importance', 'rating']
        ),
        (
            'D', 
            make_pipeline(SimpleImputer(), SMOTE(), LogisticRegression(max_iter=5000, solver='lbfgs')), 
            {'smote__k_neighbors': [2], 'smote__sampling_strategy': [1.0], 'logisticregression__C': [5e4]},
            np.arange(0.6, 0.75, 0.01),
            ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'quality', 'importance', 'rating']
        )
    ],
    'H': [
        (
            'H', 
            make_pipeline(SimpleImputer(), SMOTE(), LogisticRegression(max_iter=5000, solver='lbfgs')), 
            {'smote__k_neighbors': [3, 4], 'smote__sampling_strategy': [0.9, 0.95], 'logisticregression__C': [1e4, 5e4]},
            np.arange(0.4, 0.9, 0.01),
            ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'quality', 'importance', 'rating']
        )
    ],
    'A': [
        (
            'A', 
            make_pipeline(SimpleImputer(), SMOTE(), LogisticRegression(max_iter=5000, solver='lbfgs')), 
            {'smote__k_neighbors': [2, 3], 'smote__sampling_strategy': [0.6, 0.7, 0.8, 0.9], 'logisticregression__C': [1e4, 5e4]},
            np.arange(0.4, 0.9, 0.01),
            ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'quality', 'importance', 'rating']
        )
    ],
    'D': [
        (
            'D', 
            make_pipeline(SimpleImputer(), SMOTE(), LogisticRegression(max_iter=5000, solver='lbfgs')), 
            {'smote__k_neighbors': [2, 3], 'smote__sampling_strategy': [1.0], 'logisticregression__C': [1e4, 5e4]},
            np.arange(0.4, 0.9, 0.01),
            ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'quality', 'importance', 'rating']
        )
    ],
    'over_2.5': [
        (
            'over_2.5', 
            make_pipeline(SimpleImputer(), SMOTE(), LogisticRegression(max_iter=5000, solver='lbfgs')), 
            {'smote__k_neighbors': [2, 3], 'smote__sampling_strategy': [1.0], 'logisticregression__C': [1e4, 5e4]},
            np.arange(0.4, 0.9, 0.01),
            ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'quality', 'importance', 'rating', 'sum_proj_score']
        )
    ],
    'under_2.5': [
        (
            'under_2.5', 
            make_pipeline(SimpleImputer(), SMOTE(), LogisticRegression(max_iter=5000, solver='lbfgs')), 
            {'smote__k_neighbors': [2, 3], 'smote__sampling_strategy': [1.0], 'logisticregression__C': [1e4, 5e4]},
            np.arange(0.4, 0.9, 0.01),
            ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'quality', 'importance', 'rating', 'sum_proj_score']
        )
    ],
    'H+D': [
        (
            'H+D', 
            make_pipeline(SimpleImputer(), SMOTE(), LogisticRegression(max_iter=5000, solver='lbfgs')), 
            {'smote__k_neighbors': [3, 4], 'smote__sampling_strategy': [0.9, 0.95], 'logisticregression__C': [1e4, 5e4]},
            np.arange(0.4, 0.95, 0.01),
            ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'quality', 'importance', 'rating']
        )
    ],
    'A+D': [
        (
            'A+D', 
            make_pipeline(SimpleImputer(), SMOTE(), LogisticRegression(max_iter=5000, solver='lbfgs')), 
            {'smote__k_neighbors': [2, 3], 'smote__sampling_strategy': [1.0], 'logisticregression__C': [1e4, 5e4]},
            np.arange(0.4, 0.95, 0.01),
            ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'quality', 'importance', 'rating']
        )
    ],
    'H+A': [
        (
            'H+A', 
            make_pipeline(SimpleImputer(), SMOTE(), LogisticRegression(max_iter=5000, solver='lbfgs')), 
            {'smote__k_neighbors': [2, 3], 'smote__sampling_strategy': [1.0], 'logisticregression__C': [1e4, 5e4]},
            np.arange(0.4, 0.95, 0.01),
            ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'quality', 'importance', 'rating']
        )
    ]
}

