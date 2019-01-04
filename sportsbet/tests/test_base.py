"""
Test the base module.
"""

import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.utils import check_random_state
import pytest

from sportsbet.base import MultiClassifier

X = check_random_state(0).uniform(size=(100, 2))
y = check_random_state(0).choice(['H', 'A', 'D'], size=100)
MULTI_CLASSIFIER = MultiClassifier([DummyClassifier(), DummyClassifier(), DummyClassifier()]).fit(X, y)


def test_multi_classifier_fit():
    """Test the fit method of multi classifier."""
    np.testing.assert_array_equal(MULTI_CLASSIFIER.classes_, np.array(['A', 'D', 'H']))
    assert hasattr(MULTI_CLASSIFIER, 'classifiers_')


def test_multi_classifier_predict():
    """Test the predict method of multi classifier."""
    np.testing.assert_array_equal(np.unique(MULTI_CLASSIFIER.predict(X)), np.array(['A', 'D', 'H']))
    

def test_multi_classifier_predict_norm_proba():
    """Test the predict normalized probabilities method of multi classifier."""
    np.testing.assert_array_equal(np.unique(MULTI_CLASSIFIER.predict_norm_proba(X).sum(axis=1)), np.repeat(1.0, len(X)))
