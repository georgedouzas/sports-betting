"""
Test the optimization module.
"""

import requests
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.utils import check_random_state
import pytest

X = check_random_state(0).uniform(size=(100, 2))
y_results = check_random_state(0).choice(['H', 'A', 'D'], size=100)
y_bin = check_random_state(0).choice([0, 1], size=100)
CLASSIFIER = DummyClassifier(random_state=0)
CLASSIFIERS = [DummyClassifier(random_state=0), DummyClassifier(random_state=1), DummyClassifier(random_state=2)]


@pytest.mark.parametrize('y,label', [
    (y_results, 'H'), 
    (y_results, 'A'), 
    (y_results, 'D'),
    (y_results, 'O'), 
    (y_results, 'U')
])
def test_fit_binary_classifier(y, label):
    """Test the fit of a classifier for a binarized target."""
    classifier = fit_binary_classifier(X, y, clone(CLASSIFIER), label)
    classes = np.unique(classifier.predict(X))
    if np.array_equal(y, y_results):
        np.testing.assert_array_equal(classes, np.array(['-', label]))
    elif np.array_equal(y, y_results):
        np.testing.assert_array_equal(classes, np.array(['O']))

def test_base_bet_classifier_fit():
    """Test the fit of base bet classifier."""
    base_bet_classifier = BaseBetClassifier(CLASSIFIERS)
    base_bet_classifier.fit(X, y_mo)
    np.testing.assert_array_equal(base_bet_classifier.classes_, np.array(['A', 'D', 'H']))
    assert hasattr(base_bet_classifier, 'classifiers_')


    