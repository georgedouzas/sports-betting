"""Test the ClassifierBettor class."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from sportsbet.evaluation import ClassifierBettor
from tests.evaluation import O_fix, O_train, X_fix, X_train, Y_train


@pytest.mark.parametrize('classifier', [DummyRegressor(), None, 'classifier'])
def test_fit_raise_type_error_classifier(classifier):
    """Test raising an error on the wrong classifier type."""
    bettor = ClassifierBettor(classifier)
    msg = f'`ClassifierBettor` requires a classifier. Instead {type(classifier)} is given.'
    with pytest.raises(TypeError, match=msg):
        bettor.fit(X_train, Y_train)


def test_fit_check_classifier():
    """Test the cloned classifier is fitted and stored."""
    bettor = ClassifierBettor(classifier=DummyClassifier())
    with pytest.raises(NotFittedError):
        check_is_fitted(bettor)
    assert not hasattr(bettor, 'classifier_')
    bettor.fit(X_train, Y_train)
    check_is_fitted(bettor)
    check_is_fitted(bettor.classifier_)
    assert isinstance(bettor.classifier_, DummyClassifier)


def test_fit_predict_bet_consume_new_grammar():
    """Test fit/predict/bet consume the moment-aware X/Y/O without reshaping (CR-1, CR-4)."""
    bettor = ClassifierBettor(DummyClassifier(strategy='prior'))
    bettor.fit(X_train, Y_train, O_train)
    proba = bettor.predict_proba(X_train)
    assert proba.shape == (len(X_train), bettor.betting_markets_.size)
    assert bettor.predict(X_train).shape == (len(X_train), bettor.betting_markets_.size)
    B = bettor.bet(X_fix, O_fix)
    assert B.shape == (len(X_fix), bettor.betting_markets_.size)
    assert B.dtype == bool


def test_classes():
    """Test the fitted classes_ attribute."""
    bettor = ClassifierBettor(DummyClassifier())
    bettor.fit(X_train, Y_train)
    assert len(bettor.classes_) == bettor.betting_markets_.size
    assert all(np.array_equal(classes, [0, 1]) for classes in bettor.classes_)
