"""Test the ClassifierBettor class."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation import ClassifierBettor

X_train, Y_train, O_train = DummySoccerDataLoader().extract_train_data(odds_type='williamhill')


@pytest.mark.parametrize('classifier', [DummyRegressor(), None, 'classifier'])
def test_fit_raise_type_error_classifier(classifier):
    """Test raising an error on the wrong classifier type."""
    bettor = ClassifierBettor(classifier)
    with pytest.raises(
        TypeError,
        match=f'`ClassifierBettor` requires a classifier. Instead {type(classifier)} is given.',
    ):
        bettor.fit(X_train, Y_train)


def test_fit_check_classifier():
    """Test the cloned classifier."""
    bettor = ClassifierBettor(classifier=DummyClassifier())
    with pytest.raises(NotFittedError):
        check_is_fitted(bettor)
    assert not hasattr(bettor, 'classifier_')
    bettor.fit(X_train, Y_train)
    check_is_fitted(bettor)
    check_is_fitted(bettor.classifier_)
    assert isinstance(bettor.classifier_, DummyClassifier)


def test_bet():
    """Test the bet method."""
    bettor = ClassifierBettor(classifier=DummyClassifier(strategy='constant', constant=[False, True, True]))
    bettor.fit(X_train, Y_train)
    expected_value_bets = np.array(
        [
            [False, True, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, True],
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ],
    )
    assert np.array_equal(bettor.bet(X_train, O_train), expected_value_bets)
