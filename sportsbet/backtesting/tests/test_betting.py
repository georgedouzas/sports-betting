"""
Test the _betting module.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier
import pytest

from sportsbet.backtesting._betting import (
        extract_class_labels,
        MultiClassBettor,
        MultiOutputMetaBettor
)

X = [[0], [1], [2]] * 10
Y = pd.DataFrame([[1, 0, 0], [0, 1, 1], [0, 0, 0], [0, 1, 0], [0, 0, 1]] * 6, columns=['H', 'D', 'over_2.5'])
O = pd.DataFrame([[3.0, 1.5, 2.5], [4.0, 2.0, 3.0], [2.5, 2.5, 1.5], [1.5, 2.5, 1.5], [2.5, 1.5, 1.5]] * 6, columns=['H_odds', 'D_odds', 'over_2.5_odds'])
RND_SEED = 0
CLF = DummyClassifier(random_state=RND_SEED, strategy='prior')
MULTI_CLF = MultiOutputClassifier(CLF)
MCB = MultiClassBettor(classifier=CLF).fit(X, Y, O)
MOMB = MultiOutputMetaBettor(multi_classifier=MULTI_CLF, meta_classifier=CLF, random_state=RND_SEED).fit(X, Y, O)


def test_extract_class_labels():
    """Test the extraction of class labels."""
    expected_class_labels = ['H', 'over_2.5', '-', 'D', 'over_2.5'] * 6
    np.testing.assert_array_equal(extract_class_labels(Y, O, ['H', 'D', 'over_2.5']), expected_class_labels)


def test_multi_class_bettor_fit():
    """Test fit method of multi-class bettor.""" 
    np.testing.assert_array_equal(MCB.classifier_.classes_, np.array(['-', 'D', 'H', 'over_2.5']))


def test_multi_class_bettor_predict():
    """Test predict method of multi-class bettor."""
    assert set(MCB.predict(X)).issubset(['-', 'D', 'H', 'over_2.5'])


def test_multi_class_bettor_predict_proba():
    """Test predict probabilities method of multi-class bettor."""
    assert MCB.predict_proba(X).shape == (len(X), len(MCB.targets_) + 1)


def test_multi_output_meta_bettor_fit():
    """Test fit method of multi-output meta-bettor."""
    assert len(MOMB.multi_classifier_.estimators_) == len(MOMB.targets_)
    assert set(MOMB.meta_classifier_.classes_).issubset(['-', 'D', 'H', 'over_2.5'])


def test_multi_output_meta_bettor_predict():
    """Test predict method of multi-output meta-bettor."""
    assert set(MOMB.predict(X)).issubset(['-', 'D', 'H', 'over_2.5'])


def test_multi_output_meta_bettor_predict_proba():
    """Test predict probabilities method of multi-output meta-bettor."""
    probs = MOMB.predict_proba(X)
    assert probs.shape[0] == len(X)
    assert probs.shape[1] <= len(MOMB.targets_) + 1


def test_multi_output_meta_bettor_bet():
    """Test bet method of multi-output meta-bettor."""
    assert set(MOMB.bet(X, O)).issubset(['-', 'D', 'H', 'over_2.5'])


def test_multi_output_meta_bettor_risk_factor():
    """Test risk factor of bet method of multi-output meta-bettor."""
    bets_low_risk = MOMB.bet(X, O, risk_factor=0.0)
    bets_high_risk = MOMB.bet(X, O, risk_factor=1.0)
    assert (bets_low_risk == '-').sum() < (bets_high_risk == '-').sum()
