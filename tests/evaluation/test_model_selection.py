"""Test the backtest function and BettorGridSearchCV class."""

import re
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, TimeSeriesSplit

from sportsbet.evaluation import BettorGridSearchCV, ClassifierBettor, OddsComparisonBettor, backtest
from tests.evaluation import O_train, TestBettor, X_train, Y_train

_CV_MSG = re.escape('Parameter `cv` should be a TimeSeriesSplit cross-validator object.')


@pytest.mark.parametrize('cv', [3, 'cv'])
def test_backtest_params_cv_raise_error(cv):
    """Test raising an error on the wrong backtest cv param."""
    bettor = TestBettor()
    with pytest.raises(TypeError, match=_CV_MSG):
        backtest(bettor, X_train, Y_train, O_train, cv=cv)


def test_backtest_params_non_df_raise_value_error():
    """Test raising an error on the wrong backtest input data."""
    bettor = TestBettor()
    with pytest.raises(TypeError, match=re.escape('Input data `X` should be pandas dataframe with a date index.')):
        backtest(bettor, cast(pd.DataFrame, X_train.to_numpy()), Y_train, O_train)


def test_backtest_params_no_date_raise_value_error():
    """Test raising an error on input data without a date index."""
    bettor = TestBettor()
    with pytest.raises(TypeError, match=re.escape('Input data `X` should be pandas dataframe with a date index.')):
        backtest(bettor, pd.DataFrame(X_train.values), Y_train, O_train)


@pytest.mark.parametrize('bettor', [ClassifierBettor(DummyClassifier(strategy='prior')), OddsComparisonBettor()])
def test_backtest_returns_per_period_results(bettor):
    """Test backtest returns per-period performance on moment-aware data (T016)."""
    cv = TimeSeriesSplit(3)
    results = backtest(bettor, X_train, Y_train, O_train, cv=cv)
    assert isinstance(results, pd.DataFrame)
    assert len(results) == cv.get_n_splits()
    assert 'Number of bets' in results.columns
    assert any(col.startswith('Yield percentage per bet') for col in results.columns)


def test_bgscv_fit_raise_type_error_no_odds():
    """Test raising an error when no odds are provided to the default scoring."""
    bgscv = BettorGridSearchCV(OddsComparisonBettor(), {})
    with pytest.raises(TypeError, match='The default scoring method requires the odds data `O`'):
        bgscv.fit(X_train, Y_train)


@pytest.mark.parametrize('bettor', [DummyClassifier(), None, 'bettor'])
def test_bgscv_fit_raise_type_error_wrong_estimator(bettor):
    """Test raising an error on the wrong estimator type."""
    bgscv = BettorGridSearchCV(bettor, {})
    with pytest.raises(
        TypeError,
        match=f'`BettorGridSearchCV` requires a bettor as estimator. Instead {type(bettor)} is given.',
    ):
        bgscv.fit(X_train, Y_train, O_train)


@pytest.mark.parametrize('cv', [KFold(), None, 'cv'])
def test_bgscv_fit_raise_type_error_wrong_cross_validator(cv):
    """Test raising an error on the wrong cross validator type."""
    bgscv = BettorGridSearchCV(OddsComparisonBettor(), {}, cv=cv)
    with pytest.raises(TypeError, match=_CV_MSG):
        bgscv.fit(X_train, Y_train, O_train)


@pytest.mark.parametrize('n_splits', [2, 3])
def test_bgscv_fit(n_splits):
    """Test the fit of bettor grid search cross validation over moment-aware data (T016)."""
    bgscv = BettorGridSearchCV(
        estimator=TestBettor(),
        param_grid={'betting_markets': [None, ['draw'], ['home_win', 'away_win']]},
        cv=TimeSeriesSplit(n_splits),
    )
    bgscv.fit(X_train, Y_train, O_train)
    assert bgscv.n_splits_ == n_splits
    assert np.array_equal(bgscv.betting_markets_, bgscv.best_estimator_.betting_markets_)
    assert np.array_equal(bgscv.feature_names_out_, bgscv.best_estimator_.feature_names_out_)
    assert len(bgscv.cv_results_['params']) == len(bgscv.param_grid['betting_markets'])


def test_bgscv_scores_odds_bettor():
    """Test the grid search computes finite cross-validated scores for an odds-appending bettor."""
    bgscv = BettorGridSearchCV(
        estimator=OddsComparisonBettor(),
        param_grid={'alpha': [0.02, 0.05]},
        cv=TimeSeriesSplit(2),
    )
    bgscv.fit(X_train, Y_train, O_train)
    assert np.isfinite(bgscv.best_score_)
