"""Test the backtest function and GridSearchCV class."""

from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, TimeSeriesSplit

from sportsbet.evaluation import BettorGridSearchCV, ClassifierBettor, OddsComparisonBettor, backtest
from tests.evaluation import O_train, TestBettor, X_train, Y_train


@pytest.mark.parametrize('cv', [3, 'cv'])
def test_backtest_params_cv_raise_error(cv):
    """Test raising an error on the wrong backtest cv param."""
    bettor = TestBettor()
    with pytest.raises(
        TypeError,
        match='Parameter `cv` should be a TimeSeriesSplit cross-validator object.',
    ):
        backtest(bettor, X_train, Y_train, O_train, cv=cv)


def test_backtest_params_non_df_raise_value_error():
    """Test raising an error on the wrong backtest input data."""
    bettor = TestBettor()
    with pytest.raises(TypeError, match='Input data `X` should be pandas dataframe with a date index.'):
        backtest(bettor, cast(pd.DataFrame, X_train.to_numpy()), Y_train, O_train)


def test_backtest_params_no_date_raise_value_error():
    """Test raising an error on the wrong backtest input data."""
    bettor = TestBettor()
    with pytest.raises(TypeError, match='Input data `X` should be pandas dataframe with a date index.'):
        backtest(bettor, pd.DataFrame(X_train.values), Y_train, O_train)


def test_backtest():
    """Test the outcome of backtest."""
    n_splits = 2
    cv = TimeSeriesSplit(n_splits)
    clf = DummyClassifier(strategy='constant', constant=[True, False, True])

    # Backtesting results
    bettor = ClassifierBettor(clf)
    results = backtest(bettor, X_train, Y_train, O_train, cv)

    # Assertions
    assert results.shape[0] == n_splits
    expected_results = pd.DataFrame(
        [
            {
                'Training start': datetime.strptime('1997-04-05', '%Y-%m-%d').astimezone().date(),
                'Training end': datetime.strptime('1999-04-03', '%Y-%m-%d').astimezone().date(),
                'Testing start': datetime.strptime('2000-04-03', '%Y-%m-%d').astimezone().date(),
                'Testing end': datetime.strptime('2001-04-06', '%Y-%m-%d').astimezone().date(),
                'Number of betting days': 2,
                'Number of bets': 2,
                'Yield percentage per bet': 25.0,
                'ROI percentage': 0.2,
                'Final cash': 10025.0,
                'Number of bets (home_win__full_time_goals)': 0,
                'Number of bets (draw__full_time_goals)': 0,
                'Number of bets (away_win__full_time_goals)': 2,
                'Yield percentage per bet (home_win__full_time_goals)': 0.0,
                'Yield percentage per bet (draw__full_time_goals)': 0.0,
                'Yield percentage per bet (away_win__full_time_goals)': 25.0,
            },
            {
                'Training start': datetime.strptime('1997-04-05', '%Y-%m-%d').astimezone().date(),
                'Training end': datetime.strptime('2001-04-06', '%Y-%m-%d').astimezone().date(),
                'Testing start': datetime.strptime('2017-03-17', '%Y-%m-%d').astimezone().date(),
                'Testing end': datetime.strptime('2019-03-17', '%Y-%m-%d').astimezone().date(),
                'Number of betting days': 2,
                'Number of bets': 1,
                'Yield percentage per bet': 250.0,
                'ROI percentage': 1.2,
                'Final cash': 10125.0,
                'Number of bets (home_win__full_time_goals)': 1,
                'Number of bets (draw__full_time_goals)': 0,
                'Number of bets (away_win__full_time_goals)': 0,
                'Yield percentage per bet (home_win__full_time_goals)': 250.0,
                'Yield percentage per bet (draw__full_time_goals)': 0.0,
                'Yield percentage per bet (away_win__full_time_goals)': 0.0,
            },
        ],
    ).set_index(['Training start', 'Training end', 'Testing start', 'Testing end'])
    pd.testing.assert_frame_equal(results, expected_results, check_dtype=False)


def test_bgscv_fit_raise_type_error_no_odds():
    """Test raising an error when no odds are provided."""
    bgscv = BettorGridSearchCV(OddsComparisonBettor(), {})
    with pytest.raises(
        TypeError,
        match='The default scoring method requires the odds data `O`',
    ):
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
    with pytest.raises(
        TypeError,
        match='Parameter `cv` should be a TimeSeriesSplit cross-validator object.',
    ):
        bgscv.fit(X_train, Y_train, O_train)


@pytest.mark.parametrize('n_splits', [2, 3, 4])
def test_bgscv_fit(n_splits):
    """Test the fit of bettor grid search cross validation."""

    # Fit
    bgscv = BettorGridSearchCV(
        estimator=TestBettor(),
        param_grid={
            'betting_markets': [
                None,
                ['draw__full_time_goals'],
                ['home_win__full_time_goals', 'away_win__full_time_goals'],
            ],
        },
        cv=TimeSeriesSplit(n_splits),
    )
    bgscv.fit(X_train, Y_train, O_train)

    # Expected CV results
    expected_cv_results = []
    for params in bgscv.param_grid['betting_markets']:
        bettor = TestBettor(betting_markets=params)
        bettor_expected_cv_results = [bettor.betting_markets, {'betting_markets': bettor.betting_markets}]
        for train_indices, test_indices in bgscv.cv.split(X_train):
            X_test, Y_test, O_test = X_train.iloc[test_indices], Y_train.iloc[test_indices], O_train.iloc[test_indices]
            bettor.fit(X_train.iloc[train_indices], Y_train.iloc[train_indices], O_train.iloc[train_indices])
            B_pred = bettor.bet(X_test, O_test)
            test_scores = (
                Y_test[bettor.feature_names_out_].to_numpy() * O_test[bettor.feature_names_odds_].to_numpy() - 1
            )
            test_scores = np.sum((np.nan_to_num(test_scores) * B_pred), axis=1)
            returns = pd.DataFrame(test_scores).set_index(X_test.index).groupby('date').sum()
            dates = pd.DataFrame(pd.date_range(returns.index.min(), returns.index.max()), columns=['date'])
            returns = dates.merge(returns.reset_index(), how='left')
            returns_mean, returns_std = returns[0].fillna(0).mean(), returns[0].fillna(0).std()
            if returns_std == 0 or np.isnan(returns_std):
                max_sharpe_ratio = 100.0
                test_score = max_sharpe_ratio if returns_mean > 0 else -max_sharpe_ratio
            else:
                test_score = np.sqrt(365) * returns_mean / returns_std
            bettor_expected_cv_results.append(test_score)
        mean_test_score = np.mean([bettor_expected_cv_results[2:]])
        std_test_score = np.std([bettor_expected_cv_results[2:]])
        bettor_expected_cv_results.append(mean_test_score)
        bettor_expected_cv_results.append(std_test_score)
        expected_cv_results.append(bettor_expected_cv_results)
    columns = (
        ['param_betting_markets', 'params']
        + [f'split{ind}_test_score' for ind in range(bgscv.n_splits_)]
        + ['mean_test_score', 'std_test_score']
    )
    expected_cv_results = pd.DataFrame(expected_cv_results, columns=columns)
    expected_cv_results['rank_test_score'] = (
        expected_cv_results['mean_test_score'].rank(ascending=False).astype(np.int32)
    )

    # Actual results
    cv_results = pd.DataFrame(bgscv.cv_results_)
    cv_results = cv_results[[col for col in cv_results.columns if not col.endswith('time')]]

    # Assertions
    pd.testing.assert_frame_equal(cv_results, expected_cv_results, check_dtype=False)
    assert bgscv.n_splits_ == bgscv.cv.get_n_splits()
    assert np.array_equal(bgscv.betting_markets_, bgscv.best_estimator_.betting_markets_)
    assert bgscv.init_cash_ == bgscv.best_estimator_.init_cash_
    assert bgscv.stake_ == bgscv.best_estimator_.stake_
    assert np.array_equal(bgscv.feature_names_in_, bgscv.best_estimator_.feature_names_in_)
    assert np.array_equal(bgscv.feature_names_out_, bgscv.best_estimator_.feature_names_out_)
