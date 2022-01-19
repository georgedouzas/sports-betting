"""
Test the _classifier module.
"""

import re

import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import quantstats as qs

from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation import ClassifierBettor

X, Y, O = DummySoccerDataLoader().extract_train_data(odds_type='williamhill')


@pytest.mark.parametrize('classifier', [DummyRegressor(), None, 'classifier'])
def test_backtest_raise_error(classifier):
    """Test raising an error on the wrong classifier type."""
    with pytest.raises(
        TypeError,
        match='`ClassifierBettor` requires a classifier. '
        f'Instead {type(classifier)} is given.',
    ):
        ClassifierBettor(classifier).backtest(X, Y, O)


@pytest.mark.parametrize('classifier', [DummyRegressor(), None, 'classifier'])
def test_fit_raise_error(classifier):
    """Test raising an error on the wrong classifier type."""
    with pytest.raises(
        TypeError,
        match='`ClassifierBettor` requires a classifier. '
        f'Instead {type(classifier)} is given.',
    ):
        ClassifierBettor(classifier).fit(X, Y)


def test_fit_check_classifier():
    """Test the cloned classifier."""
    clf = DummyClassifier()
    bettor = ClassifierBettor(clf).fit(X, Y)
    check_is_fitted(bettor.classifier_)
    assert isinstance(bettor.classifier_, DummyClassifier)


def test_backtest_default_params():
    """Test the backtest default parameters."""
    bettor = ClassifierBettor(DummyClassifier()).backtest(X, Y, O)
    assert isinstance(bettor.tscv_, TimeSeriesSplit)
    assert bettor.init_cash_ == 1e3


@pytest.mark.parametrize('n_splits', [2, 3, 5])
def test_backtest_params(n_splits):
    """Test the backtest parameters."""
    bettor = ClassifierBettor(DummyClassifier()).backtest(
        X, Y, O, tscv=TimeSeriesSplit(n_splits=n_splits), init_cash=1e5
    )
    assert isinstance(bettor.tscv_, TimeSeriesSplit)
    assert bettor.tscv_.n_splits == n_splits
    assert bettor.init_cash_ == 1e5


@pytest.mark.parametrize('tscv', [3, 'tscv'])
def test_backtest_params_tscv_raise_error(tscv):
    """Test raising an error on the wrong backtest tscv param."""
    with pytest.raises(
        TypeError,
        match='Parameter `tscv` should be a TimeSeriesSplit cross-validator object.',
    ):
        ClassifierBettor(DummyClassifier()).backtest(X, Y, O, tscv=tscv)


@pytest.mark.parametrize('init_cash', [[4.5], 'init_cash'])
def test_backtest_params_init_cash_raise_type_error(init_cash):
    """Test raising an error on the wrong backtest params."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            "init_cash must be an instance of (<class 'float'>, <class 'int'>), "
            f"not {type(init_cash)}."
        ),
    ):
        ClassifierBettor(DummyClassifier()).backtest(X, Y, O, init_cash=init_cash)


@pytest.mark.parametrize('init_cash', [0.0, -30.0])
def test_backtest_params_init_cash_raise_value_error(init_cash):
    """Test raising an error on the wrong backtest params."""
    with pytest.raises(ValueError, match=f"init_cash == {init_cash}, must be > 0.0."):
        ClassifierBettor(DummyClassifier()).backtest(X, Y, O, init_cash=init_cash)


def test_backtest_params_non_df_raise_value_error():
    """Test raising an error on the wrong backtest input data."""
    with pytest.raises(
        ValueError, match='Input data `X` should be pandas dataframe with a date index.'
    ):
        ClassifierBettor(DummyClassifier()).backtest(X.values, Y, O)


def test_backtest_params_no_date_raise_value_error():
    """Test raising an error on the wrong backtest input data."""
    with pytest.raises(
        ValueError, match='Input data `X` should be pandas dataframe with a date index.'
    ):
        ClassifierBettor(DummyClassifier()).backtest(pd.DataFrame(X.values), Y, O)


def test_backtest():
    """Test the outcome of backtest."""
    n_splits = 2
    tscv = TimeSeriesSplit(n_splits)
    clf = DummyClassifier(strategy='constant', constant=[True, False, True])

    # Backtesting results
    bettor = ClassifierBettor(clf).backtest(X, Y, O, tscv=tscv)
    bettor.backtest(X, Y, O, tscv)

    # Assertions
    results = []
    for train_ind, test_ind in tscv.split(X):
        clf.fit(X.iloc[train_ind], Y.iloc[train_ind])
        Y_pred_prob = np.concatenate(
            [
                prob[:, -1].reshape(-1, 1)
                for prob in clf.predict_proba(X.iloc[test_ind])
            ],
            axis=1,
        )
        value_bets = Y_pred_prob * O.values[test_ind] > 1
        returns = np.nan_to_num(
            (Y.values[test_ind] * O.values[test_ind] - 1) * value_bets
        )
        dates = X.iloc[test_ind].index
        returns_dates = pd.Series(returns.sum(axis=1), index=dates).reindex(
            pd.date_range(dates.min(), dates.max()), fill_value=0
        )
        drawdowns = qs.stats.drawdown_details(returns_dates)
        max_drawdown = np.nan
        max_drawdown_period = pd.NaT
        if (drawdowns['max drawdown'] < 0).sum() > 0:
            dind = drawdowns['max drawdown'].idxmin()
            max_drawdown = -drawdowns['max drawdown'][dind] / bettor.init_cash_
            mask = returns_dates[drawdowns['end'][dind] :] > 0
            max_drawdown_period = mask.index[mask][0] - pd.Timestamp(
                drawdowns['start'][dind]
            )
        results.append(
            [
                X.index[train_ind[0]],
                X.index[train_ind[-1]],
                X.index[train_ind[-1]] - X.index[train_ind[0]],
                dates.min(),
                dates.max(),
                dates.max() - dates.min() + pd.Timedelta('1d'),
                bettor.init_cash_,
                bettor.init_cash_ + returns.sum(),
                100 * returns.sum() / bettor.init_cash_,
                max_drawdown,
                max_drawdown_period,
                (returns != 0).sum(),
                100 * (returns > 0).sum() / (returns != 0).sum(),
                returns.max() * 100,
                returns.min() * 100,
                returns[returns > 0].mean() * 100,
                returns[returns < 0].mean() * 100,
                -returns[returns > 0].sum() / returns[returns < 0].sum(),
                qs.stats.sharpe(
                    returns_dates, periods=len(returns_dates), rf=0.1, smart=True
                ),
                returns[returns != 0].mean() * 100,
                returns[returns != 0].std() * 100,
            ]
        )

    assert len(bettor.backtest_results_) == n_splits
    pd.testing.assert_frame_equal(
        bettor.backtest_results_,
        pd.DataFrame(results, columns=bettor.backtest_results_.columns),
        check_exact=False,
        atol=0.02,
        check_dtype=False,
    )


def test_bet_raise_not_fitted_error():
    """Test raising of not fitted error."""
    with pytest.raises(
        NotFittedError,
        match="This ClassifierBettor instance is"
        " not fitted yet. Call 'fit' with appropriate arguments before"
        " using this estimator",
    ):
        ClassifierBettor(DummyClassifier()).bet(X, O)


def test_bet():
    """Test the bet method."""
    bettor = ClassifierBettor(
        DummyClassifier(strategy='constant', constant=[False, True, True])
    ).fit(X, Y)
    np.testing.assert_array_equal(
        bettor.bet(X, O), np.array([False, True, True]) * (O > 1)
    )
