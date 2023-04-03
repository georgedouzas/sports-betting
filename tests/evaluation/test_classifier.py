"""Test the ClassifierBettor class."""

from typing import cast

import numpy as np
import pandas as pd
import pytest
import quantstats as qs
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_is_fitted
from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation import ClassifierBettor

X_train, Y_train, O_train = DummySoccerDataLoader().extract_train_data(odds_type='williamhill')


@pytest.mark.parametrize('classifier', [DummyRegressor(), None, 'classifier'])
def test_backtest_raise_error(classifier):
    """Test raising an error on the wrong classifier type."""
    with pytest.raises(
        TypeError,
        match=f'`ClassifierBettor` requires a classifier. Instead {type(classifier)} is given.',
    ):
        ClassifierBettor(classifier).backtest(X_train, Y_train, O_train)


@pytest.mark.parametrize('classifier', [DummyRegressor(), None, 'classifier'])
def test_fit_raise_error(classifier):
    """Test raising an error on the wrong classifier type."""
    with pytest.raises(
        TypeError,
        match=f'`ClassifierBettor` requires a classifier. Instead {type(classifier)} is given.',
    ):
        ClassifierBettor(classifier).fit(X_train, Y_train)


def test_fit_check_classifier():
    """Test the cloned classifier."""
    clf = DummyClassifier()
    bettor = ClassifierBettor(clf).fit(X_train, Y_train)
    check_is_fitted(bettor)
    assert isinstance(bettor.classifier_, DummyClassifier)


def test_backtest_default_params():
    """Test the backtest default parameters."""
    bettor = ClassifierBettor(DummyClassifier()).backtest(X_train, Y_train, O_train)
    default_init_cash = 1e3
    assert isinstance(bettor.tscv_, TimeSeriesSplit)
    assert bettor.init_cash_ == default_init_cash


@pytest.mark.parametrize('n_splits', [2, 3, 5])
def test_backtest_params(n_splits):
    """Test the backtest parameters."""
    init_cash = 1e5
    bettor = ClassifierBettor(DummyClassifier()).backtest(
        X_train,
        Y_train,
        O_train,
        tscv=TimeSeriesSplit(n_splits=n_splits),
        init_cash=init_cash,
    )
    assert isinstance(bettor.tscv_, TimeSeriesSplit)
    assert bettor.tscv_.n_splits == n_splits
    assert bettor.init_cash_ == init_cash


@pytest.mark.parametrize('tscv', [3, 'tscv'])
def test_backtest_params_tscv_raise_error(tscv):
    """Test raising an error on the wrong backtest tscv param."""
    with pytest.raises(
        TypeError,
        match='Parameter `tscv` should be a TimeSeriesSplit cross-validator object.',
    ):
        ClassifierBettor(DummyClassifier()).backtest(X_train, Y_train, O_train, tscv=tscv)


@pytest.mark.parametrize('init_cash', [[4.5], 'init_cash'])
def test_backtest_params_init_cash_raise_type_error(init_cash):
    """Test raising an error on the wrong backtest params."""
    with pytest.raises(
        TypeError,
        match=f'init_cash must be an instance of {{float, int}}, not {type(init_cash).__name__}.',
    ):
        ClassifierBettor(DummyClassifier()).backtest(X_train, Y_train, O_train, init_cash=init_cash)


@pytest.mark.parametrize('init_cash', [0.0, -30.0])
def test_backtest_params_init_cash_raise_value_error(init_cash):
    """Test raising an error on the wrong backtest params."""
    with pytest.raises(ValueError, match=f"init_cash == {init_cash}, must be > 0.0."):
        ClassifierBettor(DummyClassifier()).backtest(X_train, Y_train, O_train, init_cash=init_cash)


def test_backtest_params_non_df_raise_value_error():
    """Test raising an error on the wrong backtest input data."""
    with pytest.raises(TypeError, match='Input data `X` should be pandas dataframe with a date index.'):
        ClassifierBettor(DummyClassifier()).backtest(cast(pd.DataFrame, X_train.values), Y_train, O_train)


def test_backtest_params_no_date_raise_value_error():
    """Test raising an error on the wrong backtest input data."""
    with pytest.raises(TypeError, match='Input data `X` should be pandas dataframe with a date index.'):
        ClassifierBettor(DummyClassifier()).backtest(pd.DataFrame(X_train.values), Y_train, O_train)


def test_backtest():
    """Test the outcome of backtest."""
    n_splits = 2
    tscv = TimeSeriesSplit(n_splits)
    clf = DummyClassifier(strategy='constant', constant=[True, False, True])

    # Backtesting results
    bettor = ClassifierBettor(clf).backtest(X_train, Y_train, O_train, tscv=tscv)
    bettor.backtest(X_train, Y_train, O_train, tscv)

    # Assertions
    assert O_train is not None
    results = []
    for train_ind, test_ind in tscv.split(X_train):
        clf.fit(X_train.iloc[train_ind], Y_train.iloc[train_ind])
        Y_pred_prob = np.concatenate(
            [prob[:, -1].reshape(-1, 1) for prob in clf.predict_proba(X_train.iloc[test_ind])],
            axis=1,
        )
        value_bets = Y_pred_prob * O_train.to_numpy()[test_ind] > 1
        returns = np.nan_to_num((Y_train.to_numpy()[test_ind] * O_train.to_numpy()[test_ind] - 1) * value_bets)
        dates = X_train.iloc[test_ind].index
        returns_total = pd.Series(returns.sum(axis=1), index=dates).reindex(
            pd.date_range(dates.min(), dates.max()),
            fill_value=0,
        )
        results.append(
            [
                X_train.index[train_ind[0]],
                X_train.index[train_ind[-1]],
                X_train.index[train_ind[-1]] - X_train.index[train_ind[0]],
                dates.min(),
                dates.max(),
                dates.max() - dates.min() + pd.Timedelta('1d'),
                bettor.init_cash_,
                bettor.init_cash_ + returns.sum(),
                100 * returns.sum() / bettor.init_cash_,
                (returns != 0).sum(),
                100 * (returns > 0).sum() / (returns != 0).sum(),
                returns.max() * 100,
                returns.min() * 100,
                returns[returns > 0].mean() * 100,
                returns[returns < 0].mean() * 100,
                -returns[returns > 0].sum() / returns[returns < 0].sum(),
                qs.stats.sharpe(returns_total, periods=len(returns_total), rf=0.1, smart=True),
                returns[returns != 0].mean() * 100,
                returns[returns != 0].std() * 100,
            ],
        )

    assert len(bettor.backtest_results_) == n_splits
    pd.testing.assert_frame_equal(
        bettor.backtest_results_,
        pd.DataFrame(results, columns=bettor.backtest_results_.columns),
        check_exact=False,
        rtol=0.15,
        check_dtype=False,
    )


def test_bet_raise_not_fitted_error():
    """Test raising of not fitted error."""
    assert O_train is not None
    with pytest.raises(
        NotFittedError,
        match="This ClassifierBettor instance is"
        " not fitted yet. Call 'fit' with appropriate arguments before"
        " using this estimator",
    ):
        ClassifierBettor(DummyClassifier()).bet(X_train, O_train)


def test_bet():
    """Test the bet method."""
    assert O_train is not None
    bettor = ClassifierBettor(DummyClassifier(strategy='constant', constant=[False, True, True])).fit(X_train, Y_train)
    np.testing.assert_array_equal(bettor.bet(X_train, O_train), np.array([False, True, True]) * (O_train > 1))
