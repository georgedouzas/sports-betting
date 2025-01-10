"""Includes base class and functions for evaluating betting strategies."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nptyping import NDArray, Shape
from sklearn import get_config, set_config
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self

from .. import BoolData, Data, Indices
from ._base import BaseBettor

TSCV = TimeSeriesSplit(n_splits=3)


def _fit_bet(
    train_ind: Indices,
    test_ind: Indices,
    bettor: BaseBettor,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    O: pd.DataFrame,
) -> dict:

    # Fit bettor
    bettor.fit(X.iloc[train_ind], Y.iloc[train_ind], O.iloc[train_ind])

    # Predict value bets
    value_bets = bettor.bet(X.iloc[test_ind], O.iloc[test_ind])

    # Calculate returns
    returns = np.nan_to_num(
        (
            Y.loc[test_ind, bettor.feature_names_out_].to_numpy()
            * O.loc[test_ind, bettor.feature_names_odds_].to_numpy()
            - 1
        )
        * value_bets,
    )
    betting_returns = returns[returns != 0]
    n_bets = (returns != 0).sum(axis=0)
    n_bets[n_bets == 0] = 1
    results = {
        'Training start': X.index[train_ind[0]].date(),
        'Training end': X.index[train_ind[-1]].date(),
        'Testing start': X.index[test_ind[0]].date(),
        'Testing end': X.index[test_ind[-1]].date(),
        'Number of betting days': X.index[test_ind].unique().size,
        'Number of bets': (returns != 0).sum(),
        'Yield percentage per bet': round(100 * betting_returns.mean() if betting_returns.size > 0 else 0, 1),
        'ROI percentage': round(100 * bettor.stake_ * betting_returns.sum() / bettor.init_cash_, 1),
        'Final cash': bettor.init_cash_ + bettor.stake_ * betting_returns.sum(),
    }
    results = {
        **results,
        **{
            f'Number of bets ({market})': ret
            for market, ret in zip(bettor.betting_markets_, (returns != 0).sum(axis=0), strict=True)
        },
        **{
            f'Yield percentage per bet ({market})': ret
            for market, ret in zip(
                bettor.betting_markets_,
                [round(100 * val if betting_returns.size > 0 else 0, 1) for val in returns.sum(axis=0) / n_bets],
                strict=True,
            )
        },
    }
    return results


def backtest(
    bettor: BaseBettor,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    O: pd.DataFrame,
    cv: TimeSeriesSplit | None = None,
    n_jobs: int = -1,
    verbose: int = 0,
) -> pd.DataFrame:
    """Backtest the bettor.

    Args:
        bettor:
            The bettor object.

        X:
            The input data. Each row of `X` represents information that is available
            before the start of a specific match. The index should be of type
            `datetime`, named as `'date'`.

        Y:
            The multi-output targets. Each row of `Y` represents information
            that is available after the end of a specific event. The column
            names follow the convention for the output data `Y` of the method
            `extract_train_data` of dataloaders.

        O:
            The odds data. The column names follow the convention for the odds
            data `O` of the method `extract_train_data` of dataloaders.

        cv:
            Provides train/test indices to split time series data samples
            that are observed at fixed time intervals, in train/test sets. The
            default value of the parameter is `None`, corresponding to the default
            `TimeSeriesSplit` object.

        n_jobs:
            Number of CPU cores to use when parallelizing the backtesting runs.
            The default value of `-1` means using all processors.

        verbose:
            The verbosity level.

    Returns:
        results:
            The backtesting results.
    """

    # Check data
    check_consistent_length(X, Y, O)
    if not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.DatetimeIndex):
        error_msg = 'Input data `X` should be pandas dataframe with a date index.'
        raise TypeError(error_msg)
    if not isinstance(Y, pd.DataFrame):
        error_msg = 'Output data `Y` should be pandas dataframe.'
        raise TypeError(error_msg)
    if not isinstance(O, pd.DataFrame):
        error_msg = 'Odds data `O` should be pandas dataframe.'
        raise TypeError(error_msg)

    # Sort data
    indices = np.argsort(X.index)
    X, Y, O = X.iloc[indices], Y.iloc[indices], O.iloc[indices]

    # Check cross validator
    if cv is None:
        cv = TimeSeriesSplit()
    if not isinstance(cv, TimeSeriesSplit):
        error_msg = 'Parameter `cv` should be a TimeSeriesSplit cross-validator object.'
        raise TypeError(error_msg)

    # Calculate results
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_fit_bet)(train_ind, test_ind, bettor, X, Y, O) for train_ind, test_ind in cv.split(X)
    )
    results = pd.DataFrame(results).set_index(['Training start', 'Training end', 'Testing start', 'Testing end'])

    return results


class BettorGridSearchCV(GridSearchCV, BaseBettor):
    """Exhaustive search over specified parameter values for a bettor.

    BettorGridSearchCV implements a `fit`, a`predict`, a `predict_proba',
    a `bet` and a `score` method.

    The parameters of the bettor used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Read more in the [user guide][user-guide].

    Parameters:
        estimator:
            This is assumed to implement the bettor interface.

        param_grid:
            Dictionary with parameters names (`str`) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.

        scoring:
            Strategy to evaluate the performance of the cross-validated model on
            the test set.

            If `scoring` represents a single score, one can use:

            - a single string
            - a callable (see :ref:`scoring`) that returns a single value

            If `scoring` represents multiple scores, one can use:

            - a list or tuple of unique strings
            - a callable returning a dictionary where the keys are the metric
            names and the values are the metric scores
            - a dictionary with metric names as keys and callables a values

        n_jobs:
            Number of jobs to run in parallel. `None` means 1 unless in a
            `joblib.parallel_backend` context. `-1` means using all processors.

        refit:
            Refit an estimator using the best found parameters on the whole
            dataset.

            For multiple metric evaluation, this needs to be a `str` denoting the
            scorer that would be used to find the best parameters for refitting
            the estimator at the end.

            Where there are considerations other than maximum score in
            choosing a best estimator, `refit` can be set to a function which
            returns the selected `best_index_` given `cv_results_`. In that
            case, the `best_estimator_` and `best_params_` will be set
            according to the returned `best_index_` while the `best_score_`
            attribute will not be available.

            The refitted estimator is made available at the `best_estimator_`
            attribute and permits using `predict` directly on this
            `BettorGridSearchCV` instance.

            Also for multiple metric evaluation, the attributes `best_index_`,
            `best_score_` and `best_params_` will only be available if
            `refit` is set and all of them will be determined w.r.t this specific
            scorer.

            See `scoring` parameter to know more about multiple metric
            evaluation.

        cv:
            Provides train/test indices to split time series data samples
            that are observed at fixed time intervals, in train/test sets.

        verbose:
            Controls the verbosity: the higher, the more messages.

        pre_dispatch:
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:

                - `None`, in which case all the jobs are immediately
                created and spawned. Use this for lightweight and
                fast-running jobs, to avoid delays due to on-demand
                spawning of the jobs

                - An int, giving the exact number of total jobs that are
                spawned

                - A str, giving an expression as a function of n_jobs,
                as in '2*n_jobs'

        error_score:
            Value to assign to the score if an error occurs in estimator fitting.
            If set to `'raise'`, the error is raised. If a numeric value is given,
            FitFailedWarning is raised. This parameter does not affect the refit
            step, which will always raise the error.

        return_train_score:
            If `False`, the `cv_results_` attribute will not include training
            scores. Computing training scores is used to get insights on how different
            parameter settings impact the overfitting/underfitting trade-off.
            However computing the scores on the training set can be computationally
            expensive and is not strictly required to select the parameters that
            yield the best generalization performance.

    Attributes:
        cv_results_:
            A dict with keys as column headers and values as columns, that can be
            imported into a pandas `DataFrame`.

            The key `'params'` is used to store a list of parameter
            settings dicts for all the parameter candidates.

            The `mean_fit_time`, `std_fit_time`, `mean_score_time` and
            `std_score_time` are all in seconds.

            For multi-metric evaluation, the scores for all the scorers are
            available in the `cv_results_` dict at the keys ending with that
            scorer's name.

        best_estimator_:
            Estimator that was chosen by the search, i.e. estimator
            which gave highest score (or smallest loss if specified)
            on the left out data. Not available if `refit=False`.

        best_score_:
            Mean cross-validated score of the best_estimator

            For multi-metric evaluation, this is present only if `refit` is
            specified.

            This attribute is not available if `refit` is a function.

        best_params_:
            Parameter setting that gave the best results on the hold out data.

            For multi-metric evaluation, this is present only if `refit` is
            specified.

        best_index_:
            The index (of the `cv_results_` arrays) which corresponds to the best
            candidate parameter setting.

            For multi-metric evaluation, this is present only if ``refit`` is
            specified.

        scorer_:
            Scorer function used on the held out data to choose the best
            parameters for the model.

            For multi-metric evaluation, this attribute holds the validated
            `scoring` dict which maps the scorer key to the scorer callable.

        n_splits_:
            The number of cross-validation splits (folds/iterations).

        refit_time_:
            Seconds used for refitting the best model on the whole dataset.

            This is present only if `refit` is not False.

        multimetric_:
            Whether or not the scorers compute several metrics.

        classes_:
            The classes labels. This is present only if `refit` is specified and
            the underlying estimator is a classifier.

        n_features_in_:
            Number of features seen during `fit`. Only defined if
            `best_estimator_` is defined and that `best_estimator_` exposes
            `n_features_in_` when fit.

        feature_names_in_:
            Names of features seen during `fit`. Only defined if
            `best_estimator_` is defined and that `best_estimator_` exposes
            `feature_names_in_` when fit.

    Examples:
        >>> from sportsbet.evaluation import BettorGridSearchCV, OddsComparisonBettor, backtest
        >>> from sportsbet.datasets import SoccerDataLoader
        >>> from sklearn.model_selection import TimeSeriesSplit
        >>> # Select only backtesting data for the Italian and Spanish leagues and years 2019 - 2022
        >>> param_grid = {'league': ['Italy', 'Spain'], 'year': [2019, 2020, 2021, 2022]}
        >>> dataloader = SoccerDataLoader(param_grid)
        >>> # Select the market maximum odds
        >>> X, Y, O = dataloader.extract_train_data(
        ... odds_type='market_maximum',
        ... )
        >>> # Backtest the bettor
        >>> bettor = BettorGridSearchCV(
        ... estimator=OddsComparisonBettor(),
        ... param_grid={'alpha': [0.02, 0.05, 0.1, 0.2, 0.3]},
        ... cv=TimeSeriesSplit(2),
        ... )
        >>> backtest(bettor, X, Y, O, cv=TimeSeriesSplit(2)).reset_index()
          Training start ... Yield percentage per bet (under_2.5__full_time_goals)
        ...
    """

    def __init__(
        self: Self,
        estimator: BaseBettor,
        param_grid: dict | list,
        *,
        scoring: str | Callable | list | tuple | dict[str, Callable] | None = None,
        n_jobs: int | None = None,
        refit: bool | str | Callable = True,
        cv: TimeSeriesSplit = TSCV,
        verbose: int = 0,
        pre_dispatch: int | str = '2*n_jobs',
        error_score: str | float | int = np.nan,
        return_train_score: bool = False,
    ) -> None:
        GridSearchCV.__init__(
            self,
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _check_attr(self: Self, attr_name: str, raise_attr_error: bool, check_best_estimator: bool) -> None:
        if raise_attr_error:
            try:
                check_is_fitted(self)
            except NotFittedError as nfe:
                error_msg = f"'{self.__class__.__name__}' object has no attribute '{attr_name}'"
                raise AttributeError(error_msg) from nfe
        else:
            check_is_fitted(self)
        if check_best_estimator and not hasattr(self, 'best_estimator_'):
            error_msg = f"'{self.__class__.__name__}' object has no attribute '{attr_name}'"
            raise AttributeError(error_msg)

    def modify_scorer(self: Self, scorer: Callable) -> Callable:
        def _scorer(
            estimator: BaseBettor,
            X: pd.DataFrame,
            Y: pd.DataFrame,
            sample_weight: NDArray[Shape['*'], float] | None = None,  # noqa: F722
            **kwargs: dict[str, Any],
        ) -> float:
            Y = Y[estimator.feature_names_out_]
            return scorer(estimator, X, Y, sample_weight, **kwargs)

        return _scorer

    def _fit(self: Self, X: pd.DataFrame, Y: pd.DataFrame, O: pd.DataFrame | None) -> Self:
        if not isinstance(self.estimator, BaseBettor):
            error_msg = f'`BettorGridSearchCV` requires a bettor as estimator. Instead {type(self.estimator)} is given.'
            raise TypeError(error_msg)
        if not isinstance(self.cv, TimeSeriesSplit):
            error_msg = 'Parameter `cv` should be a TimeSeriesSplit cross-validator object.'
            raise TypeError(error_msg)
        initial_scoring = deepcopy(self.scoring)
        if O is not None and initial_scoring is None:
            enable_metadata_routing = get_config().get('enable_metadata_routing')
            set_config(enable_metadata_routing=True)
            self.estimator.set_fit_request(O=True).set_score_request(O=True)
            GridSearchCV.fit(self, X, Y, O=O)
            self.estimator.set_fit_request(O=None).set_score_request(O=None)
            set_config(enable_metadata_routing=enable_metadata_routing)
        else:
            if initial_scoring is None:
                error_msg = (
                    'The default scoring method requires the odds data `O` to be provided. '
                    'Invoke the fit method as `object.fit(X, Y, O)`.'
                )
                raise TypeError(error_msg)
            scorers, _ = self._get_scorers()
            self.scoring: Callable | dict[str, Callable] = (
                self.modify_scorer(scorers)
                if not isinstance(scorers, dict)
                else {name: self.modify_scorer(scorer) for name, scorer in scorers.items()}
            )
            GridSearchCV.fit(self, X, Y)
            self.scoring = initial_scoring
        return self

    def _predict_proba(self: Self, X: pd.DataFrame) -> Data:
        return self.best_estimator_._predict_proba(X)

    def fit(self: Self, X: pd.DataFrame, Y: pd.DataFrame, O: pd.DataFrame | None = None) -> Self:
        """Fit the bettor to the input data and multi-output targets.

        Args:
            X:
                The input data.

            Y:
                The multi-output targets.

            O:
                The odds data.

        Returns:
            self:
                The fitted bettor object.
        """
        self._fit(X, Y, O)
        if hasattr(self, 'best_estimator_'):
            self.init_cash_ = self.best_estimator_.init_cash_
            self.stake_ = self.best_estimator_.stake_
        if O is not None and hasattr(self, 'best_estimator_'):
            self.feature_names_odds_ = self.best_estimator_._get_feature_names_odds(O)
        return self

    def predict_proba(self: Self, X: pd.DataFrame) -> Data:
        """Predict class probabilities for multi-output targets.

        Args:
            X:
                The input data.

        Returns:
            Y:
                The positive class probabilities.
        """
        self._check_attr('predict_proba', False, True)
        return self.best_estimator_.predict_proba(X)

    def predict(self: Self, X: pd.DataFrame) -> BoolData:
        """Predict class labels for multi-output targets.

        Args:
            X:
                The input data.

        Returns:
            Y:
                The positive class labels.
        """
        self._check_attr('predict', False, True)
        return self.best_estimator_.predict(X)

    def bet(self: Self, X: pd.DataFrame, O: pd.DataFrame) -> BoolData:
        """Predict the value bets for the provided input data and odds.

        Args:
            X:
                The input data.

            O:
                The odds data.

        Returns:
            B:
                The value bets.
        """
        self._check_attr('bet', False, True)
        return self.best_estimator_.bet(X, O)

    @property
    def classes_(self: Self) -> list:
        self._check_attr('classes_', True, True)
        return [np.array([0, 1]) for _ in enumerate(self.betting_markets_)]

    @property
    def betting_markets_(self: Self) -> NDArray[Shape['*'], str]:  # noqa: F722
        self._check_attr('betting_markets_', True, True)
        return self.best_estimator_.betting_markets_

    @property
    def feature_names_out_(self: Self) -> NDArray[Shape['*'], str]:  # noqa: F722
        self._check_attr('feature_names_out_', True, True)
        return self.best_estimator_.feature_names_out_
