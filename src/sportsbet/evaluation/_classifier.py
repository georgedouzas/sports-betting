"""Create a bettor based on a classifier."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import TimeSeriesSplit
from typing_extensions import Self

from .. import BoolData, Data
from ._base import _BaseBettor


class ClassifierBettor(_BaseBettor):
    """Bettor based on a Scikit-Learn classifier.

    Read more in the [user guide][user-guide].

    Parameters:
        classifier:
            A scikit-learn classifier object implementing `fit`, `score`
            and `predict_proba`.

    Attributes:
        tscv_ (TimeSeriesSplit):
            The checked value of time series cross-validator object. If `tscv` is `None`,
            it uses the default `TimeSeriesSplit` object.

        init_cash_:
            The checked value of initial cash. If `init_cash` is `None`, it uses the value
            of `1e3`.

        backtest_results_ (pd.DataFrame):
            The backtesting results.

    Examples:
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.preprocessing import OneHotEncoder
        >>> from sklearn.impute import SimpleImputer
        >>> from sklearn.pipeline import make_pipeline
        >>> from sklearn.compose import make_column_transformer
        >>> from sportsbet.evaluation import ClassifierBettor
        >>> from sportsbet.datasets import SoccerDataLoader
        >>> # Select only backtesting data for the Italian league and years 2020, 2021
        >>> param_grid = {'league': ['Italy'], 'year': [2020, 2021]}
        >>> dataloader = SoccerDataLoader(param_grid)
        >>> # Select the odds of Pinnacle bookmaker
        >>> X, Y, O = dataloader.extract_train_data(
        ... odds_type='pinnacle',
        ... drop_na_thres=1.0
        ... )
        >>> # Create a pipeline to handle categorical features and missing values
        >>> clf_pipeline = make_pipeline(
        ... make_column_transformer(
        ... (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']),
        ... remainder='passthrough'
        ... ),
        ... SimpleImputer(),
        ... DecisionTreeClassifier(random_state=0)
        ... )
        >>> # Backtest the bettor
        >>> bettor = ClassifierBettor(clf_pipeline)
        >>> bettor.backtest(X, Y, O)
        ClassifierBettor(classifier=...
        >>> # Display backtesting results
        >>> bettor.backtest_results_
        Training Start ... Avg Bet Yield [%]  Std Bet Yield [%]
        ...
    """

    def __init__(self: Self, classifier: BaseEstimator) -> None:
        self.classifier = classifier

    def _check_classifier(self: Self) -> Self:
        if not is_classifier(self.classifier):
            error_msg = f'`ClassifierBettor` requires a classifier. Instead {type(self.classifier)} is given.'
            raise TypeError(error_msg)
        self.classifier_: Any = clone(self.classifier)
        return self

    def _fit(self: Self, X: pd.DataFrame, Y: pd.DataFrame) -> Self:
        self._check_classifier()
        self.classifier_.fit(X, Y)
        return self

    def _predict_proba(self: Self, X: pd.DataFrame) -> Data:
        """Predict class probabilities for multi-output targets.

        Args:
            X:
                The input data.

        Returns:
            Y:
                The positive class probabilities.
        """
        return np.concatenate(
            [prob[:, -1].reshape(-1, 1) for prob in self.classifier_.predict_proba(X)],
            axis=1,
        )

    def fit(self: Self, X: pd.DataFrame, Y: pd.DataFrame) -> Self:
        """Fit the bettor to the input data and multi-output targets.

        Args:
            X:
                The input data.

            Y:
                The multi-output targets.

        Returns:
            self:
                The fitted bettor object.
        """
        return super().fit(X, Y)

    def predict_proba(self: Self, X: pd.DataFrame) -> Data:
        """Predict class probabilities for multi-output targets.

        Args:
            X:
                The input data.

        Returns:
            Y:
                The positive class probabilities.
        """
        return super().predict_proba(X)

    def predict(self: Self, X: pd.DataFrame) -> BoolData:
        """Predict class probabilities for multi-output targets.

        Args:
            X:
                The input data.

        Returns:
            Y:
                The positive class labels.
        """
        return super().predict(X)

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
        return super().bet(X, O)

    def backtest(
        self: Self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        O: pd.DataFrame | None,
        tscv: TimeSeriesSplit | None = None,
        init_cash: float | None = 1e3,
        refit: bool | None = True,
    ) -> Self:
        """Backtest the bettor.

        Args:
            X:
                The input data. Each row of `X` represents information that is available
                before the start of a specific match. The rows should be
                sorted by an index named as `'date'`.

            Y:
                The multi-output targets. Each row of `Y` represents information
                that is available after the end of a specific match. The column
                names follow the convention for the output data `Y` of the method
                `extract_train_data`.

            O:
                The odds data. Each row of `O` represents information
                that is available after the end of a specific match. The column
                names follow the convention for the output data `Y` of the method
                `extract_train_data`.

            tscv:
                Provides train/test indices to split time series data samples
                that are observed at fixed time intervals, in train/test sets. The
                default value of the parameter is `None`.

            init_cash:
                The initial cash to use for backtesting.

            refit:
                Refit the bettor using the whole input data and multi-output targets.

        Returns:
            self:
                The backtested bettor.
        """
        self._check_classifier()
        super().backtest(X, Y, O, tscv, init_cash, refit)
        return self
