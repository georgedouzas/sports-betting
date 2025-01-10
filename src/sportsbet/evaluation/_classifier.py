"""Create a bettor based on a classifier."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier
from typing_extensions import Self

from .. import BoolData, Data
from ._base import BaseBettor


class ClassifierBettor(MetaEstimatorMixin, BaseBettor):
    """Bettor based on a Scikit-Learn classifier.

    Read more in the [user guide][user-guide].

    Parameters:
        classifier:
            A scikit-learn classifier object implementing `fit`, `score`
            and `predict_proba`.

        betting_markets:
            Select the betting markets from the ones included in the data.

        init_cash:
            The initial cash to use when betting.

        stake:
            The stake of each bet.

    Attributes:
        tscv_ (TimeSeriesSplit):
            The checked value of time series cross-validator object. If `tscv` is `None`,
            it uses the default `TimeSeriesSplit` object.

        init_cash_:
            The checked value of initial cash. If `init_cash` is `None`, it uses the value
            of `1e3`.

        backtesting_results_ (pd.DataFrame):
            The backtesting results.

    Examples:
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.preprocessing import OneHotEncoder
        >>> from sklearn.impute import SimpleImputer
        >>> from sklearn.pipeline import make_pipeline
        >>> from sklearn.compose import make_column_transformer
        >>> from sportsbet.evaluation import ClassifierBettor, backtest
        >>> from sportsbet.datasets import SoccerDataLoader
        >>> # Select only backtesting data for the Italian league and years 2020, 2021
        >>> param_grid = {'league': ['Italy'], 'year': [2020, 2021]}
        >>> dataloader = SoccerDataLoader(param_grid)
        >>> # Select the odds of Pinnacle bookmaker
        >>> X, Y, O = dataloader.extract_train_data(
        ... odds_type='market_average',
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
        >>> backtest(bettor, X, Y, O).reset_index()
          Training start ... Yield percentage per bet (under_2.5__full_time_goals)
        ...
    """

    _required_parameters: ClassVar = ['classifier']

    def __init__(
        self: Self,
        classifier: BaseEstimator,
        betting_markets: list[str] | None = None,
        init_cash: float | None = None,
        stake: float | None = None,
    ) -> None:
        super().__init__(betting_markets, init_cash, stake)
        self.classifier = classifier

    def _check_classifier(self: Self) -> Self:
        if not is_classifier(self.classifier):
            error_msg = f'`ClassifierBettor` requires a classifier. Instead {type(self.classifier)} is given.'
            raise TypeError(error_msg)
        self.classifier_: Any = clone(self.classifier)
        return self

    def _fit(self: Self, X: pd.DataFrame, Y: pd.DataFrame, O: pd.DataFrame) -> Self:
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
        proba = self.classifier_.predict_proba(X)
        if isinstance(proba, list):
            proba = np.concatenate(
                [prob[:, -1].reshape(-1, 1) for prob in proba],
                axis=1,
            )
        elif len(self.classes_) == 1:
            proba = proba[:, -1]
        return proba

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
        return super().fit(X, Y, O)

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
        """Predict class labels for multi-output targets.

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
