"""Create a bettor based on a classifier."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier

from ..core import Data
from ._base import BaseBettor


class ClassifierBettor(MetaEstimatorMixin, BaseBettor):
    """Bettor based on a Scikit-Learn classifier.

    Read more in the [user guide][user-guide].

    Args:
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
        classifier_ (BaseEstimator):
            The fitted clone of `classifier`.

        init_cash_ (float):
            The checked initial cash.

    Examples:
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.preprocessing import OneHotEncoder
        >>> from sklearn.impute import SimpleImputer
        >>> from sklearn.pipeline import make_pipeline
        >>> from sklearn.compose import make_column_transformer
        >>> from sportsbet.evaluation import ClassifierBettor, backtest
        >>> from sportsbet.dataloaders import DataLoader
        >>> from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats
        >>> dataloader = DataLoader(
        ...     param_grid={'league': ['England']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds()
        ... )
        >>> X, Y, O = dataloader.extract_train_data(odds_type='market_average')
        >>> # Create a pipeline to handle categorical features and missing values
        >>> clf_pipeline = make_pipeline(
        ...     make_column_transformer(
        ...         (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']),
        ...         remainder='passthrough',
        ...     ),
        ...     SimpleImputer(),
        ...     DecisionTreeClassifier(random_state=0),
        ... )
        >>> bettor = ClassifierBettor(clf_pipeline)
        >>> results = backtest(bettor, X, Y, O)
        >>> 'Number of bets' in results.columns
        True
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
        if not isinstance(self.classifier, BaseEstimator) or not is_classifier(self.classifier):
            error_msg = f'`ClassifierBettor` requires a classifier. Instead {type(self.classifier)} is given.'
            raise TypeError(error_msg)
        self.classifier_: Any = clone(self.classifier)
        return self

    def _fit(self: Self, X: pd.DataFrame, Y: pd.DataFrame, O: pd.DataFrame) -> Self:
        self._check_classifier()
        self.classifier_.fit(X, Y)
        return self

    def _predict_proba(self: Self, X: pd.DataFrame) -> Data:
        """Return the positive-class probabilities of the fitted classifier."""
        proba = self.classifier_.predict_proba(X)
        if isinstance(proba, list):
            proba = np.concatenate(
                [prob[:, -1].reshape(-1, 1) for prob in proba],
                axis=1,
            )
        elif len(self.classes_) == 1:
            proba = proba[:, -1]
        return proba
