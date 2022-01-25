"""
Create a portofolio and backtest its performance
using sports betting data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from sklearn.base import is_classifier, clone

from ._base import _BaseBettor


class ClassifierBettor(_BaseBettor):
    """Bettor based on a Scikit-Learn classifier.

    Read more in the :ref:`user guide <user_guide>`.

    Parameters
    ----------
    classifier : classifier object
        A scikit-learn classifier object implementing :term:`fit`, :term:`score`
        and :term:`predict_proba`.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> from sklearn.impute import SimpleImputer
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.compose import make_column_transformer
    >>> from sportsbet.evaluation import ClassifierBettor
    >>> from sportsbet.datasets import FDSoccerDataLoader
    >>> # Select only backtesting data for the Italian league and years 2020, 2021
    >>> param_grid = {'league': ['Italy'], 'year': [2020, 2021]}
    >>> dataloader = FDSoccerDataLoader(param_grid)
    >>> # Select the odds of Pinnacle bookmaker
    >>> X, Y, O = dataloader.extract_train_data(
    ... odds_type='pinnacle',
    ... drop_na_thres=1.0
    ... )
    Football-Data.co.uk...
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

    def __init__(self, classifier):
        self.classifier = classifier

    def _check_classifier(self):
        if not is_classifier(self.classifier):
            raise TypeError(
                '`ClassifierBettor` requires a classifier. '
                f'Instead {type(self.classifier)} is given.'
            )
        self.classifier_ = clone(self.classifier)
        return self

    def _fit(self, X, Y):
        self._check_classifier()
        self.classifier_.fit(X, Y)
        return self

    def _predict_proba(self, X):
        """Predict class probabilities for multi-output targets.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            The positive class probabilities.
        """
        # if X.size == 0:

        return np.concatenate(
            [prob[:, -1].reshape(-1, 1) for prob in self.classifier_.predict_proba(X)],
            axis=1,
        )
