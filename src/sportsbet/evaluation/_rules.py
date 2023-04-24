"""Create a bettor based on betting rules."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import check_scalar
from typing_extensions import Self

from .. import BoolData, Data
from ._base import _BaseBettor


class OddsComparisonBettor(_BaseBettor):
    """Bettor based on comparison of odds.

    It implements the betting strategy as described in the paper
    [Beating the bookies with their own numbers](https://arxiv.org/pdf/1710.02824.pdf).
    Predicted probabilities of events are based on the average of selected odds
    types for the corresponding events, adjusted by a constant value called alpha. You
    can read more in the [user guide][user-guide].

    Parameters:
        odds_types:
            The odds types to use for the calculation of concensus probabilities. The
            default value corresponds to `'market_average'` if this odds type exists or the
            average of all the other odds columns if `'market_average'` is missing.

        alpha:
            An adjustment term that corresponds to the difference between the consensus
            and real probabilities.

    Attributes:
        odds_types_ (pd.Index):
            The checked value of the odds types.

        alpha_ (float):
            The checked value of the alpha parameter.

        output_keys_ (list[str]):
            The keys of the output columns. They are used to identify
            the consensus columns.

        backtest_results_ (pd.DataFrame):
            The backtesting resuts.

    Examples:
        >>> from sportsbet.evaluation import OddsComparisonBettor
        >>> from sportsbet.datasets import SoccerDataLoader
        >>> # Select only backtesting data for the Italian and Spanish leagues and years 2019 - 2022
        >>> param_grid = {'league': ['Italy', 'Spain'], 'year': [2019, 2020, 2021, 2022]}
        >>> dataloader = SoccerDataLoader(param_grid)
        >>> # Select the market maximum odds
        >>> X, Y, O = dataloader.extract_train_data(
        ... odds_type='market_maximum',
        ... )
        >>> bettor = OddsComparisonBettor(alpha=0.03)
        >>> bettor.backtest(X, Y, O)
        OddsComparisonBettor()
        >>> # Display backtesting results
        >>> bettor.backtest_results_
        Training Start ... Avg Bet Yield [%]  Std Bet Yield [%]
        ...
    """

    def __init__(self: Self, odds_types: list[str] | None = None, alpha: float = 0.05) -> None:
        self.odds_types = odds_types
        self.alpha = alpha

    def _check_odds_types(self: Self, X: pd.DataFrame) -> Self:
        available_odds_types = {col.split('__')[1] for col in X.columns if col.startswith('odds__')}
        if not available_odds_types:
            error_msg = 'Input data do not include any odds columns.'
            raise ValueError(error_msg)
        error_msg = (
            'Parameter `odds_type` should be either `None` or a list of any of the odds types: '
            f'{", ".join(sorted(available_odds_types))}. Got {self.odds_types} instead.'
        )
        if self.odds_types is not None:
            if not isinstance(self.odds_types, list) or any(
                not isinstance(odds_type, str) for odds_type in self.odds_types
            ):
                raise TypeError(error_msg)
            elif not available_odds_types.issuperset(self.odds_types):
                raise ValueError(error_msg)
        self.odds_types_ = (
            sorted(self.odds_types)
            if self.odds_types is not None
            else (['market_average'] if 'market_average' in available_odds_types else sorted(available_odds_types))
        )
        return self

    def _fit(self: Self, X: pd.DataFrame, Y: pd.DataFrame) -> Self:
        self._check_odds_types(X)
        self.alpha_ = check_scalar(self.alpha, 'alpha', target_type=float, min_val=0.0, max_val=1.0)
        self.output_keys_ = ['__'.join(col.split('__')[1:]) for col in Y.columns]
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
        proba_cont = []
        for key in self.output_keys_:
            odds_cols = []
            for odds_type in self.odds_types_:
                odds_col = f'odds__{odds_type}__{key}'
                if odds_col in X.columns:
                    odds_cols.append(odds_col)
            proba_cont.append(1 / X[odds_cols].mean(axis=1))
        proba = (pd.concat(proba_cont, axis=1) - self.alpha_).fillna(0.0).to_numpy()
        proba[proba < 0] = 0.0
        return proba

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
        super().backtest(X, Y, O, tscv, init_cash, refit)
        return self
