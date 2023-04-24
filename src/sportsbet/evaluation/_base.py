"""Includes base class and functions for evaluating betting strategies."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from abc import ABCMeta
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from rich.progress import track
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import check_consistent_length, check_scalar
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self
from vectorbt import Portfolio

from .. import BoolData, Data


class _BaseBettor(MultiOutputMixin, ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """The base class for bettors.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def _check_backtest_params(self: Self, tscv: TimeSeriesSplit | None, init_cash: float | None) -> Self:
        """Check backtest parameters."""
        # Check cross validator
        if tscv is None:
            tscv = TimeSeriesSplit()
        if not isinstance(tscv, TimeSeriesSplit):
            error_msg = 'Parameter `tscv` should be a TimeSeriesSplit cross-validator object.'
            raise TypeError(error_msg)
        self.tscv_ = tscv

        # Check initial cash
        if init_cash is None:
            init_cash = 1e3
        check_scalar(
            init_cash,
            'init_cash',
            (float, int),
            min_val=0.0,
            include_boundaries='neither',
        )
        self.init_cash_ = float(init_cash)
        return self

    def _validate_data(
        self: Self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        O: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        indices = np.argsort(X.index)
        return X.iloc[indices], Y.iloc[indices], O.iloc[indices]

    def _extract_portfolio(self: Self, prices: pd.DataFrame, orders: pd.DataFrame) -> Portfolio:
        """Extract portfolio."""
        return Portfolio.from_orders(prices, orders, freq='0.5D', cash_sharing=True, init_cash=self.init_cash_)

    @staticmethod
    def _extract_stats(portfolio: Portfolio, training_start: int, training_end: int) -> pd.DataFrame:
        """Extract statistics from portfolio."""
        portfolio_stats = portfolio.stats()
        if portfolio_stats is None:
            return pd.DataFrame([])

        # Reshape data
        stats = pd.DataFrame(portfolio_stats.to_numpy().reshape(1, -1), columns=portfolio_stats.index)

        # Cast to numerical
        num_cols = [col for col in stats.columns if stats[col].dtype.name == 'object']
        stats[num_cols] = stats[num_cols].astype(float)

        # Select columns
        stats = stats[
            [
                'Start',
                'End',
                'Period',
                'Start Value',
                'End Value',
                'Total Return [%]',
                'Total Trades',
                'Win Rate [%]',
                'Best Trade [%]',
                'Worst Trade [%]',
                'Avg Winning Trade [%]',
                'Avg Losing Trade [%]',
                'Profit Factor',
                'Sharpe Ratio',
            ]
        ]

        # Append columns
        stats = pd.concat(
            [
                pd.DataFrame(
                    {
                        'Training Start': [training_start],
                        'Training End': [training_end],
                        'Training Period': [training_end - training_start],
                    },
                ),
                stats,
            ],
            axis=1,
        )

        # Rename columns
        stats = stats.rename(
            columns={
                'Start': 'Testing Start',
                'End': 'Testing End',
                'Period': 'Testing Period',
                **{name: name.replace('Trade', 'Bet') for name in stats.columns if 'Trade' in name},
            },
        )

        # Calculate extra statistics
        yields = 2 * portfolio.trades.records_readable['Return']
        stats = stats.assign(
            **{
                'Best Bet [%]': 100 * yields.max(),
                'Worst Bet [%]': 100 * yields.min(),
                'Avg Winning Bet [%]': 100 * yields[yields > 0].mean(),
                'Avg Losing Bet [%]': 100 * yields[yields < 0].mean(),
                'Avg Bet Yield [%]': 100 * yields.mean(),
                'Std Bet Yield [%]': 100 * yields.to_numpy().std() if yields.size > 0 else np.nan,
            },
        )
        return stats

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
        return self._fit(X, Y)

    def predict_proba(self: Self, X: pd.DataFrame) -> Data:
        """Predict class probabilities for multi-output targets.

        Args:
            X:
                The input data.

        Returns:
            Y:
                The positive class probabilities.
        """
        check_is_fitted(self)
        return self._predict_proba(X)

    def predict(self: Self, X: pd.DataFrame) -> BoolData:
        """Predict class probabilities for multi-output targets.

        Args:
            X:
                The input data.

        Returns:
            Y:
                The positive class labels.
        """
        check_is_fitted(self)
        decision_threshold = 0.5
        return self._predict_proba(X) > decision_threshold

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
        if X.shape[0] == 0 and O.shape[0] == 0:
            return np.array([], dtype=bool).reshape(0, O.shape[1])
        return self.predict_proba(X) * O > 1

    def backtest(
        self: Self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        O: pd.DataFrame | None,
        tscv: TimeSeriesSplit | None = None,
        init_cash: float | None = None,
        refit: bool | None = True,
    ) -> Self:
        """Backtest the bettor.

        Args:
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
        if O is None or O.empty:
            return self

        # Apply checks
        X, Y, O = self._validate_data(X, Y, O)
        self._check_backtest_params(tscv, init_cash)

        dates = X.index

        # Calculate cross-validation stats
        results = []
        for train_ind, test_ind in track(list(self.tscv_.split(X)), description='Backtesting bettor', transient=True):
            # Fit bettor
            self.fit(X.iloc[train_ind], Y.iloc[train_ind])

            # Predict value bets
            value_bets = self.bet(X.iloc[test_ind], O.iloc[test_ind])

            # Calculate returns
            returns = np.nan_to_num((Y.iloc[test_ind].to_numpy() * O.iloc[test_ind].to_numpy() - 1) * value_bets)

            # Convert betting market to assets prices
            prices = pd.DataFrame(returns).set_index(dates[test_ind])
            indices = prices.index.unique().reindex(pd.date_range(dates[test_ind].min(), dates[test_ind].max()))[0]
            prices = (
                prices.reset_index()
                .merge(pd.DataFrame(indices, columns=['date']), on='date', how='right')
                .fillna(0.0)
                .set_index('date')
            )
            prices = prices.groupby(by='date').aggregate(
                lambda price: (
                    sum(price != 0) + 1,
                    sum(price != 0) + sum(price) + 1,
                ),
            )
            prices = pd.DataFrame(
                np.array(prices.to_numpy().T.reshape(-1).tolist()).reshape(prices.shape[1], -1).T,
                index=np.repeat(prices.index, 2),
            )

            # Get buy and sell orders
            orders = pd.DataFrame(
                np.repeat([np.repeat([1, -1], prices.shape[1])], prices.shape[0] // 2, axis=0).reshape(
                    -1,
                    prices.shape[1],
                ),
                index=prices.index,
            )
            orders.loc[prices.index.difference(dates[test_ind]).tolist()] = 0
            orders.index.name = 'date'
            mask = (
                (prices + orders)
                .loc[dates[test_ind], :]
                .groupby('date')
                .apply(lambda row: (row.iloc[0, :] == 2) & (row.iloc[1, :] == 0))  # noqa: PLR2004
            )
            orders[mask] = 0

            # Get portofolio from prices and orders
            portfolio = self._extract_portfolio(prices, orders)
            results.append(self._extract_stats(portfolio, X.index[train_ind[0]], X.index[train_ind[-1]]))
        self.backtest_results_ = pd.concat(results, ignore_index=True)

        if refit:
            self.fit(X, Y)

        return self

    def _fit(self: Self, X: pd.DataFrame, Y: pd.DataFrame) -> Self:
        return self

    def _predict_proba(self: Self, X: pd.DataFrame) -> Data:
        return np.array([], dtype=float)

    def save(self: Self, path: str) -> Self:
        """Save the bettor object.

        Args:
            path:
                The path to save the object.

        Returns:
            self:
                The bettor object.
        """
        with Path(path).open('wb') as file:
            cloudpickle.dump(self, file)
        return self


def load_bettor(path: str) -> _BaseBettor:
    """Load the bettor object.

    Args:
        path:
            The path of the bettor pickled file.

    Returns:
        bettor:
            The bettor object.
    """
    with Path(path).open('rb') as file:
        bettor = cloudpickle.load(file)
    return bettor
