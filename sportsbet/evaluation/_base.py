"""
Includes base class and functions for evaluating betting strategies.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import ABCMeta

from vectorbt import Portfolio
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MultiOutputMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_consistent_length, check_scalar
from sklearn.model_selection import TimeSeriesSplit


class _BaseBettor(MultiOutputMixin, ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """The base class for bettors.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def _check_backtest_params(self, tscv, init_cash):
        """Check backtest parameters."""

        # Check cross validator
        if tscv is None:
            tscv = TimeSeriesSplit()
        if not isinstance(tscv, TimeSeriesSplit):
            raise TypeError(
                'Parameter `tscv` should be a TimeSeriesSplit cross-validator object.'
            )
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
        self.init_cash_ = init_cash
        return self

    def _check_dates(self, X):
        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            dates = X.index
        else:
            raise ValueError(
                'Input data `X` should be pandas dataframe with a date index.'
            )
        return dates

    def _extract_portfolio(self, prices, orders):
        """Extract portfolio."""
        return Portfolio.from_orders(
            prices, orders, freq='0.5D', cash_sharing=True, init_cash=self.init_cash_
        )

    @staticmethod
    def _extract_stats(portfolio, training_start, training_end):
        """Extract statistics from portfolio."""

        # Reshape data
        stats = pd.DataFrame(
            portfolio.stats().values.reshape(1, -1), columns=portfolio.stats().index
        )

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
                'Max Drawdown [%]',
                'Max Drawdown Duration',
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
                    }
                ),
                stats,
            ],
            axis=1,
        )

        # Rename columns
        stats.rename(
            columns={
                **{
                    name: name.replace('Trade', 'Bet')
                    for name in stats.columns
                    if 'Trade' in name
                },
                **{
                    'Start': 'Testing Start',
                    'End': 'Testing End',
                    'Period': 'Testing Period',
                },
            },
            inplace=True,
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
                'Std Bet Yield [%]': 100 * yields.values.std()
                if yields.size > 0
                else np.nan,
            }
        )
        return stats

    def backtest(self, X, Y, O, tscv=None, init_cash=1000, refit=True):
        """Backtest the bettor.

        Parameters
        ----------
        X : :class:`~pandas.DataFrame` object
            The input data. Each row of `X` represents information that is available
            before the start of a specific match. The rows should be
            sorted by an index named as ``'date'``.

        Y : :class:`~pandas.DataFrame` object
            The multi-output targets. Each row of `Y` represents information
            that is available after the end of a specific match. The column
            names follow the convention for the output data `Y` of the method
            :func:`~sportsbet.datasets._BaseDataLoader.extract_train_data`.

        O : :class:`~pandas.DataFrame` object
            The odds data. Each row of `O` represents information
            that is available after the end of a specific match. The column
            names follow the convention for the output data ``Y`` of the method
            :func:`~sportsbet.datasets._BaseDataLoader.extract_train_data`.

        tscv : :class:`~sklearn.model_selection.TimeSeriesSplit` object, default=None
            Provides train/test indices to split time series data samples
            that are observed at fixed time intervals, in train/test sets. The
            default value param ``None``.

        init_cash : init, default=1000
            The initial cash to use for backtesting.

        refit : bool, default=True
            Refit the bettor using the whole input data and multi-output targets.

        Returns
        -------
        self : bettor object.
            The backtested bettor.
        """
        check_consistent_length(X, Y, O)
        self._check_classifier()._check_backtest_params(tscv, init_cash)
        dates = self._check_dates(X)

        # Calculate cross-validation stats
        results = []
        for train_ind, test_ind in self.tscv_.split(X):

            # Fit bettor
            self.fit(X.iloc[train_ind], Y.iloc[train_ind])

            # Predict class probabilities
            Y_pred_prob = self.predict_proba(X.iloc[test_ind])

            # Predict value bets
            value_bets = Y_pred_prob * O.iloc[test_ind].values > 1

            # Calculate returns
            returns = np.nan_to_num(
                (Y.iloc[test_ind].values * O.iloc[test_ind].values - 1) * value_bets
            )

            # Convert betting market to assets prices
            prices = pd.DataFrame(returns).set_index(dates[test_ind])
            prices = prices.groupby(by='date').aggregate(
                lambda price: (
                    sum(price != 0) + 1,
                    sum(price != 0) + sum(price) + 1,
                )
            )
            prices = prices.reindex(
                pd.date_range(dates[test_ind].min(), dates[test_ind].max()),
                fill_value=(1, 1),
            )
            prices = pd.DataFrame(
                np.array(prices.values.T.reshape(-1).tolist())
                .reshape(prices.shape[1], -1)
                .T,
                index=np.repeat(prices.index, 2),
            )

            # Get buy and sell orders
            orders = pd.DataFrame(
                np.repeat(
                    [np.repeat([1, -1], prices.shape[1])], prices.shape[0] // 2, axis=0
                ).reshape(-1, prices.shape[1]),
                index=prices.index,
            )
            orders.loc[prices.index.difference(dates[test_ind])] = 0
            orders.index.name = 'date'
            mask = (
                (prices + orders)
                .loc[dates[test_ind], :]
                .groupby('date')
                .apply(lambda row: (row.iloc[0, :] == 2) & (row.iloc[1, :] == 0))
            )
            orders[mask] = 0

            # Get portofolio from prices and orders
            portfolio = self._extract_portfolio(prices, orders)
            results.append(
                (
                    self._extract_stats(
                        portfolio, X.index[train_ind[0]], X.index[train_ind[-1]]
                    ),
                    portfolio.plot_value,
                )
            )
        self.backtest_results_, plot_value_funcs = zip(*results)
        self.backtest_results_ = pd.concat(self.backtest_results_, ignore_index=True)
        self.backtest_plot_value_ = lambda ind: plot_value_funcs[ind]()

        if refit:
            self.fit(X, Y)

        return self

    def fit(self, X, Y):
        """Fit the bettor to the input data and multi-output targets.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            The multi-output targets.

        Returns
        -------
        self : Bettor object
            The fitted bettor object.
        """
        return self._fit(X, Y)

    def predict_proba(self, X):
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
        check_is_fitted(self)
        return self._predict_proba(X)

    def predict(self, X):
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
        check_is_fitted(self)
        return self._predict_proba(X) > 0.5

    def bet(self, X, O):
        """Predict the value bets for the provided input data and odds.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        O : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            The odds data.

        Returns
        -------
        B : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            The value bets.
        """
        if X.shape[0] == 0 and O.shape[0] == 0:
            return O
        return self.predict_proba(X) * O > 1

    def _fit(self, X, Y):
        return self

    def _predict_proba(self, X):
        return
