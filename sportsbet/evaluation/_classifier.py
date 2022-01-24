"""
Create a portofolio and backtest its performance
using sports betting data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sklearn.utils.validation import check_is_fitted
from vectorbt import Portfolio
import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.utils import check_consistent_length, check_scalar

from sklearn.model_selection import TimeSeriesSplit


def _predict_probas(clf, X):
    return np.concatenate(
        [prob[:, -1].reshape(-1, 1) for prob in clf.predict_proba(X)],
        axis=1,
    )


def _extract_stats(portfolio, training_start, training_end):
    stats = pd.DataFrame(
        portfolio.stats().values.reshape(1, -1), columns=portfolio.stats().index
    )
    num_cols = [col for col in stats.columns if stats[col].dtype.name == 'object']
    stats[num_cols] = stats[num_cols].astype(float)
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


class ClassifierBettor:
    """Bettor based on a scikit-learn classifier.

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
    <sportsbet.evaluation._classifier.ClassifierBettor object at...
    >>> # Display backtesting results
    >>> bettor.backtest_results_
      Training Start ... Avg Bet Yield [%]  Std Bet Yield [%]
    0     2019-01-09 ...         -0.621803         118.393155
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
        return self

    def _check_backtest_params(self, tscv, init_cash):
        if tscv is None:
            tscv = TimeSeriesSplit()
        if not isinstance(tscv, TimeSeriesSplit):
            raise TypeError(
                'Parameter `tscv` should be a TimeSeriesSplit cross-validator object.'
            )
        self.tscv_ = tscv
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

    def backtest(self, X, Y, O, tscv=None, init_cash=1000):
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

        Returns
        -------
        self : :class:`~sportsbet.evaluation.ClassifierBettor` object.
            The classifier bettor.
        """
        self._check_classifier()._check_backtest_params(tscv, init_cash)
        clf = clone(self.classifier)
        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            dates = X.index
        else:
            raise ValueError(
                'Input data `X` should be pandas dataframe with a date index.'
            )
        check_consistent_length(X, Y, O)

        # Calculate cross-validation stats
        results = []
        for train_ind, test_ind in self.tscv_.split(X):

            # Fit classifier
            clf.fit(X.iloc[train_ind], Y.iloc[train_ind])

            # Predict class probabilities
            Y_pred_prob = _predict_probas(clf, X.iloc[test_ind])

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
            portfolio = Portfolio.from_orders(
                prices,
                orders,
                freq='0.5D',
                cash_sharing=True,
                init_cash=self.init_cash_,
            )
            results.append(
                (
                    _extract_stats(
                        portfolio, X.index[train_ind[0]], X.index[train_ind[-1]]
                    ),
                    portfolio.plot_value,
                )
            )
        self.backtest_results_, plot_value_funcs = zip(*results)
        self.backtest_results_ = pd.concat(self.backtest_results_, ignore_index=True)
        self.backtest_plot_value_ = lambda ind: plot_value_funcs[ind]()

        return self

    def fit(self, X, Y):
        self._check_classifier()
        self.classifier_ = clone(self.classifier).fit(X, Y)
        return self

    def bet(self, X, O):
        check_is_fitted(self, 'classifier_')
        return _predict_probas(self.classifier_, X) * O > 1
