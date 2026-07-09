"""Includes base class and functions for evaluating betting strategies."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import ClassVar, Self

import cloudpickle
import numpy as np
import pandas as pd
from nptyping import NDArray, Shape, String
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_consistent_length, check_scalar
from sklearn.utils.validation import _check_feature_names, check_is_fitted

from .. import BoolData, Data
from ..datasets._base._dataloader import parse_event_time

# Ordering of event statuses, used to pick the latest odds snapshot for a market.
_STATUS_RANK = {'preplay': 0, 'inplay': 1, 'postplay': 2}
# Number of ``__``-delimited tokens in an odds column: provider, market, status, time.
_N_ODDS_TOKENS = 4


def market_base(market: str) -> str:
    """Return the base market name (drop the ``__status__time`` suffix)."""
    return market.split('__', maxsplit=1)[0]


def is_odds_column(col: str) -> bool:
    """Return whether a column follows the four-token odds grammar."""
    return len(col.split('__')) == _N_ODDS_TOKENS


def latest_odds_column(columns: list[str], base: str, provider: str | None = None) -> str | None:
    """Return the odds column for a market base at the latest snapshot.

    Args:
        columns:
            Candidate odds column names (`{provider}__{base}__{status}__{time}`).
        base:
            The market base to match (e.g. `home_win`).
        provider:
            If given, only match this provider.

    Returns:
        The matching column at the latest ``(status, time)``, or `None`.
    """
    best: str | None = None
    best_key: tuple[int, pd.Timedelta] | None = None
    for col in columns:
        if not is_odds_column(col):
            continue
        col_provider, col_base, status, time = col.split('__')
        if col_base != base or (provider is not None and col_provider != provider):
            continue
        key = (_STATUS_RANK.get(status, -1), parse_event_time(time))
        if best_key is None or key > best_key:
            best_key, best = key, col
    return best


class BaseBettor(MultiOutputMixin, ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """The base class for bettors.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    TOL = 1e-6
    # Mutually-exclusive markets, identified by their base name (e.g. ``home_win``).
    COMPLEMENTARY_EVENTS: ClassVar = [
        ['home_win', 'draw', 'away_win'],
        ['over_2.5', 'under_2.5'],
        ['over_3.5', 'under_3.5'],
    ]

    _append_odds = False

    def __init__(
        self: Self,
        betting_markets: list[str] | None = None,
        init_cash: float | None = None,
        stake: float | None = None,
    ) -> None:
        self.betting_markets = betting_markets
        self.init_cash = init_cash
        self.stake = stake

    def _get_feature_names_odds(self: Self, O: pd.DataFrame) -> NDArray[Shape['*'], String]:  # noqa: F722
        # One odds column per selected market base, at the latest snapshot, ordered to match
        # `betting_markets_` so positional alignment with `Y` holds.
        columns = list(O.columns)
        odds_cols = [latest_odds_column(columns, base) for base in self.betting_markets_]
        return np.array([col for col in odds_cols if col is not None])

    def _append_odds_data(self: Self, X: pd.DataFrame, O: pd.DataFrame | None) -> pd.DataFrame:
        """Merge `O` into `X` for bettors that model odds directly (e.g. odds comparison)."""
        if self._append_odds and O is not None:
            return pd.concat([X, O], axis=1)
        return X

    def _check(
        self: Self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        O: pd.DataFrame | None,
        Y_betting_markets: list[str],
    ) -> None:

        # Betting markets are identified by their base name (e.g. `home_win`)
        Y_bases = list(dict.fromkeys(Y_betting_markets))
        if self.betting_markets is None:
            self.betting_markets_ = np.array(Y_bases)
        elif not isinstance(self.betting_markets, list) or (
            isinstance(self.betting_markets, list) and not self.betting_markets
        ):
            error_msg = 'Parameter `betting_markets` should be a list of betting market names.'
            raise TypeError(error_msg)
        elif isinstance(self.betting_markets, list) and not set(self.betting_markets).issubset(Y_bases):
            error_msg = 'Parameter `betting_markets` does not contain valid names.'
            raise ValueError(error_msg)
        else:
            self.betting_markets_ = np.array(self.betting_markets)

        # Initial cash
        init_cash = self.init_cash
        if init_cash is None:
            init_cash = 1e4
        check_scalar(
            init_cash,
            'init_cash',
            (float, int),
            min_val=0.0,
            include_boundaries='neither',
        )
        self.init_cash_ = float(init_cash)

        # Stake
        stake = self.stake
        if stake is None:
            stake = 50.0
        check_scalar(
            stake,
            'stake',
            (float, int),
            min_val=0.0,
            include_boundaries='neither',
        )
        self.stake_ = float(stake)

        # Check features. Map each selected market base to its single target column.
        _check_feature_names(self, X, reset=True)
        base_to_target = {market_base(col): col for col in Y.columns}
        self.feature_names_out_ = np.array([base_to_target[base] for base in self.betting_markets_])
        if O is not None:
            self.feature_names_odds_ = self._get_feature_names_odds(O)

    @property
    def classes_(self: Self) -> list:
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            error_msg = f"'{self.__class__.__name__}' object has no attribute 'classes_'"
            raise AttributeError(error_msg) from nfe
        return [np.array([0, 1]) for _ in enumerate(self.betting_markets_)]

    def _validate_X_Y(  # noqa: N802
        self: Self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:

        # Check number of samples
        check_consistent_length(X, Y)

        # Check data type
        if not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.DatetimeIndex):
            error_msg = 'Input data `X` should be pandas dataframe with a date index.'
            raise TypeError(error_msg)
        if not isinstance(Y, pd.DataFrame):
            error_msg = 'Output data `Y` should be pandas dataframe.'
            raise TypeError(error_msg)

        # Check Y columns follow the target grammar `{betting_market}__{event_status}__{event_time}`
        Y_cols = [col.split('__') for col in Y.columns]
        error_msg = (
            "Output data column names should follow a naming "
            "convention of the form `f'{betting_market}__{event_status}__{event_time}'`"
        )
        if {len(tokens) for tokens in Y_cols} != {3}:
            raise ValueError(error_msg)
        Y_betting_markets = [market_base(col) for col in Y.columns]

        return X, Y, Y_betting_markets

    def _validate_X_O(  # noqa: N802
        self: Self,
        X: pd.DataFrame,
        O: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:

        # Check number of samples
        check_consistent_length(X, O)

        # Check data type
        if not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.DatetimeIndex):
            error_msg = 'Input data `X` should be pandas dataframe with a date index.'
            raise TypeError(error_msg)
        if not isinstance(O, pd.DataFrame):
            error_msg = 'Odds data `O` should be pandas dataframe.'
            raise TypeError(error_msg)

        # Check O columns follow `{provider}__{betting_market}__{event_status}__{event_time}`
        O_cols = [col.split('__') for col in O.columns]
        error_msg = (
            "Odds data column names should follow a naming "
            "convention of the form `f'{provider}__{betting_market}__{event_status}__{event_time}'`"
        )
        if {len(tokens) for tokens in O_cols} != {4}:
            raise ValueError(error_msg)
        O_providers = [tokens[0] for tokens in O_cols]
        if len(set(O_providers)) != 1:
            error_msg = 'Providers of odds data column names should be unique.'
            raise ValueError(error_msg)
        O_betting_markets = [tokens[1] for tokens in O_cols]

        return X, O, O_betting_markets

    @abstractmethod
    def _fit(self: Self, X: pd.DataFrame, Y: pd.DataFrame, O: pd.DataFrame | None) -> Self:
        return self

    @abstractmethod
    def _predict_proba(self: Self, X: pd.DataFrame) -> Data:
        return np.array([], dtype=float)

    def _normalize_proba(self: Self, Y_proba_pred: Data) -> Data:
        for events in self.COMPLEMENTARY_EVENTS:
            if set(self.betting_markets_).issuperset(events):
                mask = np.isin(self.betting_markets_, events)
                Y_proba_pred_sum = Y_proba_pred[:, mask].sum(axis=1)
                Y_proba_pred_sum[Y_proba_pred_sum == 0.0] = self.TOL
                Y_proba_pred[:, mask] = Y_proba_pred[:, mask] / Y_proba_pred_sum.reshape(-1, 1)
        return Y_proba_pred

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
        X, Y, Y_betting_markets = self._validate_X_Y(X, Y)
        if O is not None:
            X, O, O_betting_markets = self._validate_X_O(X, O)
            if set(Y_betting_markets) != set(O_betting_markets):
                error_msg = 'Output and odds data column names are not compatible.'
                raise ValueError(error_msg)
        X_fit = self._append_odds_data(X, O)
        self._check(X_fit, Y, O, Y_betting_markets)
        return self._fit(X_fit, Y[self.feature_names_out_], O[self.feature_names_odds_] if O is not None else None)

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
        _check_feature_names(self, X, reset=False)
        if X.empty:
            return np.empty((0, self.betting_markets_.size), dtype=float)
        Y_proba_pred = self._predict_proba(X)
        Y_proba_pred = Y_proba_pred.reshape(Y_proba_pred.shape[0], -1)
        if Y_proba_pred.shape[1] != self.betting_markets_.size:
            error_msg = 'Predicted probabilities and selected betting markets have incompatible shapes.'
            raise TypeError(error_msg)
        Y_proba_pred = self._normalize_proba(Y_proba_pred)
        return Y_proba_pred

    def predict(self: Self, X: pd.DataFrame) -> BoolData:
        """Predict class labels for multi-output targets.

        Args:
            X:
                The input data.

        Returns:
            Y:
                The positive class labels.
        """
        decision_threshold = 0.5
        Y_pred = self.predict_proba(X) > decision_threshold
        return Y_pred

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
        Y_proba_pred = self.predict_proba(self._append_odds_data(X, O))
        X, O, O_betting_markets = self._validate_X_O(X, O)
        if not set(O_betting_markets).issuperset(self.betting_markets_):
            error_msg = 'Odds data do not include selected betting markets.'
            raise ValueError(error_msg)
        O = O[self._get_feature_names_odds(O)]
        B_pred = Y_proba_pred * O > 1
        B_pred_selected = []
        for events in self.COMPLEMENTARY_EVENTS:
            events_indices = np.where(np.isin(self.betting_markets_, events))[0]
            if events_indices.size > 0:
                estimated_returns = np.nan_to_num(
                    (O.iloc[:, events_indices] * Y_proba_pred[:, events_indices] - 1).to_numpy(),
                )
                estimated_returns += [eps * self.TOL for eps in range(estimated_returns.shape[1])]
                mask = estimated_returns != np.max(estimated_returns, axis=1).reshape(-1, 1)
                B_pred_events = B_pred.iloc[:, events_indices].copy()
                B_pred_events[mask] = False
                B_pred_selected.append(B_pred_events)
        return pd.concat(B_pred_selected, axis=1).to_numpy()

    def score(self: Self, X: pd.DataFrame, Y: pd.DataFrame, O: pd.DataFrame) -> float:
        """Return the annual sharpe ratio on the given data.

        Args:
            X:
                The input data.

            Y:
                The output data.

            O:
                The odds data.

        Returns:
            score:
                Annual sharpe ratio of predicted value bets.
        """
        check_is_fitted(self)
        X, Y, Y_betting_markets = self._validate_X_Y(X, Y)
        X, O, O_betting_markets = self._validate_X_O(X, O)
        if set(Y_betting_markets) != set(O_betting_markets):
            error_msg = 'Output and odds data column names are not compatible.'
            raise ValueError(error_msg)
        Y = Y[self.feature_names_out_]
        O = O[self._get_feature_names_odds(O)]
        returns = np.sum(
            np.nan_to_num(
                (Y.to_numpy().astype(int) * O.to_numpy().astype(float) - 1) * self.bet(X, O).astype(int),
            ),
            axis=1,
        )
        returns = pd.DataFrame(returns).set_index(X.index).groupby('date').sum()
        dates = pd.DataFrame(pd.date_range(returns.index.min(), returns.index.max()), columns=['date'])
        returns = dates.merge(returns.reset_index(), how='left')
        returns_mean, returns_std = returns[0].fillna(0).mean(), returns[0].fillna(0).std()
        if returns_std == 0 or np.isnan(returns_std):
            max_sharpe_ratio = 100.0
            return max_sharpe_ratio if returns_mean > 0 else -max_sharpe_ratio
        return np.sqrt(365) * returns_mean / returns_std


def save_bettor(bettor: BaseBettor, path: str) -> None:
    """Save the bettor object.

    Args:
        bettor:
            The bettor object.

        path:
            The path to save the object.

    Returns:
        self:
            The bettor object.
    """
    with Path(path).open('wb') as file:
        cloudpickle.dump(bettor, file)


def load_bettor(path: str) -> BaseBettor:
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
