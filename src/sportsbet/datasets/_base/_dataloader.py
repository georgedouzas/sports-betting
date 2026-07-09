"""Implements the dataloader class."""

from __future__ import annotations

from pathlib import Path
from types import NoneType
from typing import Self

import cloudpickle
import pandas as pd
from sklearn.utils import check_scalar

from ._schema import BaseOddsSchema, BaseStatsSchema

DELIMITER = '__'
LEARNING_TYPES = ('supervised', 'unsupervised')
TARGET_EVENT_STATUSES = ('inplay', 'postplay')


def format_event_time(event_time: pd.Timedelta) -> str:
    """Render an event time as whole minutes, e.g. ``60min``."""
    total_minutes = int(event_time.total_seconds() / 60)
    return f'{total_minutes}min'


def parse_event_time(token: str) -> pd.Timedelta:
    """Parse an ``{n}min`` token back into a time delta."""
    return pd.Timedelta(minutes=int(token[: -len('min')]))


def feature_column(col: str, event_status: str, event_time: pd.Timedelta) -> str:
    """Build a time-varying feature column name."""
    return DELIMITER.join([col, event_status, format_event_time(event_time)])


def odds_column(provider: str, col: str, event_status: str, event_time: pd.Timedelta) -> str:
    """Build an odds column name."""
    return DELIMITER.join([provider, col, event_status, format_event_time(event_time)])


def target_column(col: str, target_event_status: str, target_event_time: pd.Timedelta) -> str:
    """Build a target (Y) column name."""
    return DELIMITER.join([col, target_event_status, format_event_time(target_event_time)])


def odds_market(odds_col: str) -> str:
    """Return the betting market of an odds column (drop the provider prefix)."""
    return DELIMITER.join(odds_col.split(DELIMITER)[1:])


class BaseDataLoader:
    """The base class for dataloaders."""

    def __init__(
        self: Self,
        stats: pd.DataFrame,
        odds: pd.DataFrame,
        stats_schema: type[BaseStatsSchema],
        odds_schema: type[BaseOddsSchema],
        targets: list[str],
    ) -> None:
        self.stats = stats
        self.odds = odds
        self.stats_schema = stats_schema
        self.odds_schema = odds_schema
        self.targets = targets

    def _identity_cols(self: Self) -> list[str]:
        """Snapshot columns that identify a match (all snapshot cols but the event ones)."""
        event_cols = ['event_status', 'event_time']
        return [col for col in self.stats_schema.snapshot_cols() if col not in event_cols]

    def _feature_mask(
        self: Self,
        data: pd.DataFrame,
        target_event_status: str,
        target_event_time: pd.Timedelta,
    ) -> pd.Series:
        """Mask of snapshots strictly before the target moment (no post-target leakage)."""
        if target_event_status == 'postplay':
            return data['event_status'].isin(['preplay', 'inplay'])
        return (data['event_status'] == 'preplay') | (
            (data['event_status'] == 'inplay') & (data['event_time'] < target_event_time)
        )

    def _pivot_features(self: Self, stats: pd.DataFrame) -> pd.DataFrame:
        """Pivot long snapshots into wide, moment-aware feature columns."""
        event_cols = ['event_status', 'event_time']
        index_cols = self._identity_cols()
        feature_cols = [col for col in stats.columns if col not in self.stats_schema.snapshot_cols()]
        X = stats.pivot_table(values=feature_cols, index=index_cols, columns=event_cols, aggfunc='first')
        keep = [
            (col, event_status, event_time)
            for col, event_status, event_time in X.columns
            if event_status in self.stats_schema.col_metadata(col)['include']
        ]
        X = X[keep]
        cols = pd.DataFrame(X.columns.tolist(), columns=['col', 'event_status', 'event_time'])
        cols = cols.groupby(['col'], group_keys=False)[['col', 'event_status', 'event_time']].apply(
            lambda group: group.iloc[:1] if self.stats_schema.col_metadata(group.iloc[0]['col'])['fixed'] else group,
        )
        X = X[list(cols.itertuples(index=False, name=None))]
        X.columns = [
            col if self.stats_schema.col_metadata(col)['fixed'] else feature_column(col, event_status, event_time)
            for col, event_status, event_time in X.columns
        ]
        return X

    def _pivot_odds(self: Self, odds: pd.DataFrame) -> pd.DataFrame:
        """Pivot long odds snapshots into wide, per-provider odds columns."""
        event_cols = ['event_status', 'event_time']
        index_cols = self._identity_cols()
        odds_cols = [col for col in odds.columns if col not in self.odds_schema.snapshot_cols() and col != 'provider']
        O = odds.pivot_table(values=odds_cols, index=index_cols, columns=[*event_cols, 'provider'], aggfunc='first')
        keep = [
            (col, event_status, event_time, provider)
            for col, event_status, event_time, provider in O.columns
            if event_status in self.odds_schema.col_metadata(col)['include']
        ]
        O = O[keep]
        cols = pd.DataFrame(O.columns.tolist(), columns=['col', 'event_status', 'event_time', 'provider'])
        cols = cols.groupby(['col', 'provider'], group_keys=False)[
            ['col', 'event_status', 'event_time', 'provider']
        ].apply(
            lambda group: group.iloc[:1] if self.odds_schema.col_metadata(group.iloc[0]['col'])['fixed'] else group,
        )
        O = O[list(cols.itertuples(index=False, name=None))]
        O.columns = [
            col if self.odds_schema.col_metadata(col)['fixed'] else odds_column(provider, col, event_status, event_time)
            for col, event_status, event_time, provider in O.columns
        ]
        return O

    def _extract(
        self: Self,
        stats: pd.DataFrame,
        odds: pd.DataFrame,
        target_event_status: str,
        target_event_time: pd.Timedelta,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build aligned, date-indexed ``X`` and ``O`` for the given snapshots."""
        X = self._pivot_features(stats[self._feature_mask(stats, target_event_status, target_event_time)])
        O = self._pivot_odds(odds[self._feature_mask(odds, target_event_status, target_event_time)])
        O = O.reindex(X.index)
        X = X.reset_index().set_index('date')
        O.index = X.index
        return X, O

    def _extract_targets(
        self: Self,
        stats: pd.DataFrame,
        index: pd.MultiIndex,
        target_event_status: str,
        target_event_time: pd.Timedelta,
    ) -> pd.DataFrame:
        """Build the target table evaluated at the target moment, aligned to ``index``."""
        mask = (stats['event_status'] == target_event_status) & (stats['event_time'] == target_event_time)
        targets = stats.loc[mask, self._identity_cols() + self.targets].set_index(self._identity_cols())
        targets = targets.reindex(index)
        columns_mapping = {
            target: target_column(target, target_event_status, target_event_time) for target in self.targets
        }
        return targets.rename(columns=columns_mapping)

    def _resolve_params(
        self: Self,
        learning_type: str | None,
        target_event_status: str | None,
        target_event_time: pd.Timedelta | None,
    ) -> tuple[str, str, pd.Timedelta]:
        """Validate and default the learning type and target moment, storing fitted state."""
        check_scalar(learning_type, 'learning_type', (NoneType, str))
        if learning_type is not None and learning_type not in LEARNING_TYPES:
            msg = f'Invalid learning type. It should be one of {LEARNING_TYPES}. Got {learning_type} instead.'
            raise ValueError(msg)
        self.learning_type_ = learning_type if learning_type is not None else 'supervised'

        check_scalar(target_event_status, 'target_event_status', (NoneType, str))
        if target_event_status is not None and target_event_status not in TARGET_EVENT_STATUSES:
            msg = (
                f'Invalid target event status. It should be one of {TARGET_EVENT_STATUSES}. '
                f'Got {target_event_status} instead.'
            )
            raise ValueError(msg)
        self.target_event_status_ = target_event_status if target_event_status is not None else 'postplay'

        check_scalar(target_event_time, 'target_event_time', (NoneType, pd.Timedelta))
        if target_event_time is not None and target_event_time < pd.Timedelta('0min'):
            msg = 'The event time should be positive.'
            raise ValueError(msg)
        self.target_event_time_ = target_event_time if target_event_time is not None else pd.Timedelta('0min')

        return self.learning_type_, self.target_event_status_, self.target_event_time_

    def extract_train_data(
        self: Self,
        *,
        learning_type: str | None = None,
        target_event_status: str | None = None,
        target_event_time: pd.Timedelta | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
        """Extract the training data.

        Read more in the [user guide][dataloader].

        It returns historical data that can be used to create a betting
        strategy based on heuristics or machine learning models.

        The method prepares data for the prediction target defined by
        `event_status` and `event_time`. For example, to predict outcomes
        at 60 minutes in-play, set `event_status='inplay'` and
        `event_time=pd.Timedelta('60min')`. All information before 60 minutes
        will be used as features (X), and the 60-minute outcomes as labels (Y).

        Features (X) include all information available before the target
        time point, while targets (Y) represent outcomes at the target point.
        Odds (O) correspond to the betting markets for the target outcomes.

        Parameters:
            learning_type:
                The type of learning task:
                    - 'supervised' (default): `Y` holds the target outcomes
                    - 'unsupervised': `Y` is `None` (features and odds only)
            target_event_status:
                Status of the target outcomes to predict:
                    - 'inplay': Target outcomes during the match
                    - 'postplay' (default): Target final outcomes after the match
            target_event_time:
                Time point of the target outcomes (e.g., pd.Timedelta('60min')
                for 60 minutes in-play). All data strictly before this time
                becomes features. Defaults to 0 minutes.

        Returns:
            (X, Y, O):
                Input features `X`, target outcomes `Y`, and corresponding odds
                `O`. For `learning_type='unsupervised'`, `Y` is `None`. The three
                components share the same date index and rows.
        """
        self.stats_schema.validate(self.stats)
        self.odds_schema.validate(self.odds)
        if self.stats_schema.snapshot_cols() != self.odds_schema.snapshot_cols():
            msg = 'Stats and odds snapshots columns do not match.'
            raise AssertionError(msg)

        # Check if inplay or postplay events exist
        event_statuses = [status for status in self.stats['event_status'].unique() if status != 'preplay']
        if not event_statuses:
            msg = 'No `inplay` or `postplay` events were found.'
            raise ValueError(msg)

        # Validate and resolve the learning type and target moment
        learning_type, target_event_status, target_event_time = self._resolve_params(
            learning_type,
            target_event_status,
            target_event_time,
        )

        # Split matches into those resolvable at the target moment (training) and the rest (fixtures)
        index_cols = self._identity_cols()
        target_mask = (self.stats['event_status'] == target_event_status) & (
            self.stats['event_time'] == target_event_time
        )
        train_ids = pd.MultiIndex.from_frame(self.stats.loc[target_mask, index_cols])
        if train_ids.empty:
            msg = 'No resolvable events were found for the requested target moment.'
            raise ValueError(msg)
        self._train_ids = train_ids
        train_mask = pd.MultiIndex.from_frame(self.stats[index_cols]).isin(train_ids)
        train_odds_mask = pd.MultiIndex.from_frame(self.odds[index_cols]).isin(train_ids)

        # Extract input and odds data
        X, O = self._extract(
            self.stats[train_mask],
            self.odds[train_odds_mask],
            target_event_status,
            target_event_time,
        )
        self.input_cols_ = X.columns
        self.odds_cols_ = O.columns
        if learning_type == 'unsupervised':
            self.output_cols_ = None
            return X, None, O

        # Extract output data
        Y = self._extract_targets(
            self.stats[train_mask],
            pd.MultiIndex.from_frame(X.reset_index()[index_cols]),
            target_event_status,
            target_event_time,
        )
        Y.index = X.index
        self.output_cols_ = Y.columns
        return X, Y, O

    def extract_fixtures_data(self: Self) -> tuple[pd.DataFrame, None, pd.DataFrame]:
        """Extract the fixtures data.

        Read more in the [user guide][dataloader].

        It returns fixtures data that can be used to make predictions for
        upcoming matches based on a betting strategy.

        Before calling the `extract_fixtures_data` method for the first time,
        the `extract_train_data` method should be called, in order to fix the
        columns of the input and odds data.

        The data contain information about the matches known before the target
        moment, i.e. the training data `X` and the odds data `O`. The
        multi-output targets `Y` is always equal to `None` and is only included
        for consistency with the method `extract_train_data`.

        Returns:
            (X, None, O):
                Each component represents the fixtures input data `X`, the
                multi-output targets `Y` equal to `None`, and the corresponding
                odds `O`, respectively.
        """
        if not hasattr(self, 'input_cols_'):
            msg = 'The `extract_train_data` method should be called before `extract_fixtures_data`.'
            raise ValueError(msg)
        index_cols = self._identity_cols()
        fixtures_mask = ~pd.MultiIndex.from_frame(self.stats[index_cols]).isin(self._train_ids)
        fixtures_odds_mask = ~pd.MultiIndex.from_frame(self.odds[index_cols]).isin(self._train_ids)
        if not fixtures_mask.any():
            X = pd.DataFrame(columns=self.input_cols_, index=pd.DatetimeIndex([], name='date'))
            O = pd.DataFrame(columns=self.odds_cols_, index=pd.DatetimeIndex([], name='date'))
            return X, None, O
        X, O = self._extract(
            self.stats[fixtures_mask],
            self.odds[fixtures_odds_mask],
            self.target_event_status_,
            self.target_event_time_,
        )
        X = X.reindex(columns=self.input_cols_)
        O = O.reindex(columns=self.odds_cols_)
        return X, None, O

    def save(self: Self, path: str) -> Self:
        """Save the dataloader object.

        Args:
            path:
                The path to save the object.

        Returns:
            self:
                The dataloader object.
        """
        with Path(path).open('wb') as file:
            cloudpickle.dump(self, file)
        return self


def load_dataloader(path: str) -> BaseDataLoader:
    """Load the dataloader object.

    Args:
        path:
            The path of the dataloader pickled file.

    Returns:
        dataloader:
            The dataloader object.
    """
    with Path(path).open('rb') as file:
        dataloader = cloudpickle.load(file)
    return dataloader
