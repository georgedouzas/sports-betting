"""Implements the base dataloader class shared by all dataloaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import NoneType
from typing import Self

import cloudpickle
import pandas as pd
from sklearn.utils import check_scalar

from ... import FixturesData, ParamGrid, TrainData
from .._store import PreparationReport
from ._schema import (
    EVENT_COLS,
    IDENTITY_COLS,
    build_odds_schema,
    build_stats_schema,
    derive_metadata,
)

DELIMITER = '__'
LEARNING_TYPES = ('supervised', 'unsupervised')
TARGET_EVENT_STATUSES = ('inplay', 'postplay')
INPUT_EVENT_STATUSES = ('preplay', 'inplay', 'postplay')
STATUS_RANK = {'preplay': 0, 'inplay': 1, 'postplay': 2}
PARAM_COLS = ['league', 'division', 'year']


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


class BaseDataLoader(ABC):
    """The abstract base class for dataloaders.

    A dataloader reads long event-snapshot `stats` and `odds` data, validates it,
    derives the available providers, markets and per-column metadata from the data
    itself, and extracts moment-aware training and fixtures data. Everything but the
    data source is implemented here; a concrete dataloader only needs to implement
    the abstract [`_snapshots`][sportsbet.datasets.BaseDataLoader] method and, when
    its data is downloadable, override the optional `_all_params` hook to enable
    parameter discovery.

    Args:
        param_grid:
            Selects the data to include. Keys are parameters like `'league'`,
            `'division'` or `'year'` and values are sequences of allowed values,
            mirroring scikit-learn's `ParameterGrid`. The default `None` selects
            all available parameters.

    Attributes:
        param_grid_ (list[dict]):
            The league/division/year combinations of the loaded data.

        stats_ (pd.DataFrame):
            The validated long `stats` snapshots.

        odds_ (pd.DataFrame):
            The validated long `odds` snapshots of the selected provider.

        drop_na_thres_ (float):
            The checked value of `drop_na_thres`.

        odds_type_ (str | None):
            The checked value of `odds_type`.

        input_cols_ (pd.Index):
            The columns of `X` for training and fixtures data.

        output_cols_ (pd.Index | None):
            The columns of `Y` for training data.

        odds_cols_ (pd.Index):
            The columns of `O` for training and fixtures data.
    """

    def __init__(self: Self, param_grid: ParamGrid | None = None) -> None:
        self.param_grid = param_grid
        self._provided_snapshots: tuple[pd.DataFrame, pd.DataFrame] | None = None
        self._downloaded: tuple[pd.DataFrame, pd.DataFrame] | None = None
        self._components: tuple | None = None

    @abstractmethod
    def _snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the long `stats`/`odds` snapshots.

        Every dataloader must implement this.
        """

    @property
    def sources(self: Self) -> tuple:
        """The data sources of the dataloader.

        Ask a source what it publishes, since a `param_grid` cannot be written before it is known what exists. It is
        empty for a dataloader whose data is provided directly rather than downloaded.
        """
        return ()

    def _all_params(self: Self) -> list[dict]:
        """Return the available parameter combinations, used only to filter `param_grid`.

        Discovery is not a dataloader concern: a `param_grid` cannot be written before it is known what exists, so the
        question is answered by the source's `available_params`. A dataloader needs the answer only to refuse a
        combination the sources do not publish.
        """
        msg = f'{type(self).__name__} does not implement parameter discovery.'
        raise NotImplementedError(msg)

    def prepare(self: Self, dry_run: bool = False, refresh: bool = False) -> PreparationReport:
        """Populate the store with the data the selected parameters require.

        A dataloader that is not backed by a source has nothing to download, so it prepares nothing and returns an
        empty report. Dataloaders backed by a source override it.

        Args:
            dry_run:
                If `True`, nothing is fetched and nothing is spent; the report says what a preparation would do.

            refresh:
                If `True`, everything is fetched again, including what the store already holds.

        Returns:
            report:
                What was fetched, what was already held, and what it cost.
        """
        return PreparationReport()

    def _filter_params(self: Self, params: list[dict]) -> list[dict]:
        """Filter the available combinations by `param_grid` (no invalid combinations are fabricated)."""
        if self.param_grid is None:
            return params
        grids = self.param_grid if isinstance(self.param_grid, list) else [self.param_grid]
        return [
            combination
            for combination in params
            if any(all(combination[key] in values for key, values in grid.items()) for grid in grids)
        ]

    def _selected_params(self: Self) -> list[dict]:
        """Return the available combinations the `param_grid` selects."""
        return self._filter_params(self._all_params())

    @staticmethod
    def _finalize(data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the `date` and `event_time` dtypes of a snapshot frame."""
        data = data.reset_index(drop=True)
        if data.empty:
            return data
        data['date'] = pd.to_datetime(data['date'], utc=True).astype('datetime64[ns, UTC]')
        if not pd.api.types.is_timedelta64_dtype(data['event_time']):
            data['event_time'] = pd.to_timedelta(data['event_time'], unit='m')
        data['event_time'] = data['event_time'].astype('timedelta64[ns]')
        return data

    def _validate_snapshots(self: Self, stats: pd.DataFrame, odds: pd.DataFrame) -> None:
        """Validate that the snapshots carry the required identity and event columns."""
        for name, frame in [('stats', stats), ('odds', odds)]:
            missing = [col for col in EVENT_COLS + IDENTITY_COLS if col not in frame.columns]
            if missing:
                msg = f'The {name} data is missing the required columns: {missing}.'
                raise ValueError(msg)
        if 'provider' not in odds.columns:
            msg = 'The odds data is missing the `provider` column.'
            raise ValueError(msg)

    def get_odds_types(self: Self) -> list[str]:
        """Return the available odds types (providers) derived from the data."""
        _, odds = self._snapshots()
        return sorted(odds['provider'].dropna().unique().tolist())

    def _load(self: Self, odds_type: str | None) -> None:
        """Read and validate the snapshots, derive their metadata and build the inputs."""
        stats, odds = self._snapshots()
        stats = self._finalize(stats)
        odds = self._finalize(odds)
        self._validate_snapshots(stats, odds)
        providers = sorted(odds['provider'].dropna().unique().tolist())
        if odds_type is not None and odds_type not in providers:
            msg = f'Invalid odds type. It should be one of {providers}. Got {odds_type} instead.'
            raise ValueError(msg)
        stats_value_cols = [col for col in stats.columns if col not in EVENT_COLS + IDENTITY_COLS]
        odds_value_cols = [col for col in odds.columns if col not in EVENT_COLS + IDENTITY_COLS + ['provider']]
        if not odds_value_cols:
            msg = (
                'The odds data carries no betting markets, so there is nothing to bet on and nothing to predict. '
                'The markets a model learns are the ones its odds price, so a dataloader needs an odds source.'
            )
            raise ValueError(msg)
        stats_metadata = derive_metadata(stats, stats_value_cols)
        odds_metadata = derive_metadata(odds, odds_value_cols, allow_fixed=False)

        odds = odds[odds['provider'] == odds_type] if odds_type is not None else odds.iloc[0:0]
        self.stats_ = build_stats_schema(stats_metadata).validate(stats)
        self.odds_ = build_odds_schema(odds_metadata).validate(odds)
        self.stats_schema_ = build_stats_schema(stats_metadata)
        self.odds_schema_ = build_odds_schema(odds_metadata)
        self.targets_ = odds_value_cols
        self.odds_type_ = odds_type
        self.param_grid_ = stats[PARAM_COLS].drop_duplicates().to_dict('records')

    def _apply_drop_na(self: Self, X: pd.DataFrame, drop_na_thres: float) -> pd.DataFrame:
        """Drop feature columns whose missingness exceeds `drop_na_thres`."""
        if not X.empty:
            keep = X.columns[X.isna().mean() <= (1.0 - drop_na_thres)]
            X = X[keep]
        self.input_cols_ = X.columns
        self.drop_na_thres_ = drop_na_thres
        return X

    def _identity_cols(self: Self) -> list[str]:
        """Snapshot columns that identify a match (all snapshot cols but the event ones)."""
        return [col for col in self.stats_schema_.snapshot_cols() if col not in EVENT_COLS]

    def _feature_mask(
        self: Self,
        data: pd.DataFrame,
        target_event_status: str,
        target_event_time: pd.Timedelta,
        input_event_status: str | None = None,
        input_event_time: pd.Timedelta | None = None,
    ) -> pd.Series:
        """Mask of snapshots strictly before the target, optionally capped at an input horizon.

        A snapshot is kept when it is strictly before the target moment (no post-target leakage) and, when an input
        horizon is given, at or before it.
        """
        rank = data['event_status'].map(STATUS_RANK)
        time = data['event_time']
        target_rank = STATUS_RANK[target_event_status]
        before_target = (rank < target_rank) | ((rank == target_rank) & (time < target_event_time))
        if input_event_status is None:
            return before_target
        input_rank = STATUS_RANK[input_event_status]
        up_to_input = (rank < input_rank) | ((rank == input_rank) & (time <= input_event_time))
        return before_target & up_to_input

    def _pivot_features(self: Self, stats: pd.DataFrame) -> pd.DataFrame:
        """Pivot long snapshots into wide, moment-aware feature columns."""
        index_cols = self._identity_cols()
        feature_cols = [col for col in stats.columns if col not in self.stats_schema_.snapshot_cols()]
        X = stats.pivot_table(values=feature_cols, index=index_cols, columns=EVENT_COLS, aggfunc='first')
        keep = [
            (col, event_status, event_time)
            for col, event_status, event_time in X.columns
            if event_status in self.stats_schema_.col_metadata(col)['include']
        ]
        X = X[keep]
        cols = pd.DataFrame(X.columns.tolist(), columns=['col', 'event_status', 'event_time'])
        cols = cols.groupby(['col'], group_keys=False)[['col', 'event_status', 'event_time']].apply(
            lambda group: group.iloc[:1] if self.stats_schema_.col_metadata(group.iloc[0]['col'])['fixed'] else group,
        )
        X = X[list(cols.itertuples(index=False, name=None))]
        X.columns = [
            col if self.stats_schema_.col_metadata(col)['fixed'] else feature_column(col, event_status, event_time)
            for col, event_status, event_time in X.columns
        ]
        return X

    def _pivot_odds(self: Self, odds: pd.DataFrame) -> pd.DataFrame:
        """Pivot long odds snapshots into wide, per-provider odds columns."""
        index_cols = self._identity_cols()
        odds_cols = [col for col in odds.columns if col not in self.odds_schema_.snapshot_cols() and col != 'provider']
        O = odds.pivot_table(values=odds_cols, index=index_cols, columns=[*EVENT_COLS, 'provider'], aggfunc='first')
        keep = [
            (col, event_status, event_time, provider)
            for col, event_status, event_time, provider in O.columns
            if event_status in self.odds_schema_.col_metadata(col)['include']
        ]
        O = O[keep]
        cols = pd.DataFrame(O.columns.tolist(), columns=['col', 'event_status', 'event_time', 'provider'])
        cols = cols.groupby(['col', 'provider'], group_keys=False)[
            ['col', 'event_status', 'event_time', 'provider']
        ].apply(
            lambda group: group.iloc[:1] if self.odds_schema_.col_metadata(group.iloc[0]['col'])['fixed'] else group,
        )
        O = O[list(cols.itertuples(index=False, name=None))]
        O.columns = [
            (
                col
                if self.odds_schema_.col_metadata(col)['fixed']
                else odds_column(provider, col, event_status, event_time)
            )
            for col, event_status, event_time, provider in O.columns
        ]
        return O

    def _extract(
        self: Self,
        stats: pd.DataFrame,
        odds: pd.DataFrame,
        target_event_status: str,
        target_event_time: pd.Timedelta,
        input_event_status: str | None = None,
        input_event_time: pd.Timedelta | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build aligned, date-indexed ``X`` and ``O`` for the given snapshots."""
        horizon = (target_event_status, target_event_time, input_event_status, input_event_time)
        X = self._pivot_features(stats[self._feature_mask(stats, *horizon)])
        O = self._pivot_odds(odds[self._feature_mask(odds, *horizon)])
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
        targets = stats.loc[mask, self._identity_cols() + self.targets_].set_index(self._identity_cols())
        targets = targets.reindex(index)
        columns_mapping = {
            target: target_column(target, target_event_status, target_event_time) for target in self.targets_
        }
        return targets.rename(columns=columns_mapping)

    def _resolve_params(
        self: Self,
        learning_type: str | None,
        target_event_status: str | None,
        target_event_time: pd.Timedelta | None,
        input_event_status: str | None,
        input_event_time: pd.Timedelta | None,
    ) -> tuple[str, str, pd.Timedelta]:
        """Validate and default the learning type, target moment and input horizon."""
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

        check_scalar(input_event_status, 'input_event_status', (NoneType, str))
        if input_event_status is not None and input_event_status not in INPUT_EVENT_STATUSES:
            msg = (
                f'Invalid input event status. It should be one of {INPUT_EVENT_STATUSES}. '
                f'Got {input_event_status} instead.'
            )
            raise ValueError(msg)
        self.input_event_status_ = input_event_status
        check_scalar(input_event_time, 'input_event_time', (NoneType, pd.Timedelta))
        if input_event_time is not None and input_event_time < pd.Timedelta('0min'):
            msg = 'The event time should be positive.'
            raise ValueError(msg)
        self.input_event_time_ = input_event_time if input_event_time is not None else pd.Timedelta('0min')

        return self.learning_type_, self.target_event_status_, self.target_event_time_

    def extract_train_data(
        self: Self,
        *,
        drop_na_thres: float = 0.0,
        odds_type: str | None = None,
        learning_type: str | None = None,
        target_event_status: str | None = None,
        target_event_time: pd.Timedelta | None = None,
        input_event_status: str | None = None,
        input_event_time: pd.Timedelta | None = None,
    ) -> TrainData:
        """Extract the moment-aware training data.

        Read more in the [user guide][user-guide].

        It returns historical data that can be used to create a betting strategy
        based on heuristics or machine learning models. The method prepares data
        for the prediction target defined by `target_event_status` and
        `target_event_time`. Every snapshot before the target becomes features
        (X) — optionally capped at an input horizon — the target-moment outcomes
        become labels (Y), and the corresponding betting odds become O.

        Args:
            drop_na_thres:
                Threshold in `[0.0, 1.0]` controlling how aggressively feature
                columns with missing values are dropped. `0.0` keeps all columns.
            odds_type:
                One of `get_odds_types()`. `None` returns no odds.
            learning_type:
                `'supervised'` (default): `Y` holds the target outcomes.
                `'unsupervised'`: `Y` is `None` (features and odds only).
            target_event_status:
                `'inplay'` or `'postplay'` (default `'postplay'`).
            target_event_time:
                In-play target time (e.g. `pd.Timedelta('60min')`). Defaults to 0.
            input_event_status:
                Latest snapshot status to keep as a feature, one of `'preplay'`,
                `'inplay'`, `'postplay'`. `None` (default) keeps every snapshot
                before the target; e.g. `'preplay'` keeps only pre-match features.
            input_event_time:
                Time of the input horizon (e.g. `pd.Timedelta('45min')`), used
                together with `input_event_status`. Defaults to 0.

        Returns:
            (X, Y, O):
                Moment-aware features `X`, target outcomes `Y` and odds `O`. For
                `learning_type='unsupervised'`, `Y` is `None`. The three components
                share the same date index and rows.
        """
        self._load(odds_type)

        self.stats_schema_.validate(self.stats_)
        self.odds_schema_.validate(self.odds_)
        if self.stats_schema_.snapshot_cols() != self.odds_schema_.snapshot_cols():
            msg = 'Stats and odds snapshots columns do not match.'
            raise AssertionError(msg)

        # Check if inplay or postplay events exist
        event_statuses = [status for status in self.stats_['event_status'].unique() if status != 'preplay']
        if not event_statuses:
            msg = 'No `inplay` or `postplay` events were found.'
            raise ValueError(msg)

        # Validate and resolve the learning type, target moment and input horizon
        learning_type, target_event_status, target_event_time = self._resolve_params(
            learning_type,
            target_event_status,
            target_event_time,
            input_event_status,
            input_event_time,
        )

        # Split matches into those resolvable at the target moment (training) and the rest (fixtures)
        index_cols = self._identity_cols()
        target_mask = (self.stats_['event_status'] == target_event_status) & (
            self.stats_['event_time'] == target_event_time
        )
        train_ids = pd.MultiIndex.from_frame(self.stats_.loc[target_mask, index_cols])
        if train_ids.empty:
            msg = 'No resolvable events were found for the requested target moment.'
            raise ValueError(msg)
        self._train_ids = train_ids
        train_mask = pd.MultiIndex.from_frame(self.stats_[index_cols]).isin(train_ids)
        train_odds_mask = pd.MultiIndex.from_frame(self.odds_[index_cols]).isin(train_ids)

        # Extract input and odds data
        X, O = self._extract(
            self.stats_[train_mask],
            self.odds_[train_odds_mask],
            target_event_status,
            target_event_time,
            self.input_event_status_,
            self.input_event_time_,
        )
        X = self._apply_drop_na(X, drop_na_thres)
        self.odds_cols_ = O.columns
        if learning_type == 'unsupervised':
            self.output_cols_ = None
            return X, None, O

        # Extract output data
        Y = self._extract_targets(
            self.stats_[train_mask],
            pd.MultiIndex.from_frame(X.reset_index()[index_cols]),
            target_event_status,
            target_event_time,
        )
        Y.index = X.index
        self.output_cols_ = Y.columns
        return X, Y, O

    def extract_fixtures_data(self: Self) -> FixturesData:
        """Extract the fixtures data.

        Read more in the [user guide][user-guide].

        It returns fixtures data that can be used to make predictions for upcoming
        matches. Before calling this method, `extract_train_data` must have been
        called to fix the columns of the input and odds data. The multi-output
        targets `Y` are always `None` and only included for consistency.

        Returns:
            (X, None, O):
                The fixtures input data `X`, `Y` equal to `None`, and the
                corresponding odds `O`, matching the training columns.
        """
        if not hasattr(self, 'input_cols_'):
            msg = 'The `extract_train_data` method should be called before `extract_fixtures_data`.'
            raise ValueError(msg)
        index_cols = self._identity_cols()
        fixtures_mask = ~pd.MultiIndex.from_frame(self.stats_[index_cols]).isin(self._train_ids)
        fixtures_odds_mask = ~pd.MultiIndex.from_frame(self.odds_[index_cols]).isin(self._train_ids)
        if not fixtures_mask.any():
            X = pd.DataFrame(columns=self.input_cols_, index=pd.DatetimeIndex([], name='date'))
            O = pd.DataFrame(columns=self.odds_cols_, index=pd.DatetimeIndex([], name='date'))
            return X, None, O
        X, O = self._extract(
            self.stats_[fixtures_mask],
            self.odds_[fixtures_odds_mask],
            self.target_event_status_,
            self.target_event_time_,
            self.input_event_status_,
            self.input_event_time_,
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
