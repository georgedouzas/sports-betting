"""Implements the base dataloader class shared by all dataloaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import NoneType
from typing import Self

import cloudpickle
import pandas as pd
from sklearn.utils import check_scalar

from .. import FixturesData, ParamGrid, TrainData
from ..sources._schema import (
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
DAY = pd.Timedelta('1D')
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


class BaseDataLoader(ABC):
    """The abstract base class for dataloaders.

    A dataloader reads long event-snapshot `stats` and `odds` data, validates it,
    derives the available providers, markets and per-column metadata from the data
    itself, and extracts moment-aware training and fixtures data. Everything but the
    data source is implemented here; a concrete dataloader only needs to implement
    the abstract [`_snapshots`][sportsbet.dataloaders.BaseDataLoader] method and, when
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

    Examples:
        >>> import pandas as pd
        >>> from sportsbet.dataloaders import BaseDataLoader
        >>>
        >>> identity = {'date': pd.Timestamp('2024-08-16', tz='UTC'), 'league': 'England', 'division': 1,
        ...             'year': 2025, 'home_team': 'A', 'away_team': 'B'}
        >>>
        >>> class MyDataLoader(BaseDataLoader):
        ...     'A dataloader of data that is already on your machine.'
        ...
        ...     def _snapshots(self):
        ...         stats = pd.DataFrame([
        ...             {**identity, 'event_status': 'preplay', 'event_time': pd.Timedelta(0), 'home_form': 1.0},
        ...             {**identity, 'event_status': 'postplay', 'event_time': pd.Timedelta(0), 'home_win': 1},
        ...         ])
        ...         odds = pd.DataFrame([
        ...             {**identity, 'event_status': 'preplay', 'event_time': pd.Timedelta(0),
        ...              'provider': 'acme', 'home_win': 2.5},
        ...         ])
        ...         return stats, odds
        >>>
        >>> dataloader = MyDataLoader()
        >>> # The providers and the markets are derived from the data, so nothing has to be registered.
        >>> dataloader.get_odds_types()
        ['acme']
        >>> X, Y, O = dataloader.extract_train_data(odds_type='acme')
        >>> list(Y.columns)
        ['home_win__postplay__0min']
        >>> list(O.columns)
        ['acme__home_win__preplay__0min']
    """

    def __init__(self: Self, param_grid: ParamGrid | None = None) -> None:
        self.param_grid = param_grid

    @abstractmethod
    def _snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the long training `stats`/`odds` snapshots.

        Every dataloader implements this. A dataloader backed by sources downloads them; one carrying its own data
        returns it.
        """

    def _fixtures_snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the long `stats`/`odds` snapshots of the upcoming matches.

        The default is the training snapshots, which is right for a dataloader carrying its own data: the fixtures are
        the unplayed matches already in it. A dataloader backed by sources overrides it to download the current data the
        upcoming matches need.
        """
        return self._snapshots()

    @property
    def sources(self: Self) -> tuple:
        """The data sources of the dataloader.

        A source says what it publishes, which is where a `param_grid` starts. It is empty for a dataloader carrying its
        own data.
        """
        return ()

    def _all_params(self: Self) -> list[dict]:
        """Return the combinations the sources publish, used to filter `param_grid`.

        A `param_grid` names what to select, and the source's `available_params` says what there is to select. A
        dataloader carrying its own data has no catalogue to answer from.
        """
        msg = f'{type(self).__name__} carries its own data, so it publishes no catalogue of parameters.'
        raise NotImplementedError(msg)

    def _filter_params(self: Self, params: list[dict]) -> list[dict]:
        """Filter the available combinations by `param_grid`, keeping only combinations that exist."""
        if self.param_grid is None:
            return params
        grids = self.param_grid if isinstance(self.param_grid, list) else [self.param_grid]
        return [
            combination
            for combination in params
            if any(all(combination[key] in values for key, values in grid.items()) for grid in grids)
        ]

    @staticmethod
    def _upcoming(data: pd.DataFrame) -> pd.Series:
        """Return which snapshots belong to a match that has not been played yet.

        A match counts as upcoming by its date, which lies in the future. A missing result is a separate matter: a feed
        sometimes drops one — an abandoned game, a season it never finished recording — and those matches stay in the
        past. Dating from the future keeps the fixtures to the matches genuinely still to come.
        """
        return data['date'] >= pd.Timestamp.now(tz='UTC')

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
        if not [col for col in odds.columns if col not in EVENT_COLS + IDENTITY_COLS + ['provider']]:
            odds = self.no_odds()
        odds = self._finalize(odds)
        self._validate_snapshots(stats, odds)
        providers = sorted(odds['provider'].dropna().unique().tolist())
        if odds_type is not None and odds_type not in providers:
            msg = f'Invalid odds type. It should be one of {providers}. Got {odds_type} instead.'
            raise ValueError(msg)
        stats_value_cols = [col for col in stats.columns if col not in EVENT_COLS + IDENTITY_COLS]
        odds_value_cols = [col for col in odds.columns if col not in EVENT_COLS + IDENTITY_COLS + ['provider']]
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

    @staticmethod
    def no_odds() -> pd.DataFrame:
        """Return the odds of a dataloader that has none: the right shape, and no rows.

        An extraction that asks for targets is told there are no markets to predict, and the features remain available
        on their own.
        """
        odds = pd.DataFrame(columns=[*EVENT_COLS, *IDENTITY_COLS, 'provider'])
        odds['date'] = pd.to_datetime(odds['date'], utc=True).astype('datetime64[ns, UTC]')
        odds['event_time'] = pd.to_timedelta(odds['event_time']).astype('timedelta64[ns]')
        for col in ('event_status', 'league', 'home_team', 'away_team', 'provider'):
            odds[col] = odds[col].astype('string[pyarrow]')
        for col in ('division', 'year'):
            odds[col] = odds[col].astype('int64')
        return odds

    def _apply_drop_na(self: Self, X: pd.DataFrame, drop_na_thres: float) -> pd.DataFrame:
        """Drop feature columns whose missingness exceeds `drop_na_thres`.

        It is a proportion in `[0.0, 1.0]`, checked to lie in that range.
        """
        check_scalar(drop_na_thres, 'drop_na_thres', (int, float), min_val=0.0, max_val=1.0)
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

        A snapshot is kept when it is strictly before the target moment and, when an input horizon is given, at or
        before it.
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
        """Pivot long snapshots into wide, moment-aware feature columns.

        Every match is kept, even one whose features are all missing. The first round of a season has no form behind it,
        so its features are empty, and the pivot reindexes onto the full set of matches to keep it: it has two teams, a
        date and a price, and is bettable.
        """
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
        matches = stats[index_cols].drop_duplicates().sort_values(index_cols).set_index(index_cols).index
        return X.reindex(matches)

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

    def _bet_moment(
        self: Self,
        odds: pd.DataFrame,
        target_event_status: str,
        target_event_time: pd.Timedelta,
    ) -> tuple[str, pd.Timedelta] | None:
        """Return the latest moment the odds price, which is the moment a bet is placed."""
        priced = odds.loc[self._feature_mask(odds, target_event_status, target_event_time), list(EVENT_COLS)]
        priced = priced.drop_duplicates()
        if priced.empty:
            return None
        latest = priced.loc[priced['event_status'].map(STATUS_RANK).mul(DAY).add(priced['event_time']).idxmax()]
        return str(latest['event_status']), pd.Timedelta(latest['event_time'])

    def _extract(
        self: Self,
        stats: pd.DataFrame,
        odds: pd.DataFrame,
        target_event_status: str,
        target_event_time: pd.Timedelta,
        input_event_status: str | None = None,
        input_event_time: pd.Timedelta | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build aligned, date-indexed ``X`` and ``O`` for the given snapshots.

        A bet is placed at the moment its odds are quoted, so the features come from that moment or earlier, the ones
        the bettor actually had. The features stop where the odds do, unless a horizon says otherwise.
        """
        bet = self._bet_moment(odds, target_event_status, target_event_time)
        if input_event_status is None:
            if bet is not None:
                input_event_status, input_event_time = bet
        elif bet is not None:
            wanted = (STATUS_RANK[input_event_status], input_event_time or pd.Timedelta('0min'))
            priced = (STATUS_RANK[bet[0]], bet[1])
            if wanted > priced:
                msg = (
                    f'The features would be taken at {input_event_status} {input_event_time or pd.Timedelta("0min")}, '
                    f'and the odds are quoted at {bet[0]} {bet[1]}. A bet is placed when its odds are quoted, so a '
                    f'feature from any later moment is one the bettor could not have had. Use odds that are quoted at '
                    f'that moment, or take the features no later than the odds.'
                )
                raise ValueError(msg)
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
        if self.learning_type_ == 'supervised' and not self.targets_:
            msg = (
                'There are no odds, so there are no markets to predict: the markets a model learns are the ones its '
                'odds price. Pass an `odds` source, or ask for `learning_type="unsupervised"` to get the features on '
                'their own.'
            )
            raise ValueError(msg)

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

        It downloads the selected seasons and returns the historical data a betting strategy is built and backtested on.
        Every snapshot before the target moment (`target_event_status`, `target_event_time`) becomes a feature in `X` —
        optionally capped at an input horizon — the target-moment outcomes become the labels `Y`, and the odds become
        `O`. Call it again to download the data again; keep it with `save`.

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
        self.odds_type_ = odds_type

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

        # A match is trained on when it is resolvable at the target moment. The training data is the selected
        # seasons, so a match still to be played has no target yet, and the fixtures download it separately.
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

        # A supervised model cannot be fitted on a target it does not have. scikit-learn does not accept a missing
        # value in `y`, and a match whose outcome the feed never recorded has no target to learn from, so it is dropped
        # rather than imputed: an invented outcome is a match that never happened.
        labelled = Y.notna().all(axis=1)
        return X[labelled], Y[labelled], O[labelled]

    def extract_fixtures_data(self: Self) -> FixturesData:
        """Extract the fixtures data.

        Read more in the [user guide][user-guide].

        A fixture is a match that has not been played yet. This downloads the upcoming matches of the selected leagues
        and returns them shaped exactly like the training data, so the model trained on the history bets on the
        fixtures. The two share their columns, not their contents: `param_grid` chose the seasons to train on, and a
        match still to be played is in none of them.

        `extract_train_data` fixes those columns, so it is called first. The multi-output targets `Y` are always `None`,
        and are returned only for consistency.

        Returns:
            (X, None, O):
                The fixtures input data `X`, `Y` equal to `None`, and the
                corresponding odds `O`, matching the training columns.
        """
        if not hasattr(self, 'input_cols_'):
            msg = 'Call `extract_train_data` before `extract_fixtures_data`, since it fixes the columns to match.'
            raise ValueError(msg)
        stats, odds = self._fixtures_snapshots()
        stats = self._finalize(stats)
        odds = self._finalize(odds)
        odds = odds[odds['provider'] == self.odds_type_] if self.odds_type_ is not None else odds.iloc[0:0]

        index_cols = self._identity_cols()
        played = pd.MultiIndex.from_frame(
            stats.loc[
                (stats['event_status'] == self.target_event_status_) & (stats['event_time'] == self.target_event_time_),
                index_cols,
            ],
        )
        fixtures_mask = ~pd.MultiIndex.from_frame(stats[index_cols]).isin(played) & self._upcoming(stats)
        fixtures_odds_mask = ~pd.MultiIndex.from_frame(odds[index_cols]).isin(played) & self._upcoming(odds)
        if not fixtures_mask.any():
            X = pd.DataFrame(columns=self.input_cols_, index=pd.DatetimeIndex([], name='date'))
            O = pd.DataFrame(columns=self.odds_cols_, index=pd.DatetimeIndex([], name='date'))
            return X, None, O
        X, O = self._extract(
            stats[fixtures_mask],
            odds[fixtures_odds_mask],
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

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from sportsbet.dataloaders import DataLoader, load_dataloader
        >>> from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats
        >>> path = str(Path(tempfile.mkdtemp()) / 'dataloader.pkl')
        >>> dataloader = DataLoader(
        ...     param_grid={'league': ['England']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds()
        ... )
        >>> X, Y, O = dataloader.extract_train_data(odds_type='market_average')
        >>> _ = dataloader.save(path)
        >>> # It comes back knowing what it was told, so the fixtures take the shape the training data took.
        >>> loaded = load_dataloader(path)
        >>> loaded.param_grid_ == dataloader.param_grid_
        True
        >>> X_fix, _, O_fix = loaded.extract_fixtures_data()
        >>> list(X_fix.columns) == list(X.columns)
        True
    """
    with Path(path).open('rb') as file:
        dataloader = cloudpickle.load(file)
    return dataloader
