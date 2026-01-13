"""Implements the dataloader class."""

from __future__ import annotations

from pathlib import Path
from types import NoneType
from typing import Literal, Self

import cloudpickle
import pandas as pd
import pandera.pandas as pa
from sklearn.utils import check_scalar


def _format_timedelta(td: pd.Timedelta) -> str:
    """Format timedelta to readable string."""
    total_minutes = int(td.total_seconds() / 60)
    return f'{total_minutes}min'


class BaseDataLoader:
    """The base class for dataloaders."""

    def __init__(
        self: Self,
        stats: pd.DataFrame,
        odds: pd.DataFrame,
        stats_schema: pa.DataFrameModel,
        odds_schema: pa.DataFrameModel,
        targets: list[str],
    ) -> None:
        self.stats = stats
        self.odds = odds
        self.stats_schema = stats_schema
        self.odds_schema = odds_schema
        self.targets = targets

    def extract_train_data(
        self: Self,
        learning_type: Literal['supervised', 'unsupervised', 'reinforcement'] | None = None,
        target_event_status: Literal['inplay', 'postplay'] | None = None,
        target_event_time: pd.Timedelta | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
                The type of learning algorithm:
                    - 'supervised': Returns (X, Y, O) for prediction tasks
                    - 'unsupervised': Returns (X, O) for analysis
                    - 'reinforcement': Returns gym environment for RL agents
            target_event_status:
                Status of the target outcomes to predict:
                    - 'inplay': Target outcomes during the match
                    - 'postplay': Target final outcomes after match
            target_event_time:
                Time point of the target outcomes (e.g., pd.Timedelta('60min')
                for 60 minutes in-play). All data before this time becomes
                features, data at this time becomes targets. None includes all times.

        Returns:
            Supervised learning:
                (X, Y, O): Input features, target outcomes, and corresponding odds
            Unsupervised learning:
                (X, O): Input features and odds for analysis without targets
            Reinforcement learning:
                gym.Env: Gymnasium environment for RL training
        """
        self.stats_schema.validate(self.stats)
        self.odds_schema.validate(self.odds)
        assert (
            self.stats_schema.snapshot_cols() == self.odds_schema.snapshot_cols()
        ), 'Stats and odds snapshots columns do not match.'

        # Check if inplay or postplay events exist
        event_statuses = [status for status in self.stats['event_status'].unique() if status != 'preplay']
        if not event_statuses:
            msg = 'No `inplay` or `postplay` events were found.'
            raise ValueError(msg)

        # Check learning type
        check_scalar(learning_type, 'learning_type', (NoneType, str))
        if learning_type is not None and learning_type not in ('supervised', 'unsupervised', 'reinforcement'):
            msg = f'Invalid learning type. It should be one of (supervised, unsupervised, reinforcement). Got {learning_type} instead.'
            raise ValueError(msg)
        elif learning_type is None:
            learning_type = 'supervised'
        self.learning_type_ = learning_type

        # Check event status
        check_scalar(target_event_status, 'event_status', (NoneType, str))
        if target_event_status is not None and target_event_status not in ('inplay', 'postplay'):
            msg = f'Invalid learning type. It should be one of (inplay, postplay). Got {target_event_status} instead.'
            raise ValueError(msg)
        elif target_event_status is None:
            target_event_status = 'postplay'
        self.target_event_status_ = target_event_status

        # Check event time
        check_scalar(target_event_time, 'event_time', (NoneType, pd.Timedelta))
        if target_event_time is not None and target_event_time < pd.Timedelta('0min'):
            msg = 'The event time should be positive.'
            raise ValueError(msg)
        elif target_event_time is None:
            target_event_time = pd.Timedelta('0min')
        self.target_event_time_ = target_event_time

        # Extract input data
        if target_event_status == 'postplay':
            feature_mask = self.stats['event_status'].isin(['preplay', 'inplay'])
        elif target_event_status == 'inplay':
            feature_mask = (self.stats['event_status'] == 'preplay') | (
                (self.stats['event_status'] == 'inplay') & (self.stats['event_time'] < target_event_time)
            )
        feature_data = self.stats[feature_mask]
        event_cols = ['event_status', 'event_time']
        index_cols = [col for col in self.stats_schema.snapshot_cols() if col not in event_cols]
        feature_cols = [col for col in self.stats.columns if col not in self.stats_schema.snapshot_cols()]
        X = feature_data.pivot_table(
            values=feature_cols,
            index=index_cols,
            columns=event_cols,
            aggfunc='first',
        )
        for col, event_status, event_time in X.columns:
            metadata = self.stats_schema.col_metadata(col)
            if event_status not in metadata['include']:
                X = X.drop(columns=(col, event_status, event_time))
        X_cols = pd.DataFrame(X.columns.tolist(), columns=['col', 'event_status', 'event_time'])
        X_cols_filtered = X_cols.groupby(['col', 'event_status'])[['col', 'event_status', 'event_time']].apply(
            lambda group: group.iloc[:1] if self.stats_schema.col_metadata(group.iloc[0]['col'])['fixed'] else group,
        ).reset_index(drop=True)
        X = X[list(X_cols_filtered.itertuples(index=False, name=None))]
        X.columns = [col if self.stats_schema.col_metadata(col)['fixed'] else f'{col}__{event_status}__{_format_timedelta(event_time)}' for col, event_status, event_time in X.columns]
        X = X.reset_index()

        # Extract odds data
        odd_mask = self.odds['event_status'].isin(['preplay', 'inplay'])
        odds_data = self.odds[odd_mask]
        odds_cols = [col for col in odds_data.columns if col not in self.stats_schema.snapshot_cols() and col != 'provider']
        O = odds_data.pivot_table(
            values=odds_cols,
            index=index_cols,
            columns=[*event_cols, 'provider'],
            aggfunc='first',
        )
        for col, event_status, event_time, provider in O.columns:
            metadata = self.odds_schema.col_metadata(col)
            if event_status not in metadata['include']:
                O = O.drop(columns=(col, event_status, event_time, provider))
        O_cols = pd.DataFrame(O.columns.tolist(), columns=['col', 'event_status', 'event_time', 'provider'])
        O_cols_filtered = O_cols.groupby(['col', 'event_status', 'provider'])[['col', 'event_status', 'event_time', 'provider']].apply(
            lambda group: group.iloc[:1] if self.odds_schema.col_metadata(group.iloc[0]['col'])['fixed'] else group,
        ).reset_index(drop=True)
        O = O[list(O_cols_filtered.itertuples(index=False, name=None))]
        O.columns = [col if self.odds_schema.col_metadata(col)['fixed'] else f'{provider}__{col}__{event_status}__{_format_timedelta(event_time)}' for col, event_status, event_time, provider in O.columns]
        O = O.reset_index()

        # Extract output data
        target_mask = (self.stats['event_status'] == target_event_status) & (self.stats['event_time'] == target_event_time)
        target_data = self.stats.loc[target_mask, self.stats_schema.snapshot_cols() + self.targets]
        columns_mapping = {target: f'{target}__{target_event_status}__{_format_timedelta(target_event_time)}' for target in self.targets}
        Y = target_data[self.targets].rename(
            columns=columns_mapping,
        )

        return X, Y, O

    def extract_fixtures_data(self: Self):
        """Extract the fixtures data.

        Read more in the [user guide][dataloader].

        It returns fixtures data that can be used to make predictions for
        upcoming matches based on a betting strategy.

        Before calling the `extract_fixtures_data` method for
        the first time, the `extract_training_data` should be called, in
        order to match the columns of the input, output and odds data.

        The data contain information about the matches known before the
        start of the match, i.e. the training data `X` and the odds
        data `O`. The multi-output targets `Y` is always equal to `None`
        and are only included for consistency with the method `extract_train_data`.

        The `param_grid` parameter of the initialization method has no effect
        on the fixtures data.

        Returns:
            (X, None, O):
                Each of the components represent the fixtures input data `X`, the
                multi-output targets `Y` equal to `None` and the
                corresponding odds `O`, respectively.
        """
        pass

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
