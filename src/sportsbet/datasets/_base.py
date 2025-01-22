"""Includes base class and functions for data preprocessing and loading."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path

import cloudpickle
import polars as pl
import pytz
from typing_extensions import Self

from .. import FixturesData, OutputsMapping, TrainData


class BaseDataLoader(metaclass=ABCMeta):
    """The base class for dataloaders.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self: Self, prediction_stage: str | None = None) -> None:
        self.prediction_stage = prediction_stage

    @property
    @abstractmethod
    def _outputs_mapping(self: Self) -> OutputsMapping:
        return {}

    @property
    @abstractmethod
    def _required_cols(self: Self) -> list[str]:
        return []

    @property
    @abstractmethod
    def _stages(self: Self) -> list[str]:
        return []

    @abstractmethod
    def _get_stats_data(self: Self) -> pl.DataFrame:
        return pl.DataFrame()

    @abstractmethod
    def _get_odds_data(self: Self) -> pl.DataFrame:
        return pl.DataFrame()

    def _validate_stats_data(self: Self, stats_data: pl.DataFrame) -> pl.DataFrame:
        if not isinstance(stats_data, pl.DataFrame):
            error_msg = f'Statistics data should be a polars dataframe. Got {type(stats_data).__name__} instead.'
            raise TypeError(error_msg)
        if stats_data.is_empty():
            error_msg = 'Statistics data should be a polars dataframe with positive size.'
            raise ValueError(error_msg)
        if 'date' not in stats_data.columns or stats_data['date'].dtype != pl.Date:
            error_msg = 'Statistics data should include a date column `date` to represent the date.'
            raise KeyError(error_msg)
        if 'stage' not in stats_data.columns:
            error_msg = 'Statistics data should include a stage column `stage` to represent the event stage.'
            raise KeyError(error_msg)
        for col in self._required_cols:
            if col not in stats_data.columns:
                error_msg = f'Statistics data should include a required column `{col}`.'
                raise KeyError(error_msg)
        initial_cols = ['date', 'stage', *self._required_cols]
        remaining_cols = [col for col in stats_data.columns if col not in initial_cols]
        stats_data = stats_data[initial_cols + remaining_cols]
        return stats_data

    def _validate_odds_data(self: Self, odds_data: pl.DataFrame) -> pl.DataFrame:  # noqa: C901
        if not isinstance(odds_data, pl.DataFrame):
            error_msg = f'Odds data should be a polars dataframe. Got {type(odds_data).__name__} instead.'
            raise TypeError(error_msg)
        if odds_data.is_empty():
            error_msg = 'Odds data should be a polars dataframe with positive size.'
            raise ValueError(error_msg)
        if 'datetime' not in odds_data.columns or odds_data['datetime'].dtype != pl.Datetime:
            error_msg = 'Odds data should include a datetime column `datetime` to represent the date and time.'
            raise KeyError(error_msg)
        if 'stage' not in odds_data.columns:
            error_msg = 'Odds data should include a stage column `stage` to represent the event stage.'
        if 'provider' not in odds_data.columns:
            error_msg = 'Odds data should include a column `provider`.'
            raise KeyError(error_msg)
        if 'market' not in odds_data.columns:
            error_msg = 'Odds data should include a column `market`.'
            raise KeyError(error_msg)
        if 'value' not in odds_data.columns:
            error_msg = 'Odds data should include a column `value`.'
            raise KeyError(error_msg)
        for col in self._required_cols:
            if col not in odds_data.columns:
                error_msg = f'Odds data should include a required column `{col}`.'
                raise KeyError(error_msg)
        markets = {market for cols in self._outputs_mapping.values() for market in cols}
        for market in odds_data['market'].unique():
            if market not in markets:
                error_msg = f'Betting market {market} is not supported.'
                raise ValueError(error_msg)
        odds_data = odds_data[['datetime', 'stage', *self._required_cols, 'provider', 'market', 'value']]
        return odds_data

    def _extract_data(self: Self) -> pl.DataFrame:
        # Get the data
        stats_data = self._get_stats_data()
        odds_data = self._get_odds_data()

        # Check and set stage
        available_stages = set(stats_data['stage']).intersection(odds_data['stage'])
        if not available_stages:
            error_msg = 'Statistics data and odds data do not include any common game stages.'
            raise ValueError(error_msg)
        elif not available_stages.intersection(self._stages):
            error_msg = 'Data include not supported game stages.'
        if self.prediction_stage is None:
            self.prediction_stage_ = [stage for stage in self._stages if stage in available_stages][-1]
        else:
            self.prediction_stage_ = self.prediction_stage
        if self.prediction_stage_ not in available_stages:
            error_msg = (
                f'Parameter `prediction_stage` should be one of {", ".join(available_stages)} or None'
                f'. Got {self.prediction_stage} instead.'
            )
            raise ValueError(error_msg)
        mask = stats_data['stage'].map_elements(
            lambda stage: self._stages.index(stage) <= self._stages.index(self.prediction_stage_),
            return_dtype=bool,
        )
        stats_data = stats_data.filter(mask)
        mask = odds_data['stage'].map_elements(
            lambda stage: self._stages.index(stage) <= self._stages.index(self.prediction_stage_),
            return_dtype=bool,
        )
        odds_data = odds_data.filter(mask)

        # Validate the data
        stats_data = self._validate_stats_data(stats_data)
        odds_data = self._validate_odds_data(odds_data)

        # Remove missing values
        stats_data = stats_data.drop_nulls(subset=['date', *self._required_cols])
        odds_data = odds_data.drop_nulls(subset=['datetime', *self._required_cols])

        # Statistics data
        markets = odds_data['market'].unique()
        outputs_cols = []
        for cols, mappings in self._outputs_mapping.items():
            if set(stats_data.columns).issuperset(cols):
                for market, func in mappings.items():
                    if market in markets:
                        output = func(stats_data)
                        output = output.rename(market)
                        stats_data = stats_data.with_columns(output)
                        outputs_cols.append(market)
            stats_data = stats_data.drop(cols)
        stats_data = stats_data.pivot('stage', values=outputs_cols, separator='__')

        # Odds data
        odds_data = odds_data.pivot(['provider', 'market', 'stage'], values='value')

        # Join data
        data = stats_data.join(odds_data, how='inner', on=self._required_cols)
        mask = data[['date']] - data[['datetime']] < pl.duration(days=4)
        data = data.filter(mask.to_series())

        # Sort data
        data = data.sort('datetime')

        return data

    def _extract_modelling_data(self: Self, data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        Y = data[[col for col in data.columns if col.endswith(f'__{self.prediction_stage_}')]]
        O = data[[col for col in data.columns if col.startswith('{') and col.endswith('}')]]
        X = data[[col for col in data.columns if col not in Y.columns + O.columns]]
        X_cols_initial = ['date', 'datetime', *self._required_cols]
        X = X[[*X_cols_initial, *set(X.columns).difference(X_cols_initial)]]
        O = O.rename(lambda name: name.removeprefix('{"').removesuffix('"}').replace('","', '__'))
        return X, Y, O

    def extract_train_data(self: Self) -> TrainData:
        """Extract the training data.

        Read more in the [user guide][dataloader].

        It returns historical data that can be used to create a betting
        strategy based on heuristics or machine learning models.

        The data contain information about the matches that belong
        in two categories. The first category includes any information
        known before the start of the match, i.e. the input data `X`
        and the odds data `O`. The second category includes the outcomes of
        matches i.e. the multi-output targets `Y`.

        The method selects only the the data allowed by the `param_grid`
        parameter of the initialization method.

        Returns:
            (X, Y, O):
                Each of the components represent the training input data `X`, the
                multi-output targets `Y` and the corresponding odds `O`, respectively.
        """
        # Extract data
        data = self._extract_data()

        # Keep past data
        data = data.filter(data['date'] < datetime.now(tz=pytz.utc).date())

        # Extract input, output and odds data
        X, Y, O = self._extract_modelling_data(data)

        # Remove output's data missing values from data
        Y = Y.with_row_index().drop_nulls()
        X = X[Y['index']]
        O = O[Y['index']]
        Y = Y[:, 1:]

        return X, Y, O

    def extract_fixtures_data(self: Self) -> FixturesData:
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
        # Extract data
        data = self._extract_data()

        # Keep fixtures data
        data = data.filter(data['date'] >= datetime.now(tz=pytz.utc).date())

        # Extract input, output and odds data
        X, Y, O = self._extract_modelling_data(data)

        return X, Y, O

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
