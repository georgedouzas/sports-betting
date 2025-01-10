"""Includes base class and functions for data preprocessing and loading."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from difflib import SequenceMatcher
from pathlib import Path
from typing import ClassVar

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_scalar
from typing_extensions import Self

from .. import FixturesData, Outputs, Param, ParamGrid, Schema, TrainData


def _create_names_mapping_table(data_source1: pd.DataFrame, data_source2: pd.DataFrame, keys: list) -> pd.DataFrame:
    def _cols(x: str) -> list[str]:
        return [f'{col}_team{x}' for col in ('home', 'away')]

    # Generate teams names combinations
    names_combinations = data_source1[keys + _cols('')].merge(data_source2[keys + _cols('')], on=keys)

    # Calculate similarity index
    similarity = names_combinations.apply(
        lambda row: SequenceMatcher(None, row.home_team_x, row.home_team_y).ratio()
        + SequenceMatcher(None, row.away_team_x, row.away_team_y).ratio(),
        axis=1,
    )

    # Append similarity index
    names_combinations_similarity = pd.concat([names_combinations[_cols('_x') + _cols('_y')], similarity], axis=1)

    # Filter correct matches
    indices = names_combinations_similarity.groupby(_cols('_x')).iloc[0].idxmax().to_numpy()
    names_matching = names_combinations.take(indices)

    # Teams matching
    matching1 = names_matching.loc[:, ['home_team_x', 'home_team_y']].drop_duplicates()
    matching2 = names_matching.loc[:, ['away_team_x', 'away_team_y']].drop_duplicates()
    matching1.columns = matching2.columns = cols = ['team1', 'team2']
    matching = pd.concat([matching1, matching2])
    similarity = matching.apply(lambda row: SequenceMatcher(None, row.team1, row.team2).ratio(), axis=1)
    names_combinations_similarity = pd.concat([matching, similarity], axis=1).reset_index(drop=True)
    indices = names_combinations_similarity.groupby('team1').iloc[0].idxmax()
    names_mapping = names_combinations_similarity.take(indices)[cols].reset_index(drop=True)

    return names_mapping


class BaseDataLoader(metaclass=ABCMeta):
    """The base class for dataloaders.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    SCHEMA: ClassVar[Schema] = []
    OUTPUTS: ClassVar[Outputs] = []

    def __init__(self: Self, param_grid: ParamGrid | None = None) -> None:
        self.param_grid = param_grid

    @classmethod
    @abstractmethod
    def _get_full_param_grid(cls: type[BaseDataLoader]) -> ParameterGrid:
        return ParameterGrid([])

    @abstractmethod
    def _get_data(self: Self) -> pd.DataFrame:
        return pd.DataFrame()

    @staticmethod
    def _cols(data: pd.DataFrame, col_type: str) -> list[str]:
        if col_type == 'input':
            return [col for col in data.columns if not col.startswith('target')]
        return [col for col in data.columns if col.startswith(col_type)]

    def _check_param_grid(self: Self) -> Self:
        """Check the parameters grid."""
        full_param_grid = self._get_full_param_grid()
        if self.param_grid is not None:
            full_param_grid_df = self._convert_data_types(pd.DataFrame(full_param_grid))

            # False names
            param_grid_df = self._convert_data_types(pd.DataFrame(ParameterGrid(self.param_grid)))
            available_names = set(full_param_grid_df.columns)
            names = set(param_grid_df.columns)
            if not available_names.issuperset(names):
                error_msg = (
                    'Parameter grid includes the parameters name(s) '
                    f'{list(names.difference(available_names))} that are not not '
                    'allowed by available data.',
                )
                raise ValueError(error_msg)

            # False values
            param_grid = []
            for params in ParameterGrid(self.param_grid):
                params_df = self._convert_data_types(
                    pd.DataFrame({name: [value] for name, value in params.items()}),
                ).merge(full_param_grid_df)
                if params_df.size == 0:
                    error_msg = (
                        'Parameter grid includes the parameters value(s) '
                        f'{params} that are not allowed by available data.',
                    )
                    raise ValueError(error_msg)
                params_df = pd.DataFrame(params_df).merge(full_param_grid_df)
                param_grid.append(params_df)
            param_grid_df = pd.concat(param_grid, ignore_index=True)
            self.param_grid_ = ParameterGrid(
                [{k: [v] for k, v in params.to_dict().items()} for _, params in param_grid_df.iterrows()],
            )
        else:
            self.param_grid_ = full_param_grid
        return self

    def _convert_data_types(self: Self, data: pd.DataFrame) -> pd.DataFrame:
        """Cast the data type of columns."""
        data_types = {data_type for _, data_type in self.SCHEMA}
        for data_type in data_types:
            converted_cols = list(
                {
                    col
                    for col, selected_data_type in self.SCHEMA
                    if selected_data_type is data_type and col in data.columns
                },
            )
            if converted_cols:
                data_converted_cols = data[converted_cols]
                if data_type is float or data_type is np.int64:
                    data_converted_cols = data_converted_cols.replace('-', np.nan)
                    data_converted_cols = data_converted_cols.infer_objects().fillna(
                        -1 if data_type is np.int64 else np.nan,
                    )
                data[converted_cols] = (
                    data_converted_cols.to_numpy().astype(data_type)
                    if data_type is not np.datetime64
                    else pd.to_datetime(data_converted_cols.iloc[:, 0])
                )
        return data

    def _validate_data(self: Self) -> pd.DataFrame:
        """Validate the data."""
        data = self._get_data()
        if not isinstance(data, pd.DataFrame):
            error_msg = f'Data should be a pandas dataframe. Got {type(data).__name__} instead.'
            raise TypeError(error_msg)
        if data.size == 0:
            error_msg = 'Data should be a pandas dataframe with positive size.'
            raise ValueError(error_msg)
        if 'fixtures' not in data.columns or data['fixtures'].dtype.name != 'bool':
            error_msg = (
                'Data should include a boolean column `fixtures` to distinguish between train and fixtures data.'
            )
            raise KeyError(error_msg)
        if 'date' not in data.columns or data['date'].dtype.name != 'datetime64[ns]':
            error_msg = 'Data should include a datetime column `date` to represent the date.'
            raise KeyError(error_msg)
        if self.SCHEMA and not {col for col, _ in self.SCHEMA}.issuperset(data.columns.difference(['fixtures'])):
            error_msg = 'Data contains columns not included in schema.'
            raise ValueError(error_msg)

        # Reorder columns
        data = data[[col for col, _ in self.SCHEMA if col in data.columns] + ['fixtures']]

        # Set date as index
        data = data.set_index('date').sort_values('date')

        # Remove missing values of data
        data = data[~data.index.isna()]

        # Check consistency with available parameters
        mask = data['fixtures']
        train_data = data[~mask].drop(columns=['fixtures'])
        full_param_grid_df = pd.DataFrame(self._get_full_param_grid())
        param_grid_df = full_param_grid_df[
            [col for col in full_param_grid_df.columns if col in train_data.columns]
        ].drop_duplicates()
        train_data_params = train_data[param_grid_df.columns].drop_duplicates().reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(train_data_params.merge(param_grid_df), train_data_params, check_dtype=False)
        except AssertionError as e:
            error_msg = 'The raw data and available parameters are incompatible.'
            raise ValueError(error_msg) from e

        return data

    def _extract_train_data(self: Self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[~data['fixtures']].drop(columns=['fixtures'])
        data = data.reset_index().merge(pd.DataFrame(self.param_grid_)).set_index('date').sort_index()
        return data

    def _check_dropped_na_cols(self: Self, data: pd.DataFrame, drop_na_thres: float) -> Self:
        thres = int(data.shape[0] * drop_na_thres)
        dropped_all_na_cols = data.columns.difference(data.dropna(axis=1, how='all').columns)
        dropped_thres_na_cols = data.columns.difference(data.dropna(axis=1, thresh=thres).columns)
        dropped_na_cols = dropped_all_na_cols.union(dropped_thres_na_cols)
        self.dropped_na_cols_ = pd.Index(
            [col for col in data.columns if col in dropped_na_cols and col not in self._cols(data, 'target')],
            dtype=object,
        )
        if self._cols(data, 'input') == self.dropped_na_cols_.tolist():
            error_msg = 'All input columns were removed. Set `drop_na_thres` parameter to a lower value.'
            raise ValueError(error_msg)
        return self

    def extract_train_data(
        self: Self,
        drop_na_thres: float = 0.0,
        odds_type: str | None = None,
    ) -> TrainData:
        """Extract the training data.

        Read more in the [user guide][dataloader].

        It returns historical data that can be used to create a betting
        strategy based on heuristics or machine learning models.

        The data contain information about the matches that belong
        in two categories. The first category includes any information
        known before the start of the match, i.e. the training data `X`
        and the odds data `O`. The second category includes the outcomes of
        matches i.e. the multi-output targets `Y`.

        The method selects only the the data allowed by the `param_grid`
        parameter of the initialization method. Additionally, columns with missing
        values are dropped through the `drop_na_thres` parameter, while the
        types of odds returned is defined by the `odds_type` parameter.

        Args:
            drop_na_thres:
                The threshold that specifies the input columns to drop. It is a float in
                the `[0.0, 1.0]` range. Higher values result in dropping more values.
                The default value `drop_na_thres=0.0` keeps all columns while the
                maximum value `drop_na_thres=1.0` keeps only columns with non
                missing values.

            odds_type:
                The selected odds type. It should be one of the available odds columns
                prefixes returned by the method `get_odds_types`. If `odds_type=None`
                then no odds are returned.

        Returns:
            (X, Y, O):
                Each of the components represent the training input data `X`, the
                multi-output targets `Y` and the corresponding odds `O`, respectively.
        """

        # Check param grid
        self._check_param_grid()

        # Validate the data
        data = self._validate_data()

        # Extract train data
        data = self._extract_train_data(data)

        # Check dropped columns
        self.drop_na_thres_ = check_scalar(drop_na_thres, 'drop_na_thres', float, min_val=0.0, max_val=1.0)
        self._check_dropped_na_cols(data, drop_na_thres)

        # Check odds type
        dropped_all_na_cols = data.columns.difference(data.dropna(axis=1, how='all').columns)
        odds_types = sorted({col.split('__')[1] for col in self._cols(data, 'odds') if col not in dropped_all_na_cols})
        if odds_type is not None and odds_type not in odds_types:
            error_msg = (
                f'Parameter `odds_type` should be a prefix of available odds columns. Got `{odds_type}` instead.'
            )
            if isinstance(odds_type, str):
                raise ValueError(error_msg)
            else:
                raise TypeError(error_msg)
        self.odds_type_ = odds_type

        # Extract input, odds and output columns
        output_keys = [col.split('__')[1:] for col, _ in self.OUTPUTS]
        target_keys = [col.split('__')[2:] for col in self._cols(data, 'target')]
        odds_keys = [col.split('__')[2:] for col in self._cols(data, 'odds') if col.split('__')[1] == self.odds_type_]
        output_keys = [
            key
            for key in (odds_keys if self.odds_type_ is not None else output_keys)
            if key in output_keys and key[-1:] in target_keys
        ]
        target_output_keys = list({key for *_, key in output_keys})
        self.input_cols_ = pd.Index(
            [col for col in self._cols(data, 'input') if col not in self.dropped_na_cols_],
            dtype=object,
        )
        self.odds_cols_ = pd.Index(
            [f'odds__{self.odds_type_}__{key1}__{key2}' for key1, key2 in output_keys if self.odds_type_ is not None],
            dtype=object,
        )
        self.output_cols_ = pd.Index(
            [f'output__{key1}__{key2}' for key1, key2 in output_keys if key2 in target_output_keys],
            dtype=object,
        )
        self.target_cols_ = pd.Index(
            [col for col in self._cols(data, 'target') if col.split('__')[-1] in target_output_keys],
            dtype=object,
        )

        # Remove missing target data
        data = data.dropna(subset=self.target_cols_, how='any')

        # Convert data types
        data = self._convert_data_types(data)

        # Extract outputs
        Y_train = []
        outputs_mapping = dict(self.OUTPUTS)
        for col in self.output_cols_:
            func = outputs_mapping[col]
            Y_train.append(pd.Series(func(data[self.target_cols_]), name=col))
        Y_train = pd.concat(Y_train, axis=1).reset_index(drop=True)

        # Extract odds
        O_train = data[self.odds_cols_].reset_index(drop=True) if self.odds_type_ is not None else None

        self.train_data_ = data[self.input_cols_], Y_train, O_train
        if hasattr(self, 'fixtures_data_'):
            delattr(self, 'fixtures_data_')

        return self.train_data_

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
        # Extract fixtures data
        if not hasattr(self, 'train_data_'):
            error_msg = 'Extract the training data before extracting the fixtures data.'
            raise AttributeError(error_msg)

        data = self._validate_data()

        # Extract fixtures data
        data = data[data['fixtures']].drop(columns=['fixtures'])

        # Convert data types
        data = self._convert_data_types(data)

        # Remove past data
        data = data.loc[data.index >= pd.Timestamp(pd.to_datetime('today').date())]

        # Extract odds
        O_fix = data[self.odds_cols_].reset_index(drop=True) if self.odds_type_ is not None else None

        self.fixtures_data_ = data[self.input_cols_], None, O_fix

        return self.fixtures_data_

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

    @classmethod
    def get_all_params(cls: type[BaseDataLoader]) -> list[Param]:
        """Get the available parameters.

        It can be used to get the allowed names and values for the
        `param_grid` parameter of the dataloader object.

        Returns:
            param_grid: list
                A list of all allowed params and values.
        """
        full_param_grid = cls._get_full_param_grid()
        params_names = sorted({param_name for params in full_param_grid for param_name in params})
        all_params = sorted(
            full_param_grid,
            key=lambda params: tuple(
                params.get(name, '' if dict(cls.SCHEMA)[name] is object else 0) for name in params_names
            ),
        )
        return all_params

    def get_odds_types(self: Self) -> list[str]:
        """Get the available odds types.

        It can be used to get the allowed odds types of the dataloader's method
        `extract_train_data`.

        Returns:
            odds_types:
                A list of available odds types.
        """
        # Check param grid
        self._check_param_grid()

        # Validate the data
        data = self._validate_data()

        # Extract train data
        data = self._extract_train_data(data)

        # Drop columns with only missing values
        dropped_all_na_cols = data.columns.difference(data.dropna(axis=1, how='all').columns)

        return sorted({col.split('__')[1] for col in self._cols(data, 'odds') if col not in dropped_all_na_cols})


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
