"""
Includes base class and functions for data preprocessing and loading.
"""

from difflib import SequenceMatcher
from abc import ABCMeta, abstractmethod

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_scalar


def _cols(x):
    return [f'{col}_team{x}' for col in ('home', 'away')]


def _create_names_mapping_table(data_source1, data_source2, keys):
    """Map most similar teams names between two data sources."""

    # Generate teams names combinations
    names_combinations = pd.merge(
        data_source1[keys + _cols('')], data_source2[keys + _cols('')], on=keys
    )

    # Calculate similarity index
    similarity = names_combinations.apply(
        lambda row: SequenceMatcher(None, row.home_team_x, row.home_team_y).ratio()
        + SequenceMatcher(None, row.away_team_x, row.away_team_y).ratio(),
        axis=1,
    )

    # Append similarity index
    names_combinations_similarity = pd.concat(
        [names_combinations[_cols('_x') + _cols('_y')], similarity], axis=1
    )

    # Filter correct matches
    indices = names_combinations_similarity.groupby(_cols('_x'))[0].idxmax().values
    names_matching = names_combinations.take(indices)

    # Teams matching
    matching1 = names_matching.loc[:, ['home_team_x', 'home_team_y']].drop_duplicates()
    matching2 = names_matching.loc[:, ['away_team_x', 'away_team_y']].drop_duplicates()
    matching1.columns = matching2.columns = cols = ['team1', 'team2']
    matching = matching1.append(matching2)
    similarity = matching.apply(
        lambda row: SequenceMatcher(None, row.team1, row.team2).ratio(), axis=1
    )
    names_combinations_similarity = pd.concat(
        [matching, similarity], axis=1
    ).reset_index(drop=True)
    indices = names_combinations_similarity.groupby('team1')[0].idxmax()
    names_mapping = names_combinations_similarity.take(indices)[cols].reset_index(
        drop=True
    )

    return names_mapping


def _combine_odds(odds):
    """Combine the odds of different outcomes."""
    combined_odds = 1 / (1 / odds).sum(axis=1)
    return combined_odds


def _is_odds_col(col):
    return len(col.split('__')) == 3 and col.split('__')[-1] == 'odds'


def _is_output_col(col):
    return len(col.split('__')) == 2


def _is_input_col(col):
    return not _is_output_col(col)


class _BaseDataLoader(metaclass=ABCMeta):
    """The base class for dataloaders.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    SCHEMA = []
    OUTCOMES = []
    PARAMS = ParameterGrid([])

    def __init__(self, param_grid=None):
        self.param_grid = param_grid

    @abstractmethod
    def _get_data(self):
        return pd.DataFrame()

    def _check_param_grid(self):
        """Check the parameters grid."""
        if self.param_grid is not None:
            full_params_grid_df = pd.DataFrame(self.PARAMS)

            # False names
            params_grid_df = pd.DataFrame(ParameterGrid(self.param_grid))
            available_names = set(full_params_grid_df.columns)
            names = set(params_grid_df.columns)
            if not available_names.issuperset(names):
                raise ValueError(
                    'Parameter grid includes the parameters name(s) '
                    f'{list(names.difference(available_names))} that are not not '
                    'allowed by available data.'
                )

            # False values
            param_grid = []
            for params in ParameterGrid(self.param_grid):
                params_df = pd.DataFrame(
                    {name: [value] for name, value in params.items()}
                ).merge(full_params_grid_df)
                if params_df.size == 0:
                    raise ValueError(
                        'Parameter grid includes the parameters value(s) '
                        f'{params} that are not allowed by available data.'
                    )
                param_grid.append(pd.DataFrame(params_df).merge(full_params_grid_df))
            param_grid = pd.concat(param_grid, ignore_index=True)
            self.param_grid_ = ParameterGrid(
                [
                    {k: [v] for k, v in params.to_dict().items()}
                    for _, params in param_grid.iterrows()
                ]
            )
        else:
            self.param_grid_ = self.PARAMS
        return self

    def _convert_data_types(self, data):
        """Cast the data type of columns."""
        data_types = set([data_type for _, data_type in self.SCHEMA])
        for data_type in data_types:
            converted_cols = list(
                {
                    col
                    for col, selected_data_type in self.SCHEMA
                    if selected_data_type is data_type and col in data.columns
                }
            )
            if converted_cols:
                data_converted_cols = data[converted_cols].fillna(
                    -1 if data_type is int else np.nan
                )
                data.loc[:, converted_cols] = (
                    data_converted_cols.values.astype(data_type)
                    if data_type is not np.datetime64
                    else pd.to_datetime(data_converted_cols.iloc[:, 0])
                )
        return data

    def _validate_data(self):
        """Validate the data."""
        data = self._get_data()
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data should be a pandas dataframe. Got '
                f'{type(data).__name__} instead.'
            )
        if data.size == 0:
            raise ValueError('Data should be a pandas dataframe with positive size.')
        if 'fixtures' not in data.columns or data['fixtures'].dtype.name != 'bool':
            raise KeyError(
                'Data should include a boolean column `fixtures` to distinguish '
                'between train and fixtures data.'
            )
        if 'date' not in data.columns or data['date'].dtype.name != 'datetime64[ns]':
            raise KeyError(
                'Data should include a datetime column `date` to represent the date.'
            )
        if self.SCHEMA and not set([col for col, _ in self.SCHEMA]).issuperset(
            data.columns.difference(['fixtures'])
        ):
            raise ValueError('Data contains columns not included in schema.')

        # Set date as index
        data = data.set_index('date').sort_values('date')

        # Remove missing values of data
        data = data[~data.index.isna()]

        # Check consistency with available parameters
        mask = data['fixtures']
        train_data = data[~mask].drop(columns=['fixtures'])
        params = pd.DataFrame(self.PARAMS)
        train_data_params = (
            train_data[params.columns].drop_duplicates().reset_index(drop=True)
        )
        try:
            pd.testing.assert_frame_equal(
                train_data_params.merge(params), train_data_params, check_dtype=False
            )
        except AssertionError:
            raise ValueError('The raw data and available parameters are incompatible.')

        return data

    def _extract_train_data(self, data):

        data = data[~data['fixtures']].drop(columns=['fixtures'])

        # Keep selected parameters from train data
        data = (
            data.reset_index().merge(pd.DataFrame(self.param_grid_)).set_index('date')
        )

        # Drop rows and columns with missing values
        data.dropna(
            subset=self._extract_cols(data, _is_output_col), how='any', inplace=True
        )
        data.dropna(axis=1, how='all', inplace=True)

        return data

    def _check_dropped_na_cols(self, data, drop_na_thres):
        remaining_cols = data.dropna(
            axis=1, thresh=int(data.shape[0] * drop_na_thres)
        ).columns
        self.dropped_na_cols_ = pd.Index(
            [
                col
                for col in self._extract_cols(data, _is_input_col)
                if col not in remaining_cols
            ]
        )
        if self._extract_cols(data, _is_input_col) == self.dropped_na_cols_.tolist():
            raise ValueError(
                'All input columns were removed. Set `drop_na_thres` parameter to a '
                'lower value.'
            )
        return self

    def _extract_cols(self, data, func):
        return [col for col, _ in self.SCHEMA if col in data.columns if func(col)]

    def extract_train_data(self, drop_na_thres=0.0, odds_type=None):
        """Extract the training data.

        Read more in the :ref:`user guide <user_guide>`.

        It returns historical data that can be used to create a betting
        strategy based on heuristics or machine learning models.

        The data contain information about the matches that belong
        in two categories. The first category includes any information
        known before the start of the match, i.e. the training data ``X``
        and the odds data ``O``. The second category includes the outcomes of
        matches i.e. the multi-output targets ``Y``.

        The method selects only the the data allowed by the ``param_grid``
        parameter of the initialization method
        :func:`~sportsbet.datasets._base._BaseDataLoader.__init__`.
        Additionally, columns with missing values are dropped through the
        ``drop_na_thres`` parameter, while the types of odds returned is defined
        by the ``odds_type`` parameter.

        Parameters
        ----------
        drop_na_thres : float, default=0.0
            The threshold that specifies the input columns to drop. It is a float in
            the :math:`[0.0, 1.0]` range. Higher values result in dropping more values.
            The default value ``drop_na_thres=0.0`` keeps all columns while the
            maximum value ``drop_na_thres=1.0`` keeps only columns with non
            missing values.

        odds_type : str, default=None
            The selected odds type. It should be one of the available odds columns
            prefixes returned by the method
            :func:`~sportsbet.datasets._base._BaseDataLoader.get_odds_types`. If
            ``odds_type=None`` then no odds are returned.

        Returns
        -------
        (X, Y, O) : tuple of :class:`~pandas.DataFrame` objects
            Each of the components represent the training input data ``X``, the
            multi-output targets ``Y`` and the corresponding odds ``O``, respectively.
        """

        # Validate the data
        data = self._validate_data()

        # Check param grid
        self._check_param_grid()

        # Extract train data
        data = self._extract_train_data(data)

        # Check dropped columns
        self.drop_na_thres_ = check_scalar(
            drop_na_thres, 'drop_na_thres', float, min_val=0.0, max_val=1.0
        )
        self._check_dropped_na_cols(data, drop_na_thres)

        # Check odds type
        odds_types = sorted(
            {col.split('__')[0] for col in data.columns if _is_odds_col(col)}
        )
        if odds_type is not None:
            if odds_type not in odds_types:
                raise ValueError(
                    "Parameter `odds_type` should be a prefix of available odds "
                    f"columns. Got `{odds_type}` instead."
                )
        self.odds_type_ = odds_type

        # Extract input, output and odds columns
        self.input_cols_ = pd.Index(
            [
                col
                for col in self._extract_cols(data, _is_input_col)
                if col not in self.dropped_na_cols_
            ],
            dtype=object,
        )
        self.odds_cols_ = pd.Index(
            [
                col
                for col in self._extract_cols(data, _is_odds_col)
                if col.split('__')[0] == self.odds_type_
            ],
            dtype=object,
        )

        # Convert data types
        data = self._convert_data_types(data)

        # Extract targets
        Y = []
        if self.odds_cols_.size == 0:
            for col, func in self.OUTCOMES:
                Y.append(pd.Series(func(data).reset_index(drop=True), name=col))
        else:
            for odds_col in self.odds_cols_:
                outcomes = [
                    (col, func)
                    for col, func in self.OUTCOMES
                    if col.split('__')[0] == odds_col.split('__')[1]
                ]
                if outcomes:
                    col, func = outcomes[0]
                    Y.append(pd.Series(func(data).reset_index(drop=True), name=col))
                else:
                    self.odds_cols_ = self.odds_cols_.drop(odds_col)
        Y = pd.concat(Y, axis=1) if Y else None
        self.output_cols_ = Y.columns

        return (
            data[self.input_cols_],
            Y,
            data[self.odds_cols_].reset_index(drop=True)
            if self.odds_type_ is not None
            else None,
        )

    def extract_fixtures_data(self):
        """Extract the fixtures data.

        Read more in the :ref:`user guide <user_guide>`.

        It returns fixtures data that can be used to make predictions for
        upcoming matches based on a betting strategy.

        Before calling the
        :func:`~sportsbet.datasets._BaseDataLoader.extract_fixtures_data` method for
        the first time, the :func:`~sportsbet.datasets._BaseDataLoader.extract__data`
        should be called, in order to match the columns of the input, output and
        odds data.

        The data contain information about the matches known before the
        start of the match, i.e. the training data ``X`` and the odds
        data ``O``. The multi-output targets ``Y`` is always equal to ``None``
        and are only included for consistency with the method
        :func:`~sportsbet.datasets._base._BaseDataLoader.extract_train_data`.

        The ``param_grid`` parameter of the initialization method
        :func:`~sportsbet.datasets._base._BaseDataLoader.__init__` has no effect
        on the fixtures data.

        Returns
        -------
        (X, None, O) : tuple of :class:`~pandas.DataFrame` objects
            Each of the components represent the fixtures input data ``X``, the
            multi-output targets ``Y`` equal to ``None`` and the
            corresponding odds ``O``, respectively.
        """

        # Extract fixtures data
        if not (
            hasattr(self, 'input_cols_')
            and hasattr(self, 'output_cols_')
            and hasattr(self, 'odds_cols_')
        ):
            raise AttributeError(
                'Extract the training data before extracting the fixtures data.'
            )

        data = self._validate_data()

        # Extract fixtures data
        data = data[data['fixtures']].drop(columns=['fixtures'])

        # Convert data types
        data = self._convert_data_types(data)

        # Remove past data
        data = data[data.index >= pd.to_datetime('today').floor('D')]

        return (
            data[self.input_cols_],
            None,
            data[self.odds_cols_].reset_index(drop=True)
            if self.odds_type_ is not None
            else None,
        )

    def save(self, path):
        """Save the dataloader object.

        Parameters
        ----------
        path : str
            The path to save the object.

        Returns
        -------
        self: object
            The dataloader object.
        """
        with open(path, 'wb') as file:
            cloudpickle.dump(self, file)
        return self

    @classmethod
    def get_all_params(cls):
        """Get the available parameters.

        It can be used to get the allowed names and values for the
        ``param_grid`` parameter of the dataloader object.

        Returns
        -------
        param_grid: list
            A list of all allowed params and values.
        """
        all_params = pd.DataFrame(cls.PARAMS)
        all_params = [
            {k: [v] for k, v in params.to_dict().items()}
            for _, params in all_params.sort_values(list(all_params.columns)).iterrows()
        ]
        return all_params

    def get_odds_types(self):
        """Get the available odds types.

        It can be used to get the allowed odds types of the dataloader's method
        :func:`~sportsbet.datasets._base._BaseDataLoader.extract_train_data`.

        Returns
        -------
        odds_types: list of str
            A list of available odds types.
        """
        # Validate the data
        data = self._validate_data()

        # Check param grid
        self._check_param_grid()

        # Extract train data
        data = self._extract_train_data(data)

        return sorted({col.split('__')[0] for col in data.columns if _is_odds_col(col)})


def load(path):
    """Load the dataloader object.

    Parameters
    ----------
    path : str
        The path of the dataloader pickled file.

    Returns
    -------
    dataloader: object
        The dataloader object.
    """
    with open(path, 'rb') as file:
        dataloader = cloudpickle.load(file)
    return dataloader
