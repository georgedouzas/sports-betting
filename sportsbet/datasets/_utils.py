"""
Includes utilities to download and transform data.
"""

from abc import abstractmethod, ABCMeta
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.utils import Bunch, check_scalar


def _combine_odds(odds):
    """Combine odds of different targets."""
    combined_odds = 1 / (1 / odds).sum(axis=1)
    return combined_odds


def _create_names_mapping_table(left_data, right_data):
    """Create names mapping table."""

    # Rename columns
    key_columns = ['key0', 'key1']
    left_data.columns = key_columns + ['left_team1', 'left_team2']
    right_data.columns = key_columns + ['right_team1', 'right_team2']

    # Generate teams names combinations
    names_combinations = pd.merge(left_data, right_data, how='outer').dropna().drop(columns=key_columns).reset_index(drop=True)

    # Calculate similarity index
    similarity = names_combinations.apply(lambda row: SequenceMatcher(None, row.left_team1, row.right_team1).ratio() * SequenceMatcher(None, row.left_team2, row.right_team2).ratio(), axis=1)

    # Append similarity index
    names_combinations_similarity = pd.concat([names_combinations, similarity], axis=1)

    # Filter correct matches
    indices = names_combinations_similarity.groupby(['left_team1', 'left_team2'])[0].idxmax().values
    names_matching = names_combinations.take(indices)

    # Teams matching
    matching1 = names_matching.loc[:, ['left_team1', 'right_team1']].rename(columns={'left_team1': 'left_team', 'right_team1': 'right_team'})
    matching2 = names_matching.loc[:, ['left_team2', 'right_team2']].rename(columns={'left_team2': 'left_team', 'right_team2': 'right_team'})
        
    # Combine matching
    matching = matching1.append(matching2)
    matching = matching.groupby(matching.columns.tolist()).size().reset_index()
    indices = matching.groupby('left_team')[0].idxmax().values
        
    # Generate mapping
    names_mapping = matching.take(indices).drop(columns=0).reset_index(drop=True)

    return names_mapping


class _DataLoader(metaclass=ABCMeta):
    """Base class to load data for model training and testing.

        parameters
        ----------
        config : list of tuples
            The configuration of the data preprocessing stage. It contains tuples
            with each tuple having four elements. The first one is the initial
            column name (None for a created column), the second one is the final column
            name (None for a removed column), the third one is the column data type
            and the last one is the id of an output columns (None if it is not an
            output column).
        
        targets : list of tuples
            It defines the calculation of the target columns. It contains tuples
            with each tuple having three elements. The first one is the name of the
            created target column, the second is a function that accepts as input
            the output columns and generates a binary target and the third one is
            the id of the output columns.

        param_grid : dict of str to sequence, or sequence of such parameter, default=None
            The parameter grid to explore, as a dictionary mapping data parameters
            to sequences of allowed values. An empty dict signifies default
            parameters. A sequence of dicts signifies a sequence of grids to search,
            and is useful to avoid exploring parameter combinations that do not
            exist. The default value corresponds to all parameters.
        
        drop_na_cols : float, default=None
            The threshold of input columns with missing values to drop. It is a
            float in the [0.0, 1.0] range. The default value ``None``
            corresponds to ``0.0`` i.e. all columns are kept while the value
            ``1.0`` keeps only columns with non missing values.
        
        drop_na_rows : float, default=None
            The threshold of rows with missing values to drop. It is a
            float in the [0.0, 1.0] range. The default value ``None``
            corresponds to ``0.0`` i.e. all rows are kept while the value
            ``1.0`` keeps only rows with non missing values.
            
        odds_type : str, default=None
            The prefix of the odds column to be used for generating the odds
            data. The default value does not return any odds data.
        
        testing_duration : int, default=None
            The number of future weeks to include in the testing data. The
            default value corresponds to one week.
        """

    def __init__(self, config, targets, param_grid=None, drop_na_cols=None, drop_na_rows=None, odds_type=None, testing_duration=None):
        self.config = config
        self.targets =  targets
        self.param_grid = param_grid
        self.drop_na_cols = drop_na_cols
        self.drop_na_rows = drop_na_rows
        self.odds_type = odds_type
        self.testing_duration = testing_duration

    def _fetch(self, return_only_params, **load_params):
        """Fetch the full parameter grid and the data."""
        self._fetch_full_param_grid(**load_params)
        if not return_only_params:
            self._check_parameters()._check_param_grid()._fetch_data(**load_params)._check_data()._check_cols()._transform_data()
        return self
    
    def _fetch_full_param_grid(self, **load_params):
        """Fetch the full parameter grid."""
        self.full_param_grid_ = None
        return self
    
    def _check_parameters(self):
        """Check input parameters."""

        # Check parameter to drop columns
        self.drop_na_cols_ = self.drop_na_cols if self.drop_na_cols is not None else 0.0
        check_scalar(self.drop_na_cols_, 'drop_na_cols', float, min_val=0.0, max_val=1.0)
        
        # Check parameter to drop rows
        self.drop_na_rows_ = self.drop_na_rows if self.drop_na_rows is not None else 0.0
        check_scalar(self.drop_na_rows_, 'drop_na_cols', float, min_val=0.0, max_val=1.0)

        # Check testing duration
        self.testing_duration_ = self.testing_duration if self.testing_duration is not None else 1
        check_scalar(self.testing_duration_, 'testing_duration', int, min_val=1)

        # Check odds type
        if self.odds_type is not None:
            if not isinstance(self.odds_type, str):
                raise TypeError(f'Parameter `odds_type` should be a string or None. Got {type(self.odds_type)} instead.')
            cols_odds = [col for _, col, _ in self.config if col is not None and col.split('__')[0] == self.odds_type and col.split('__')[-1] == 'odds']
            if not cols_odds:
                raise ValueError(f'Parameter `odds_type` should be a prefix of available odds columns. Got {self.odds_type} instead.')
            self.odds_type_ = self.odds_type
        else:
            self.odds_type_ = ''
        
        return self

    def _check_param_grid(self):
        """Check parameter grid."""
        if self.param_grid is not None and self.full_param_grid_ is not None:
            param_grid = ParameterGrid(self.param_grid)
            param_grid_df = pd.DataFrame(param_grid)
            full_param_grid_df = pd.DataFrame(self.full_param_grid_)
            if np.any(pd.merge(param_grid_df, full_param_grid_df, how='left').isna()):
                raise ValueError(f'Parameter grid includes values not allowed by available data.')
            else:
                param_grid_df = pd.merge(param_grid_df, full_param_grid_df)
                self.param_grid_ =  ParameterGrid([{k: [v] for k, v in row.to_dict().items()} for ind, row in param_grid_df.iterrows()])
        else:
            self.param_grid_ =  self.full_param_grid_
        return self
    
    @abstractmethod
    def _fetch_data(self, **load_params):
        """Fetch the data."""
        self.data_ = pd.DataFrame()
        return self
    
    def _check_data(self):
        """Check data format, rename and drop columns."""
        if not isinstance(self.data_, pd.DataFrame):
            raise TypeError(f'Attribute `data_` should be a pandas dataframe. Got {type(self.data_).__name__} instead.')
        if self.data_.size == 0:
            raise ValueError(f'Attribute `data_` should be a pandas dataframe with positive size.')
        if 'test' not in self.data_.columns:
            raise KeyError('Attribute `data_` should include a boolean column `test` to distinguish between train and test data.')
        cols_removed = pd.Index([old_col for old_col, new_col, _ in self.config if new_col is None and old_col in self.data_], dtype=object)
        cols_rename_mapping = {old_col: new_col for old_col, new_col, _ in self.config if old_col is not None and new_col is not None}
        self.data_ = self.data_.drop(columns=cols_removed).rename(columns=cols_rename_mapping)
        set(self.data_.columns).difference([col for _, col, _ in self.config if col is not None])
        if not set(self.data_.columns).issubset():
            raise ValueError(f'Columns {self.data_.columns} of attribute `data_` should be included in the schema.')
        return self
    
    def _check_cols(self):
        """Check columns."""
        
        # Inputs
        self.cols_inputs_ = pd.Index([col for col in self.data_.columns if len(col.split('__')) != 2 and col != 'test'], dtype=object)
        
        # Targets
        targets = pd.DataFrame([target.split('__') for target, _ in self.targets], columns=['target', 'key'])

        # Odds
        odds = []
        for col in self.data_:
            splitted_col = col.split('__')
            if splitted_col[0] == self.odds_type_ and splitted_col[-1] == 'odds':
                odds.append((col, '__'.join(splitted_col[1:-1])))
        odds = pd.merge(pd.DataFrame(odds, columns=['odd', 'target']), targets)
        
        # Outputs
        outputs = [(col, col.split('__')[-1]) for col in self.data_.columns if col.split('__')[-1] != 'odds']
        outputs = pd.DataFrame(outputs, columns=['output', 'key']).dropna()
        outputs = pd.merge(outputs, targets)
        
        if odds.size:
            odds = outputs = pd.merge(odds, outputs)
        self.cols_odds_ = pd.Index(odds['odd'].unique(), dtype=object)
        self.cols_outputs_ = pd.Index(outputs['output'].unique(), dtype=object)

        return self

    def _transform_data(self):
        """Rename, drop and change data type of columns."""
        
        # Drop rows with only missing values
        mask = self.data_['test']
        data_train, data_test = self.data_[~mask].drop(columns=['test']), self.data_[mask].drop(columns=['test'])
        data_train_init = data_train.copy()
        
        # Drop columns and rows with missing values
        data_train = data_train.dropna(how='all', axis=0)
        if self.drop_na_cols_ > 0.0:
            data_train_dropped_na_cols = data_train[self.cols_inputs_].dropna(axis=1, thresh=int(data_train.shape[0] * self.drop_na_cols_))
            data_train_dropped_na_cols.drop(columns=self.cols_odds_.intersection(data_train_dropped_na_cols.columns), inplace=True)
            data_train = pd.concat([
                data_train_dropped_na_cols,
                data_train[self.cols_outputs_.append(self.cols_odds_)]
            ], axis=1)
        if self.drop_na_rows_ > 0.0:
            data_train = data_train.dropna(how='any', axis=0, subset=self.cols_outputs_.append(self.cols_odds_))
            data_train = data_train.dropna(axis=0, subset=self.cols_inputs_.intersection(data_train.columns), thresh=int(data_train.shape[1] * self.drop_na_rows_))
        self.dropped_na_cols_ = data_train_init.columns.difference(data_train.columns, sort=False)
        self.cols_inputs_ = self.cols_inputs_.difference(self.dropped_na_cols_, sort=False)
        self.dropped_na_rows_ = data_train_init.index.difference(data_train.index)
        
        # Combine data
        cols_init = [col for col in data_train_init if col in data_train]
        data_train = data_train[cols_init]
        data_test = data_test[cols_init]
        data = pd.concat([data_train, data_test], ignore_index=True)

        # Convert data types
        data_types = set([data_type for _, _, data_type in self.config if data_type is not None])
        for data_type in data_types:
            converted_cols = list({new_col for _, new_col, dt in self.config if new_col is not None and dt is data_type and new_col in data.columns})
            data_converted_cols = data[converted_cols].fillna(-1 if data_type is int else np.nan)
            data.loc[:, converted_cols] = data_converted_cols.values.astype(data_type) if data_type is not np.datetime64 else pd.to_datetime(data_converted_cols.iloc[:, 0])
        
        # Filter data
        self.data_train_ = data[:data_train.shape[0]]
        self.data_test_ = data[data_train.shape[0]:].drop(columns=self.cols_outputs_).reset_index(drop=True)
        if self.param_grid_ is not None:
            self.data_train_ = pd.merge(self.data_train_, pd.DataFrame(self.param_grid_))

        return self
    
    def _extract_test(self):
        """Extract test data."""
        current_date = datetime.now()
        end_date = current_date + timedelta(weeks=self.testing_duration_)
        mask = (self.data_test_.date.dt.date >= current_date.date()) & (self.data_test_.date.dt.date <= end_date.date())
        X = self.data_test_.loc[mask, self.cols_inputs_].reset_index(drop=True)
        O = X[self.cols_odds_] if self.cols_odds_.size else None
        return Bunch(X=X, Y=None, O=O)
    
    def _extract_train(self):
        """Extract train data."""
        X = self.data_train_
        Y = []
        for target, func in self.targets:
            outputs = X[self.cols_outputs_]
            Y.append(pd.Series(func(outputs), name=target).reindex_like(outputs))
        Y = pd.concat(Y, axis=1)
        O = X[self.cols_odds_] if self.cols_odds_.size else None
        X = X[self.cols_inputs_]
        return Bunch(X=X, Y=Y, O=O)
    
    def load(self, return_only_params=False, **load_params):
        """Extract and load model data."""
        self._fetch(return_only_params, **load_params)
        bunch = Bunch(
            training=None,
            testing=None,
            dropped=Bunch(cols=None, rows=None),
            params = Bunch(selected=None, available=pd.DataFrame(self.full_param_grid_))
        )
        if not return_only_params:
            bunch.training = self._extract_train()
            bunch.testing = self._extract_test()
            bunch.dropped.cols=self.dropped_na_cols_
            bunch.dropped.rows=self.dropped_na_rows_
            bunch.params.selected = pd.DataFrame(self.param_grid_)
        return bunch
