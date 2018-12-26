"""
Includes classes and functions to test and select the optimal 
betting strategy on historical and current data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from os import listdir
from os.path import join
from pickle import dump, load
from pathlib import Path
from itertools import product, combinations
from collections import OrderedDict
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.utils import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from .. import PATH
from ..utils import (
    fit_predict,
    generate_month_indices,
    CLASSIFIERS
)
from .data import load_data

RESULTS = ['H', 'A', 'D']
RESULTS_PATH = join(PATH, 'results')

class BettingAgent:
    """Class that is used for model evaluation, training and predictions 
    on upcoming matches."""

    @staticmethod
    def calculate_backtest_results(backtest_data):
        """Calculate the results of backtesting."""

        backtest_results = pd.DataFrame()
        for predicted_result in ['H', 'A', 'D', 'HA', 'HD', 'AD', 'HAD']:

            # Extract predicted results
            predicted_results = list(predicted_result)
        
            # Extract predictions and odds
            backtest_data['Bet'] = backtest_data[predicted_results].idxmax(axis=1)
            backtest_data.loc[~(backtest_data[predicted_results] > 0.5).sum(axis=1).astype(bool), 'Bet'] = '-'
            backtest_data['Odd'] = backtest_data[['H Odd', 'A Odd', 'D Odd']].apply(lambda odds: dict(zip(RESULTS, odds.values)), axis=1)

            # Extract intermediate results
            mask = backtest_data['Bet'] != '-'
            yields = backtest_data.loc[mask, ['Result', 'Bet', 'Odd']].apply(lambda row: row.Odd[row.Bet] - 1.0 if row.Result == row.Bet else -1.0, axis=1)

            # Extract results
            results = (
                predicted_result,
                len(backtest_data),
                mask.sum(),
                precision_score(backtest_data['Result'], backtest_data['Bet'], labels=predicted_results, average='micro'),
                yields.sum(),
                yields.mean()
            )

            # Combine results
            backtest_results = backtest_results.append(pd.DataFrame([results], columns=['Predicted results', 'Number of matches', 'Number of bets', 'Precision', 'Profit', 'Yield']))

        return backtest_results
        
    def backtest(self, clfs_name, max_odds, test_season, random_states, save_results):
        """Apply backtesting to betting agent."""

        # Load and prepare data
        X, y, odds, matches = load_data('historical', max_odds)

        # Extract classifiers and parameters grids
        classifiers = {label: clf for label, (clf, _) in CLASSIFIERS[clfs_name].items()}
        param_grids = {label: param_grid for label, (_ , param_grid) in CLASSIFIERS[clfs_name].items()}

        # Define train and test indices of each month
        month_indices = generate_month_indices(matches, test_season)

        # Flatten and enumerate parameter grid
        self.param_grids_ = [(label, ind, params) for label, param_grid in param_grids.items() for ind, params in enumerate(ParameterGrid(param_grid))]

        # Run cross-validation
        cv_data = Parallel(n_jobs=-1)(delayed(fit_predict)(classifiers[params[0]], params, X, y, odds,
                                                           train_indices, test_indices, random_state) 
        for (train_indices, test_indices), params, random_state in tqdm(list(product(month_indices, self.param_grids_, random_states)), desc='Backtesting: '))
        
        # Combine data
        backtest_data = OrderedDict()
        for start, random_state, key, data in cv_data:
            backtest_data.setdefault((start, random_state), []).append((key, data))
        
        columns = ['Result', 'H', 'A', 'D', 'H Odd', 'A Odd', 'D Odd']
        self.backtest_data_ = []
        for start, random_state in backtest_data.keys():
            for (key1, data1), (key2, data2), (key3, data3) in combinations(backtest_data[(start, random_state)], 3):
                if len(set([key1[0], key2[0], key3[0]])) == 3:
                    data = reduce(lambda df1, df2: pd.merge(df1, df2), (data1, data2, data3))[columns]
                    indexH, indexA, indexD = [dict((key1, key2, key3))[label] for label in RESULTS]
                    data = data.assign(IndexH=indexH, IndexA=indexA, IndexD=indexD, Start=start, Experiment=random_state)
                    self.backtest_data_.append(data)
        self.backtest_data_ = pd.concat(self.backtest_data_).sort_values(['IndexH', 'IndexA', 'IndexD', 'Experiment', 'Start']).drop(columns='Start').reset_index()
                    
        # Extract results
        self.backtest_results_ = self.backtest_data_.groupby(['IndexH', 'IndexA', 'IndexD', 'Experiment']).apply(self.calculate_backtest_results)
        
        # Group results by parameters and predicted result
        grouped_results = self.backtest_results_.groupby(['IndexH', 'IndexA', 'IndexD', 'Predicted results'])
        
        # Calculate mean values
        self.backtest_results_ = grouped_results.mean()
        
        # Calculate COV
        self.backtest_results_['COV'] = grouped_results['Yield'].std() / np.abs(self.backtest_results_['Yield']) 
        
        # Cast columns
        self.backtest_results_['Number of matches'] = self.backtest_results_['Number of matches'].astype(int)
        self.backtest_results_['Number of bets'] = self.backtest_results_['Number of bets'].astype(int)
        
        # Sort results
        self.backtest_results_ = self.backtest_results_.sort_values('Yield', ascending=False).reset_index()

        # Combine identical results
        keys = ['Predicted results', 'Number of matches', 'Number of bets', 'Precision', 'Profit', 'Yield', 'COV']
        self.backtest_results_ = self.backtest_results_.groupby(keys, sort=False).apply(lambda df: list(zip(df.IndexH.values, df.IndexA.values, df.IndexD.values))).reset_index()
        self.backtest_results_.rename(columns={0: 'Indices'}, inplace=True)

        # Save results
        if save_results:
            Path(RESULTS_PATH).mkdir(exist_ok=True)
            self.backtest_results_.to_csv(join(RESULTS_PATH, 'backtesting.csv'), index=False)

    def fit_dump_classifiers(self, backtest_index):
        """Fit and dump classifiers."""

        # Create directory
        path = join(RESULTS_PATH, backtest_index)
        Path(path).mkdir(exist_ok=True)

        # Load training data
        X, y, _, _ = load_data('historical', ['pinnacle', 'bet365', 'bwin'])

        # Extract classifiers and parameters grids
        classifiers = {label: clf for label, (clf, _) in CLASSIFIERS['default'].items()}
        param_grids = {label: ParameterGrid(param_grid) for label, (_ , param_grid) in CLASSIFIERS['default'].items()}

        # Load backtesting results
        predicted_result, params_indices = pd.read_csv(join(RESULTS_PATH, 'backtesting.csv'), usecols=['Predicted results', 'Indices']).values[int(backtest_index)]
        
        # Extract predicted results and parameters indices
        predicted_results = list(predicted_result)
        params_indices = dict(zip(RESULTS, eval(params_indices)[0]))

        # Fit and dump classifiers
        for label in predicted_results:

            # Select classifier and parameters
            classifier = classifiers[label]
            params = param_grids[label][params_indices[label]]

            # Binarize labels
            y_bin = y.copy()
            y_bin[y_bin != label] = '-'

            # Fit classifier
            classifier.set_params(**params).fit(X, y_bin)
            
            # Dump classifier
            with open(join(path, '%s.pkl' % label) , 'wb') as file:
                dump(classifier, file)
        
    def predict(self, backtest_index):
        """Generate predictions using fitted classifiers."""

        # Load predictions data
        X, _, odds, matches  = load_data('predictions', ['pinnacle', 'bet365', 'bwin'])

        # Define file path and pickled classifiers
        path = join(RESULTS_PATH, backtest_index)
        pickled_classifiers = [file for file in listdir(path) if 'pkl' in file]

        # Iterate through results
        y_pred = pd.DataFrame()
        for pickled_classifier in pickled_classifiers:

            # Load classifier
            with open(join(path, pickled_classifier) , 'rb') as file:
                classifier = load(file)

            # Populate predictions
            y_pred = pd.concat([y_pred, pd.DataFrame(classifier.predict_proba(X)[:, -1], columns=[pickled_classifier[0]])], axis=1)
        
        # Extract predictions
        y_pred['Bet'] = y_pred.idxmax(axis=1)
        y_pred.loc[~(y_pred[ y_pred.columns[:-1]] > 0.5).sum(axis=1).astype(bool), 'Bet'] = '-'

        # Extract maximum odds
        y_pred['Maximum Odds'] = odds.apply(lambda odds: dict(zip(RESULTS, odds.values)), axis=1)
        y_pred['Maximum Odds'] = y_pred.apply(lambda row: row['Maximum Odds'][row['Bet']] if row['Bet'] != '-' else np.nan, axis=1)
        
        # Combine data
        predictions = pd.concat([matches.drop(columns=['Month', 'Day']), y_pred[['Bet', 'Maximum Odds']]], axis=1)
        predictions = predictions[predictions['Bet'] != '-'].reset_index(drop=True)

        return predictions
