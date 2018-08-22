"""
Includes wrapper functions of main function and classes. 
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from os.path import join
from warnings import filterwarnings
import numpy as np
import pandas as pd
from .configuration import ( 
    ESTIMATORS, 
    PARAM_GRIDS,
    FIT_PARAMS,
    MIN_N_MATCHES,
    ODDS_THRESHOLDS, 
    GENERATE_WEIGHTS,
    SCORERS_MAPPING
)
from .data import fetch_raw_data, extract_training_data, extract_odds_dataset
from .optimization import Betting


def fetch_simulation_data(dirpath=None):
    """Download and save training and odds data."""
    spi_data, fd_data = fetch_raw_data()
    training_data = extract_training_data(spi_data, fd_data)
    odds_data = extract_odds_dataset(fd_data)
    if dirpath is not None:
        training_data.to_csv(join(dirpath, 'training_data.csv'), index=False)
        odds_data.to_csv(join(dirpath, 'odds_data.csv'), index=False)
    return training_data, odds_data


def generate_grid_scores(predicted_result, dirpath='data', test_season='17-18', random_state=None):
    """Generate the grid scores of a test season."""
    filterwarnings('ignore')
    training_data = pd.read_csv(join(dirpath, 'training_data.csv'))
    betting = Betting(ESTIMATORS[predicted_result], PARAM_GRIDS[predicted_result], FIT_PARAMS[predicted_result])
    grid_scores = betting.return_grid_scores(SCORERS_MAPPING[predicted_result], training_data, test_season, MIN_N_MATCHES, random_state)
    return grid_scores


def generate_simulation_results(predicted_result, dirpath='data', test_season='17-18', generate_weights=GENERATE_WEIGHTS, random_state=None):
    """Run betting simulation and generate the results."""
    filterwarnings('ignore')
    training_data = pd.read_csv(join(dirpath, 'training_data.csv'))
    odds_data = pd.read_csv(join(dirpath, 'odds_data.csv'))
    betting = Betting(ESTIMATORS[predicted_result], PARAM_GRIDS[predicted_result], FIT_PARAMS[predicted_result])
    results = betting.simulate_results(training_data, odds_data, test_season, predicted_result, MIN_N_MATCHES, 
                                       ODDS_THRESHOLDS[predicted_result], generate_weights, random_state)
    return results
