#!/usr/bin/env python

"""
Download and prepare soccer historical data from various leagues.
"""

from os.path import join, dirname
from re import sub
import pandas as pd
from sportsbet.soccer import MAIN_URL, URLS, DATA_FEATURES, RESULTS_MAPPING, TRAINING_FEATURES


def download_datasets():
    """Download all the datasets."""
    data_list = []
    for url in URLS:
        season = sub('^' + MAIN_URL, '', url).split('/')[1]
        data = pd.read_csv(url)
        data['Season'] = season[:2] + '-' + season[2:]
        data_list.append(data)
    return data_list


def combine_datasets(data_list):
    """Combine various datasets."""
    data = pd.DataFrame()
    for league_data in data_list:
        data = data.append(league_data.loc[:, DATA_FEATURES])
    data.reset_index(drop=True, inplace=True)
    data = data[~data.Div.isnull()]
    return data


def extract_features(data):
    """Extract features for modelling."""
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')
    data['FTHG'] = data['FTHG'].astype(int)
    data['FTAG'] = data['FTAG'].astype(int)
    data['FTR'] = data['FTR'].apply(lambda result: RESULTS_MAPPING[result])
    date_range = data.groupby(['Div', 'Season']).agg({'Date': [min, max]}).reset_index()
    date_range.columns = ['Div', 'Season', 'MinDate', 'MaxDate']
    data = pd.merge(data, date_range, on=['Div', 'Season'])
    data['Progress'] = (100 * (data['Date'] - data['MinDate']).apply(lambda time: time.value) /
                        (data['MaxDate'] - data['MinDate']).apply(lambda time: time.value)).astype(int)
    data = data.sort_values(['Date', 'Div']).reset_index(drop=True)
    data['TimeIndex'] = (data.Date - min(data.Date)).dt.days
    data['TotalGoals'] = data['FTHG'] + data['FTAG']
    data.drop(columns=['Date', 'MaxDate', 'MinDate', 'FTHG', 'FTAG'], inplace=True)
    return data


if __name__ =='__main__':
    data_list = download_datasets()
    data = combine_datasets(data_list)
    data = extract_features(data)
    training_odds_data = data.loc[:,  TRAINING_FEATURES + ['FTR']]
    training_over_under_data = data.loc[:, TRAINING_FEATURES + ['TotalGoals']]
    training_odds_data.to_csv(join(dirname(__file__), '..', '..', 'data', 'training_odds_data.csv'), index=False)
    training_over_under_data.to_csv(join(dirname(__file__), '..', '..', 'data', 'training_over_under_data.csv'), index=False)
