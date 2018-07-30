#!/usr/bin/env python

"""
Download and prepare soccer historical data from various leagues.
"""

from os.path import join
from re import sub
from itertools import product
import pandas as pd


MAIN_URL = 'http://www.football-data.co.uk/mmz4281'
YEARS_URLS = ['1314', '1415', '1516', '1617', '1718']
LEAGUES_URLS = ['E0', 'E1', 'D1', 'D2', 'I1', 'I2', 'SP1',
                'SP2', 'F1', 'F2', 'N1', 'P1']
SUFFIX_URLS = [join(league, year) for league, year in product(YEARS_URLS, LEAGUES_URLS)]
URLS = [join(MAIN_URL, suffix) for suffix in SUFFIX_URLS]
ID_FEATURES = ['Div', 'Date', 'Season', 'HomeTeam', 'AwayTeam']
RESULTS_FEATURES = ['FTHG', 'FTAG', 'FTR']
ODDS_FEATURES = [agent + result for agent, result in product(
    ['B365', 'BW', 'IW', 'LB', 'PS', 'VC', 'BbMx', 'BbAv'],
    ['H', 'D', 'A'])]
GOALS_ODDS_FEATURES = ['BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5']
ASIAN_ODDS_FEATURES = ['BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'BbAHh']
CLOSING_ODDS = ['PSCH', 'PSCD', 'PSCA']
TOTAL_DATA_FEATURES = ID_FEATURES + RESULTS_FEATURES + ODDS_FEATURES + GOALS_ODDS_FEATURES + ASIAN_ODDS_FEATURES + CLOSING_ODDS
TRAINING_FEATURES = ['TimeIndex', 'Progress', 'Div', 'Season', 'HomeTeam', 'AwayTeam'] + ODDS_FEATURES + GOALS_ODDS_FEATURES + ASIAN_ODDS_FEATURES + CLOSING_ODDS


def download_datasets():
    """Download all the datasets."""
    data_list = []
    for url in URLS:
        season = sub('^' + MAIN_URL, '', url).split('/')[1]
        data = pd.read_csv(url)
        data['Season'] = season
        data_list.append(data)
    return data_list


def combine_datasets(data_list):
    """Combine various datasets."""
    data = pd.DataFrame()
    for league_data in data_list:
        data = data.append(league_data.loc[:, TOTAL_DATA_FEATURES])
    data.reset_index(drop=True, inplace=True)
    data = data[~data.Div.isnull()]
    return data


def extract_features(data):
    """Extract features for modelling."""
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')
    data['FTHG'] = data['FTHG'].astype(int)
    data['FTAG'] = data['FTAG'].astype(int)
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
    training_odds_data.to_csv(join('data', 'training_odds_data.csv'), index=False)
    training_over_under_data.to_csv(join('data', 'training_over_under_data.csv'), index=False)
