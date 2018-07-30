#!/usr/bin/env python

"""
Download and prepare soccer historical data from various leagues.
"""

from os.path import join
from itertools import product
import pandas as pd


MAIN_URL = 'http://www.football-data.co.uk/mmz4281'
YEARS_URLS = ['1314', '1415', '1516', '1617', '1718']
LEAGUES_URLS = ['E0', 'E1', 'D1', 'D2', 'I1', 'I2', 'SP1',
                'SP2', 'F1', 'F2', 'N1', 'P1']
SUFFIX_URLS = [join(league, year) for league, year in product(YEARS_URLS, LEAGUES_URLS)]
URLS = [join(MAIN_URL, suffix) for suffix in SUFFIX_URLS]
ID_FEATURES = ['Div', 'Date', 'HomeTeam', 'AwayTeam']
RESULTS_FEATURES = ['FTHG', 'FTAG', 'FTR']
MATCH_FEATURES = ['HS', 'AS', 'HST', 'AST', 'HHW', 'AHW',
                  'HC', 'AC', 'HF', 'AF', 'HFKC', 'AFKC',
                  'HO', 'AO', 'HY', 'AY', 'HR', 'AR']
ODDS_FEATURES = [elem1 + elem2 for elem1, elem2 in product(
    ['B365', 'BW', 'IW', 'LB', 'PS', 'VC', 'BbMx', 'BbAv'],
    ['H', 'D', 'A'])]
GOALS_ODDS_FEATURES = ['BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5']
ASIAN_ODDS_FEATURES = ['BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'BbAHh']
CLOSING_ODDS = ['PSCH', 'PSCD', 'PSCA']
TOTAL_DATA_FEATURES = ID_FEATURES + RESULTS_FEATURES + ODDS_FEATURES + GOALS_ODDS_FEATURES + ASIAN_ODDS_FEATURES + CLOSING_ODDS


data_list = [pd.read_csv(url) for url in URLS]

total_data = pd.DataFrame()
for data in data_list:
    total_data = total_data.append(data.loc[:, TOTAL_DATA_FEATURES])
total_data.reset_index(drop=True, inplace=True)
total_data = total_data[~total_data.Div.isnull()]
