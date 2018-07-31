from collections import OrderedDict
from os.path import join
from itertools import product

# Downloading URLs
MAIN_URL = 'http://www.football-data.co.uk/mmz4281'
YEARS_URLS = ['1314', '1415', '1516', '1617', '1718']
LEAGUES_URLS = ['E0', 'E1', 'D1', 'D2', 'I1', 'I2', 'SP1', 'SP2', 'F1', 'F2', 'N1', 'P1']
SUFFIX_URLS = [join(league, year) for league, year in product(YEARS_URLS, LEAGUES_URLS)]
URLS = [join(MAIN_URL, suffix) for suffix in SUFFIX_URLS]

# Betting agents
ODDS_BETTING_AGENTS = ['B365', 'BW', 'IW', 'LB', 'PS', 'VC', 'BbMx', 'BbAv']
GOALS_ODDS_BETTING_AGENTS = ['BbMx', 'BbAv']
ASIAN_ODDS_BETTING_AGENTS = ['BbMx', 'BbAv']

# Parameters
RESULTS_MAPPING = OrderedDict({'H': 0, 'D': 1, 'A': 2})
TEST_SEASON = '17-18'
BETTING_INTERVAL = 7

# Features
ID_FEATURES = ['Div', 'Date', 'Season', 'HomeTeam', 'AwayTeam']
TRAINING_ID_FEATURES = ['TimeIndex', 'Progress', 'Div', 'Season', 'HomeTeam', 'AwayTeam']
RESULTS_FEATURES = ['FTHG', 'FTAG', 'FTR']
ODDS_FEATURES = [agent + result for agent in ODDS_BETTING_AGENTS for result in RESULTS_MAPPING.keys()]
GOALS_ODDS_FEATURES = [agent + token for agent in GOALS_ODDS_BETTING_AGENTS for token in ['>2.5', '<2.5']]
ASIAN_ODDS_FEATURES = [agent + token for agent in ASIAN_ODDS_BETTING_AGENTS for token in ['AHH', 'AHA']] + ['BbAHh']
CLOSING_ODDS_FEATURES = ['PSCH', 'PSCD', 'PSCA']
TOTAL_DATA_FEATURES = ID_FEATURES + RESULTS_FEATURES + ODDS_FEATURES + GOALS_ODDS_FEATURES + ASIAN_ODDS_FEATURES + CLOSING_ODDS_FEATURES
TRAINING_FEATURES = TRAINING_ID_FEATURES + ODDS_FEATURES + GOALS_ODDS_FEATURES + ASIAN_ODDS_FEATURES + CLOSING_ODDS_FEATURES