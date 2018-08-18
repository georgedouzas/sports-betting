from collections import OrderedDict, Counter
from os.path import join
from itertools import product
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from category_encoders import OrdinalEncoder
from xgboost import XGBClassifier

# Parameters mappings
TEAMS_MAPPING = OrderedDict({
    'Borussia Monchengladbach': "M'gladbach",
    'Brighton and Hove Albion': 'Brighton',
    'Deportivo La Coruña': 'La Coruna',
    'Internazionale': 'Inter',
    'Chievo Verona': 'Chievo',
    'Wolverhampton': 'Wolves',
    'Cardiff City': 'Cardiff'
})
LEAGUES_MAPPING = OrderedDict({
    'Barclays Premier League': 'E0',
    'German Bundesliga': 'D1',
    'Italy Serie A': 'I1',
    'Spanish Primera Division': 'SP1'
})
SPI_FEATURES_MAPPING = OrderedDict({
    'date': 'Date',
    'league': 'League',
    'team1': 'HomeTeam',
    'team2': 'AwayTeam',
    'spi1': 'HomeTeamSPI',
    'spi2': 'AwayTeamSPI',
    'prob1': 'ProbHomeTeamSPI',
    'prob2': 'ProbAwayTeamSPI',
    'probtie': 'ProbDrawSPI'
})
FD_FEATURES_MAPPING = OrderedDict({
    'Div': 'League',
    'FTR': 'Target'
})
RESULTS_MAPPING = OrderedDict({
    'H': 0, 'D': 1, 'A': 2
})

# General parameters
MIN_N_MATCHES = 20
YEARS = ['1617', '1718', '1819']

# URLs
SPI_URL = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
FD_URL = 'http://www.football-data.co.uk/mmz4281'
FD_SUFFIX = [join(year, league) for year, league in product(YEARS, LEAGUES_MAPPING.values())]
FD_URLS = [join(FD_URL, suffix) for suffix in FD_SUFFIX]

# SPI data features
SPI_DATA_FEATURES = ['date', 'league', 'team1', 'team2', 'spi1', 'spi2', 'prob1', 'prob2', 'probtie']

# Football data features
FD_MAX_ODDS, FD_AVG_ODDS = 'BbMx', 'BbAv'
FD_MAX_ODDS_FEATURES = [FD_MAX_ODDS + result for result in RESULTS_MAPPING.keys()]
FD_AVG_ODDS_FEATURES = [FD_AVG_ODDS + result for result in RESULTS_MAPPING.keys()]
FD_KEYS_FEATURES = ['Div', 'Date', 'Season', 'HomeTeam', 'AwayTeam']
FD_DATA_FEATURES = FD_MAX_ODDS_FEATURES + FD_AVG_ODDS_FEATURES + FD_KEYS_FEATURES + ['FTR']

# Training data features
TRAIN_SPI_FEATURES = ['HomeTeamSPI', 'AwayTeamSPI']
TRAIN_PROB_SPI_FEATURES = ['ProbHomeTeamSPI', 'ProbDrawSPI', 'ProbAwayTeamSPI']
TRAIN_PROB_FD_FEATURES = ['ProbHomeTeamFD', 'ProbDrawFD', 'ProbAwayTeamFD']
TRAIN_KEYS_FEATURES = ['Day', 'Season', 'League', 'HomeTeam', 'AwayTeam', 'Target']
TRAIN_FEATURES = TRAIN_SPI_FEATURES + TRAIN_PROB_SPI_FEATURES + TRAIN_PROB_FD_FEATURES + TRAIN_KEYS_FEATURES

# Various features
KEYS_FEATURES = ['Date', 'League', 'HomeTeam', 'AwayTeam']
RESULTS_FEATURES = ['Days', 'Profit', 'Total profit', 'Precision', 'Bets precision',
                   'Bets precision ratio', 'Predictions ratio', 'Threshold', 'Hyperparameters']

# Simulation parameters
def RATIO_1(y): 
    return {1: int(1.1 * Counter(y)[0])}
def RATIO_2(y): 
    return {1: int(1.2 * Counter(y)[0])}
def RATIO_3(y): 
    return {1: int(1.3 * Counter(y)[0])}
def RATIO_4(y): 
    return {1: int(1.4 * Counter(y)[0])}
ODDS_THRESHOLD = 3.75
GENERATE_WEIGHTS = np.exp
ESTIMATOR = make_pipeline(
    OrdinalEncoder(return_df=False), 
    SMOTE(), 
    XGBClassifier()
)
PARAM_GRID = dict(
    smote__k_neighbors=[2],
    smote__ratio=['auto', RATIO_2, RATIO_3, RATIO_4],
    xgbclassifier__max_depth=[3, 4],
    xgbclassifier__n_estimators=[1000],
    xgbclassifier__learning_rate=[0.01, 0.02],
    xgbclassifier__reg_lambda=[0.1, 1.0, 1.2, 1.5]
)
FIT_PARAMS = dict(
    test_size=0.1,
    xgbclassifier__eval_metric='map',
    xgbclassifier__early_stopping_rounds=10,
    xgbclassifier__verbose=False
)
