from collections import OrderedDict
from os.path import join
from itertools import product

# Parameters
TEAMS_MAPPING = OrderedDict({
    'Borussia Monchengladbach': "M'gladbach",
    'Brighton and Hove Albion': 'Brighton',
    'Deportivo La Coruña': 'La Coruna',
    'Internazionale': 'Inter',
    'Chievo Verona': 'Chievo'
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
RESULTS_MAPPING = OrderedDict({'H': 0, 'D': 1, 'A': 2})
YEARS = ['1617', '1718']
MIN_N_MATCHES = 20
SEASON_STARTING_DAY = {'16-17': 0, '17-18': 363}

# SPI data
SPI_URL = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
SPI_DATA_FEATURES = ['date', 'league', 'team1', 'team2', 'spi1', 'spi2', 'prob1', 'prob2', 'probtie']

# Football data
FD_URL = 'http://www.football-data.co.uk/mmz4281'
FD_SUFFIX = [join(year, league) for year, league in product(YEARS, LEAGUES_MAPPING.values())]
FD_URLS = [join(FD_URL, suffix) for suffix in FD_SUFFIX]

FD_MAX_ODDS, FD_AVG_ODDS = 'BbMx', 'BbAv'
FD_MAX_ODDS_FEATURES = [FD_MAX_ODDS + result for result in RESULTS_MAPPING.keys()]
FD_AVG_ODDS_FEATURES = [FD_AVG_ODDS + result for result in RESULTS_MAPPING.keys()]
FD_ID_FEATURES = ['Div', 'Date', 'Season', 'HomeTeam', 'AwayTeam']
FD_DATA_FEATURES = FD_MAX_ODDS_FEATURES + FD_AVG_ODDS_FEATURES + FD_ID_FEATURES + ['FTR']

# Training data
SPI_FEATURES = ['HomeTeamSPI', 'AwayTeamSPI']
PROB_SPI_FEATURES = ['ProbHomeTeamSPI', 'ProbDrawSPI', 'ProbAwayTeamSPI']
PROB_FD_FEATURES = ['ProbHomeTeamFD', 'ProbDrawFD', 'ProbAwayTeamFD']
ID_FEATURES = ['Day', 'Season', 'League', 'HomeTeam', 'AwayTeam', 'Target']
TRAINING_FEATURES = SPI_FEATURES + PROB_SPI_FEATURES + PROB_FD_FEATURES + ID_FEATURES

# ID features
KEYS_FEATURES = ['Date', 'League', 'HomeTeam', 'AwayTeam']



