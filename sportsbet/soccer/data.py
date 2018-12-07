"""
Download and prepare historical and upcoming matches 
data from various leagues.
"""

from os.path import join
from pathlib import Path
from itertools import product
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import PATH

LEAGUES_MAPPING = {
    'E0': 'Barclays Premier League',
    'B1': 'Belgian Jupiler League',
    'N1': 'Dutch Eredivisie',
    'E1': 'English League Championship',
    'E2': 'English League One',
    'E3': 'English League Two',
    'F1': 'French Ligue 1',
    'F2': 'French Ligue 2',
    'D1': 'German Bundesliga',
    'D2': 'German 2. Bundesliga',
    'G1': 'Greek Super League',
    'I1': 'Italy Serie A',
    'I2': 'Italy Serie B',
    'P1': 'Portuguese Liga',
    'SC0': 'Scottish Premiership',
    'SP1': 'Spanish Primera Division',
    'SP2': 'Spanish Segunda Division',
    'T1': 'Turkish Turkcell Super Lig'
}
SPI_COLUMNS_MAPPING = {
    'league': 'League',
    'date': 'Date',
    'team1': 'Home Team',
    'team2': 'Away Team',
    'score1': 'Home Goals',
    'score2': 'Away Goals',
    'spi1': 'Home SPI',
    'spi2': 'Away SPI',
    'prob1': 'Home SPI Probabilities',
    'prob2': 'Away SPI Probabilities',
    'probtie': 'Draw SPI Probabilities',
    'proj_score1': 'Home SPI Goals',
    'proj_score2': 'Away SPI Goals'
}
FD_COLUMNS_MAPPING = {
    'Div': 'League',
    'Date': 'Date',
    'Season': 'Season',
    'Month': 'Month',
    'Day': 'Day',
    'HomeTeam': 'Home Team',
    'AwayTeam': 'Away Team',
    'FTHG': 'Home Goals',
    'FTAG': 'Away Goals',
    'BbAvH': 'Home Average Odds',
    'BbAvA': 'Away Average Odds',
    'BbAvD': 'Draw Average Odds',
    'BbMxH': 'Home Maximum Odds',
    'BbMxA': 'Away Maximum Odds',
    'BbMxD': 'Draw Maximum Odds',
    'PSH': 'Home Pinnacle Odds',
    'PSA': 'Away Pinnacle Odds',
    'PSD': 'Draw Pinnacle Odds',
    'B365H': 'Home Bet365 Odds',
    'B365A': 'Away Bet365 Odds',
    'B365D': 'Draw Bet365 Odds',
    'BWH': 'Home bwin Odds',
    'BWA': 'Away bwin Odds',
    'BWD': 'Draw bwin Odds'
}
SEASONS = ['1617', '1718', '1819']
HISTORICAL_DATA_PATH = join(PATH, 'historical_data.csv')
PREDICTIONS_DATA_PATH = join(PATH, 'predictions_data.csv')


def validate_leagues(leagues):
    """Validate leagues input."""
    valid_leagues = [league_id for league_id in LEAGUES_MAPPING.keys()]
    if leagues != 'all' and not set(leagues).issubset(valid_leagues):
        msg = "The `leagues` parameter should be either equal to 'all' or a list of valid league ids. Got {} instead."
        raise ValueError(msg.format(leagues))


def match_teams_names(teams_names):
    """Match teams names between spi and fd data."""

    # Calculate similarity index
    similarity = teams_names.apply(lambda row: SequenceMatcher(None, row[0], row[1]).ratio() * SequenceMatcher(None, row[2], row[3]).ratio(), axis=1)

    # Append similarity index
    teams_names_similarity = pd.concat([teams_names, similarity], axis=1)

    # Filter correct matches
    indices = teams_names_similarity.groupby(['Home Team_x', 'Away Team_x'])[0].idxmax().values
    teams_names_matching = teams_names.take(indices)

    # Generate mapping
    matching1 = teams_names_matching.iloc[:, 0:2]
    matching2 = teams_names_matching.iloc[:, 2:]
    matching1.columns, matching2.columns = ['x', 'y'], ['x', 'y']
    matching = matching1.append(matching2)
    matching = matching.groupby(['x', 'y']).size().reset_index()
    indices = matching.groupby('x')[0].idxmax().values
    matching = matching.take(indices)
    mapping = dict(zip(matching.x, matching.y))

    return mapping


def fetch_spi_data(leagues, data_type):
    """Fetch the data containing match-by-match SPI ratings."""

    # Define url
    url = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'

    # Download data
    data = pd.read_csv(url)
    
    # Select and rename columns
    data = data.loc[:, SPI_COLUMNS_MAPPING.keys()]
    data.rename(columns=SPI_COLUMNS_MAPPING, inplace=True)

    # Filter leagues
    leagues = [LEAGUES_MAPPING[league_id] for league_id in (leagues if leagues != 'all' else LEAGUES_MAPPING.keys())]
    data = data[data.League.isin(leagues)]

    # Rename leagues
    inverse_leagues_mapping = {league_name: league_id for league_id, league_name in LEAGUES_MAPPING.items()}
    data.loc[:, 'League'] = data.loc[:, 'League'].apply(lambda league_name: inverse_leagues_mapping[league_name])

    # Filter historical data
    if data_type == 'historical':
        data = data[(~data['Home Goals'].isna()) & (~data['Away Goals'].isna())]
    elif data_type == 'predictions':
        data = data[(data['Home Goals'].isna()) & (data['Away Goals'].isna())]

    # Cast columns
    data.loc[:, 'Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    if data_type == 'historical':
        data['Home Goals'] = data['Home Goals'].astype(int)
        data['Away Goals'] = data['Away Goals'].astype(int)

    # Sort data
    data = data.sort_values(['Date', 'League', 'Home Team', 'Away Team']).reset_index(drop=True)

    return data


def fetch_fd_data(leagues, data_type):
    """Fetch the data from football-data.co.uk."""

    # Define leagues
    leagues = LEAGUES_MAPPING.keys() if leagues == 'all' else leagues

    if data_type == 'historical':

        # Define url
        url = 'http://www.football-data.co.uk/mmz4281'

        # Generate urls
        urls = [join(url, year, league_id) for year, league_id in product(['1617', '1718', '1819'], leagues)]

    elif data_type == 'predictions':
        
        # Define url
        urls = ['http://www.football-data.co.uk/fixtures.csv']

    # Download data
    data = pd.DataFrame()
    for url in tqdm(urls, desc='Download %s data' % data_type):

        # Create partial dataframe
        partial_data = pd.read_csv(url)

        # Cast columns
        partial_data['Date'] = pd.to_datetime(partial_data['Date'], format='%d/%m/%y')

        # Create columns
        partial_data['Season'] = url.split('/')[-2] if data_type == 'historical' else SEASONS[-1]
        partial_data['Month'] = partial_data['Date'].apply(lambda date: date.month)
        partial_data['Day'] = partial_data['Date'].apply(lambda date: date.day)
        
        # Select and rename columns
        partial_data = partial_data.loc[:, FD_COLUMNS_MAPPING.keys()]
        partial_data.rename(columns=FD_COLUMNS_MAPPING, inplace=True)
        
        # Append data
        data = data.append(partial_data, ignore_index=True)
    
    # Filter leagues
    if data_type == 'predictions':
        data = data[data.League.isin(leagues)]

    # Sort data
    data = data.sort_values(['Date', 'League', 'Home Team', 'Away Team']).reset_index(drop=True)

    return data


def download_data(leagues, data_type):
    """Download and save historical or predictions data."""

    # Validate leagues
    validate_leagues(leagues)

    # Define keys
    keys = ['Date', 'League', 'Home Team', 'Away Team', 'Home Goals', 'Away Goals']
    
    # Fetch data
    spi_data, fd_data = fetch_spi_data(leagues, data_type), fetch_fd_data(leagues, data_type)

    # Teams names matching
    teams_names_columns = ['Home Team_x', 'Home Team_y', 'Away Team_x', 'Away Team_y']
    teams_names = pd.merge(spi_data, fd_data, on=['Date', 'League'], how='outer').loc[:, teams_names_columns].dropna().reset_index(drop=True)
    try:
        mapping = match_teams_names(teams_names)
    except ValueError:
        raise ValueError('No common upcoming matches between SPI and FD data sources were found.')

    # Convert names
    spi_data['Home Team'] = spi_data['Home Team'].apply(lambda team: mapping[team] if team in mapping.keys() else team)
    spi_data['Away Team'] = spi_data['Away Team'].apply(lambda team: mapping[team] if team in mapping.keys() else team)

    # Combine data
    if data_type == 'historical':
        data = pd.merge(spi_data, fd_data, on=keys)
    elif data_type == 'predictions':
        data = pd.merge(spi_data.drop(columns=['Home Goals', 'Away Goals']), fd_data.drop(columns=['Date', 'Home Goals', 'Away Goals']), on=keys[1:-2])

    # Sort data
    data = data.sort_values(keys[:-2]).reset_index(drop=True)

    # Save data
    Path(PATH).mkdir(exist_ok=True)
    data.to_csv(HISTORICAL_DATA_PATH if data_type == 'historical' else PREDICTIONS_DATA_PATH, index=False)


def load_data(data_type, max_odds):
    """Load the data used for model training and predictions."""

    # Load data
    data_path = HISTORICAL_DATA_PATH if data_type == 'historical' else PREDICTIONS_DATA_PATH
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError('%s data do not exist. Download them before loading.' % data_type.capitalize())
        
    # Define parameters
    input_spi_cols = [col for col in data.columns if 'SPI' in col]
    input_odds_cols = [col for col in data.columns if 'Maximum' in col or 'Average' in col]
    max_odds_cols = [col for col in data.columns if 'Odds' in col and col.split(' ')[1].lower() in max_odds]
    matches_cols = ['League', 'Season', 'Date', 'Month', 'Day', 'Home Team', 'Away Team']

    # Filter data
    input_odds_mask = (~data[input_odds_cols].isnull()).product(axis=1).astype(bool)
    max_odds_mask = (data[max_odds_cols].isnull().sum(axis=1) != len(max_odds_cols))
    data = data[input_odds_mask & max_odds_mask].reset_index(drop=True)

    # Input data
    X = pd.concat([data.loc[:, input_odds_cols], data.loc[:, input_spi_cols]], axis=1)
    X['Difference SPI Goals'] = X['Home SPI Goals'] - X['Away SPI Goals']
    X['Difference SPI'] = X['Home SPI'] - X['Away SPI']
    X['Difference SPI Probabilities'] = X['Home SPI Probabilities'] - X['Away SPI Probabilities']
    X = X.values

    # Maximum odds data
    odds = data.loc[:, max_odds_cols]
    grouped_cols = [np.take(max_odds_cols, range(i, len(max_odds_cols) + i, 3)).tolist() for i in range(3)]
    for cols in grouped_cols:
        odds['%s Odd' % cols[0][0]] = np.nanmax(odds[cols], axis=1)
    odds = odds.drop(columns=max_odds_cols)

    # Match data
    matches = data[matches_cols]

    # Target
    if data_type == 'historical':
        y = (data['Home Goals'] - data['Away Goals']).apply(lambda sign: 'H' if sign > 0 else 'D' if sign == 0 else 'A').values
    elif data_type == 'predictions':
        y = None
            
    return X, y, odds, matches
