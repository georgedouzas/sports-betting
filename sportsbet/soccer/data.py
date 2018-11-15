"""
Download and prepare historical and upcoming matches 
data from various leagues.
"""

from os.path import join
from urllib.request import urljoin
from itertools import product
from difflib import SequenceMatcher

import pandas as pd
from tqdm import tqdm

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
    'HomeTeam': 'Home Team',
    'AwayTeam': 'Away Team',
    'FTHG': 'Home Goals',
    'FTAG': 'Away Goals',
    'BbAvH': 'Home Average Odds',
    'BbAvA': 'Away Average Odds',
    'BbAvD': 'Draw Average Odds',
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


def _fetch_spi_data(leagues):
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

    return data
    

def _fetch_historical_spi_data(leagues):
    """Fetch historical data containing match-by-match SPI ratings."""

    # Fetch data
    data = _fetch_spi_data(leagues)

    # Filter historical data
    data = data[(~data['Home Goals'].isna()) & (~data['Away Goals'].isna())]

    # Cast columns
    data.loc[:, 'Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data['Home Goals'] = data['Home Goals'].astype(int)
    data['Away Goals'] = data['Away Goals'].astype(int)

    # Sort data
    data = data.sort_values(['Date', 'League', 'Home Team', 'Away Team']).reset_index(drop=True)
    
    return data


def _fetch_predictions_spi_data(leagues):
    """Fetch data containing future match-by-match SPI ratings."""

    # Fetch data
    data = _fetch_spi_data(leagues)

    # Filter future data
    data = data[(data['Home Goals'].isna()) & (data['Away Goals'].isna())]

    # Cast to date
    data.loc[:, 'Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

    # Sort data
    data = data.sort_values(['Date', 'League', 'Home Team', 'Away Team']).reset_index(drop=True)

    return data


def _fetch_fd_data(urls):
    """Fetch the data from football-data.co.uk."""

    # Download data
    data = pd.DataFrame()
    for url in tqdm(urls, desc='Download data'):

        # Create partial dataframe
        partial_data = pd.read_csv(url)
        
        # Select and rename columns
        partial_data = partial_data.loc[:, FD_COLUMNS_MAPPING.keys()]
        partial_data.rename(columns=FD_COLUMNS_MAPPING, inplace=True)
        
        # Append data
        data = data.append(partial_data, ignore_index=True)

    return data


def _fetch_historical_fd_data(leagues):
    """Fetch the historical data from football-data.co.uk."""

    # Define url
    url = 'http://www.football-data.co.uk/mmz4281'

    # Define leagues
    leagues = LEAGUES_MAPPING.keys() if leagues == 'all' else leagues

    # Generate urls
    urls = [join(url, year, league_id) for year, league_id in product(['1617', '1718', '1819'], leagues)]

    # Fetch data
    data = _fetch_fd_data(urls)

    # Filter matches
    data = data[(~data['Home Goals'].isna()) & (~data['Away Goals'].isna())]

    # Cast columns
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')
    data['Home Goals'] = data['Home Goals'].astype(int)
    data['Away Goals'] = data['Away Goals'].astype(int)

    # Sort data
    data = data.sort_values(['Date', 'League', 'Home Team', 'Away Team']).reset_index(drop=True)
    
    return data


def _fetch_predictions_fd_data(leagues):
    """Fetch the data from football-data.co.uk containing future matches."""

    # Define url
    url = 'http://www.football-data.co.uk/fixtures.csv'

    # Fetch data
    data = _fetch_fd_data([url])

    # Filter leagues
    leagues = LEAGUES_MAPPING.keys() if leagues == 'all' else leagues
    data = data[data.League.isin(leagues)]

    # Cast to date
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')

    # Sort data
    data = data.sort_values(['Date', 'League', 'Home Team', 'Away Team']).reset_index(drop=True)

    return data


def _match_teams_names(teams_names):
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