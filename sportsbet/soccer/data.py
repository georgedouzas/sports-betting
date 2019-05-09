#!/usr/bin/env python3

"""
Download and prepare training and fixtures data 
from various leagues.
"""

from os.path import join
from itertools import product
from difflib import SequenceMatcher
from sqlite3 import connect
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from scipy.stats import hmean
import numpy as np
import pandas as pd

from sportsbet import SOCCER_PATH
from sportsbet.soccer import TARGETS

DB_CONNECTION = connect(join(SOCCER_PATH, 'soccer.db'))
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
SEASONS = ['1617', '1718', '1819']
SPI_KEYS = ['date', 'league', 'team1', 'team2']
FD_KEYS = ['Date', 'Div', 'HomeTeam', 'AwayTeam']
SPI_INPUT_COLS = ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2']
FD_INPUT_COLS = ['BbAvH', 'BbAvA', 'BbAvD', 'BbAv>2.5', 'BbAv<2.5', 'BbAHh' , 'BbAvAHH', 'BbAvAHA']
OUTPUT_COLS = ['score1', 'score2', 'xg1', 'xg2', 'nsxg1', 'nsxg2', 'adj_score1', 'adj_score2']
ODDS_COLS_MAPPING = {'PSH': 'H', 'PSA': 'A', 'PSD': 'D', 'BbMx>2.5': 'over_2.5', 'BbMx<2.5': 'under_2.5', 'BbAHh': 'handicap', 'BbMxAHH': 'handicap_home', 'BbMxAHA': 'handicap_away'}
INPUT_COLS = SPI_INPUT_COLS + FD_INPUT_COLS


def combine_odds(odds):
    """Combine odds of different targets."""
    combined_odds = 1 / (1 / odds).sum(axis=1)
    return combined_odds


def check_leagues_ids(leagues_ids):
    """Check correct leagues ids input."""
    
    # Set error message
    leagues_ids_error_msg = 'Parameter `leagues_ids` should be equal to `all` or a list that contains any of %s elements. Got %s instead.' % (', '.join(LEAGUES_MAPPING.keys()), leagues_ids)

    # Check types
    if not isinstance(leagues_ids, (str, list)):
        raise TypeError(leagues_ids_error_msg)
    
    # Check values
    if leagues_ids != 'all' and not set(LEAGUES_MAPPING.keys()).issuperset(leagues_ids):
        raise ValueError(leagues_ids_error_msg)
    
    leagues_ids = list(LEAGUES_MAPPING.keys()) if leagues_ids == 'all' else leagues_ids[:]

    return leagues_ids


def create_spi_tables(leagues_ids):
    """Download spi data and save them to database."""

    # Check leagues ids
    leagues_ids = check_leagues_ids(leagues_ids)

    # Download data
    spi = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv').drop(columns=['league_id'])

    # Cast to date
    spi['date'] = pd.to_datetime(spi['date'], format='%Y-%m-%d')

    # Filter leagues
    leagues = [LEAGUES_MAPPING[league_id] for league_id in leagues_ids]
    spi = spi[spi['league'].isin(leagues)]

    # Convert league names to ids
    inverse_leagues_mapping = {league: league_id for league_id, league in LEAGUES_MAPPING.items()}
    spi['league'] = spi['league'].apply(lambda league: inverse_leagues_mapping[league])

    # Filter matches
    mask = (~spi['score1'].isna()) & (~spi['score2'].isna())
    spi_historical, spi_fixtures = spi[mask], spi[~mask]

    return spi_historical, spi_fixtures


def create_fd_tables(leagues_ids):
    """Download fd data and save them to database."""

    # Check leagues ids
    leagues_ids = check_leagues_ids(leagues_ids)

    # Define parameters
    base_url = 'http://www.football-data.co.uk'

    # Download historical data
    fd_historical = []
    for league_id, season in product(leagues_ids, SEASONS):
        data = pd.read_csv(join(base_url, 'mmz4281', season, league_id), usecols=FD_KEYS + FD_INPUT_COLS + list(ODDS_COLS_MAPPING.keys()))
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        data['season'] = season
        fd_historical.append(data)
    fd_historical = pd.concat(fd_historical, ignore_index=True)

    # Download fixtures data
    fd_fixtures = pd.read_csv(join(base_url, 'fixtures.csv'), usecols=FD_KEYS + FD_INPUT_COLS + list(ODDS_COLS_MAPPING.keys()))
    fd_fixtures['Date'] = pd.to_datetime(fd_fixtures['Date'], dayfirst=True)
    fd_fixtures = fd_fixtures[fd_fixtures['Div'].isin(leagues_ids)]

    return fd_historical, fd_fixtures


def create_names_mapping_table(left_data, right_data):
    """Create names mapping table."""

    # Rename columns
    key_columns = ['key0', 'key1']
    left_data.columns = key_columns + ['left_team1', 'left_team2']
    right_data.columns = key_columns + ['right_team1', 'right_team2']

    # Generate teams names combinations
    names_combinations = pd.merge(left_data, right_data, how='outer').dropna().drop(columns=key_columns).reset_index(drop=True)

    # Calculate similarity index
    similarity = names_combinations.apply(lambda row: SequenceMatcher(None, row.left_team1, row.right_team1).ratio() * SequenceMatcher(None, row.left_team2, row.right_team2).ratio(), axis=1)

    # Append similarity index
    names_combinations_similarity = pd.concat([names_combinations, similarity], axis=1)

    # Filter correct matches
    indices = names_combinations_similarity.groupby(['left_team1', 'left_team2'])[0].idxmax().values
    names_matching = names_combinations.take(indices)

    # Teams matching
    matching1 = names_matching.loc[:, ['left_team1', 'right_team1']].rename(columns={'left_team1': 'left_team', 'right_team1': 'right_team'})
    matching2 = names_matching.loc[:, ['left_team2', 'right_team2']].rename(columns={'left_team2': 'left_team', 'right_team2': 'right_team'})
        
    # Combine matching
    matching = matching1.append(matching2)
    matching = matching.groupby(matching.columns.tolist()).size().reset_index()
    indices = matching.groupby('left_team')[0].idxmax().values
        
    # Generate mapping
    names_mapping = matching.take(indices).drop(columns=0).reset_index(drop=True)

    return names_mapping


def create_modeling_tables(spi_historical, spi_fixtures, fd_historical, fd_fixtures, names_mapping):
    """Create tables for machine learning modeling."""

    # Rename teams
    for col in ['team1', 'team2']:
        for df in (spi_historical, spi_fixtures):
            df = pd.merge(df, names_mapping, left_on=col, right_on='left_team', how='left').drop(columns=[col, 'left_team']).rename(columns={'right_team': col})

    # Combine data
    historical = pd.merge(spi_historical, fd_historical, left_on=SPI_KEYS, right_on=FD_KEYS).dropna(subset=ODDS_COLS_MAPPING.keys(), how='any').reset_index(drop=True)
    fixtures = pd.merge(spi_fixtures, fd_fixtures, left_on=SPI_KEYS, right_on=FD_KEYS)

    # Extract training, odds and fixtures
    X = historical.loc[:, ['season'] + SPI_KEYS + INPUT_COLS]
    y = historical.loc[:, OUTPUT_COLS]
    odds = historical.loc[:, SPI_KEYS + list(ODDS_COLS_MAPPING.keys())].rename(columns=ODDS_COLS_MAPPING)
    X_test = fixtures.loc[:, SPI_KEYS + INPUT_COLS]
    odds_test = fixtures.loc[:, SPI_KEYS + list(ODDS_COLS_MAPPING.keys())].rename(columns=ODDS_COLS_MAPPING)

    # Add average scores columns
    for ind in (1, 2):
        avg_score =  y[['adj_score%s' % ind, 'xg%s' % ind, 'nsxg%s' % ind]].mean(axis=1)
        avg_score[avg_score.isna()] = y['score%s' % ind]
        y['avg_score%s' % ind] = avg_score

    # Add combined odds columns
    for target, _ in TARGETS:
        if '+' in target:
            targets = target.split('+')
            odds[target] = combine_odds(odds[targets])
            odds_test[target] = combine_odds(odds_test[targets])

    # Feature extraction
    with np.errstate(divide='ignore', invalid='ignore'):
        for df in (X, X_test):
            df['quality'] = hmean(df[['spi1', 'spi2']], axis=1)
            df['importance'] = df[['importance1', 'importance2']].mean(axis=1)
            df['rating'] = df[['quality', 'importance']].mean(axis=1)
            df['sum_proj_score'] = df['proj_score1'] + df['proj_score2']
    
    return X, y, odds, X_test, odds_test


def download():
    """Command line function to download data and update database."""
    
    # Create parser description
    description = 'Select the leagues parameter from the following leagues:\n\n'
    for league_id, league_name in LEAGUES_MAPPING.items():
        description += '{} ({})\n'.format(league_id, league_name)

    # Create parser
    parser = ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)

    # Add arguments
    parser.add_argument('leagues', nargs='*', default=['all'], help='One of all or any league ids from above.')
    
    # Parse arguments
    args = parser.parse_args()

    # Extract leagues ids
    leagues_ids = args.leagues
    if len(leagues_ids) == 1 and leagues_ids[0] == 'all':
        leagues_ids = leagues_ids[0]

    # Create historical, fixtures and names mapping tables
    spi_historical, spi_fixtures = create_spi_tables(leagues_ids)
    fd_historical, fd_fixtures = create_fd_tables(leagues_ids)
    names_mapping = create_names_mapping_table(spi_historical[SPI_KEYS], fd_historical[FD_KEYS])

    # Create modeling tables
    X, y, odds, X_test, odds_test = create_modeling_tables(spi_historical, spi_fixtures, fd_historical, fd_fixtures, names_mapping)

    # Save modeling tables
    for name, df in zip(['X', 'y', 'odds', 'X_test', 'odds_test'], [X, y, odds, X_test, odds_test]):
        df.to_sql(name, DB_CONNECTION, index=False, if_exists='replace')
