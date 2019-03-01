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
from sportsbet.soccer import TARGET_TYPES_MAPPING

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


def combine_odds(odds, target_types):
    """Combine odds of different betting types."""
    combined_odds = 1 / pd.concat([1 / odds[target_type] for target_type in target_types], axis=1).sum(axis=1)
    combined_odds.name = '+'.join(target_types)
    return pd.concat([odds, combined_odds], axis=1)


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

    # Save tables
    for name, df in zip(['spi_historical', 'spi_fixtures'], [spi_historical, spi_fixtures]):
        df.to_sql(name, DB_CONNECTION, index=False, if_exists='replace')


def create_fd_tables(leagues_ids):
    """Download fd data and save them to database."""

    # Define parameters
    base_url = 'http://www.football-data.co.uk'
    cols = ['Date', 'Div', 'HomeTeam', 'AwayTeam']
    features_cols = ['BbAvH', 'BbAvA', 'BbAvD', 'BbAv>2.5', 'BbAv<2.5', 'BbAHh' , 'BbAvAHH', 'BbAvAHA']
    odds_cols = ['PSH', 'PSA', 'PSD', 'BbMx>2.5', 'BbMx<2.5', 'BbAHh', 'BbMxAHH', 'BbMxAHA']
    seasons = ['1617', '1718', '1819']

    # Check leagues ids
    leagues_ids = check_leagues_ids(leagues_ids)

    # Download historical data
    fd_historical = []
    for league_id, season in product(leagues_ids, seasons):
        data = pd.read_csv(join(base_url, 'mmz4281', season, league_id), usecols=cols + features_cols + odds_cols)
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        data['season'] = season
        fd_historical.append(data)
    fd_historical = pd.concat(fd_historical, ignore_index=True)

    # Download fixtures data
    fd_fixtures = pd.read_csv(join(base_url, 'fixtures.csv'), usecols=cols + features_cols + odds_cols)
    fd_fixtures['Date'] = pd.to_datetime(fd_fixtures['Date'], dayfirst=True)
    fd_fixtures = fd_fixtures[fd_fixtures['Div'].isin(leagues_ids)]

    # Save tables
    for name, df in zip(['fd_historical', 'fd_fixtures'], [fd_historical, fd_fixtures]):
        df.to_sql(name, DB_CONNECTION, index=False, if_exists='replace')


def create_names_mapping_table():
    """Create names mapping table."""

    # Load data
    left_data = pd.read_sql('select date, league, team1, team2 from spi_historical', DB_CONNECTION)
    right_data = pd.read_sql('select Date, Div, HomeTeam, AwayTeam from fd_historical', DB_CONNECTION)

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

    # Save table
    names_mapping.to_sql('names_mapping', DB_CONNECTION, index=False, if_exists='replace')


def create_modeling_tables():
    """Create tables for machine learning modeling."""

    # Define parameters
    spi_keys = ['date', 'league', 'team1', 'team2']
    fd_keys = ['Date', 'Div', 'HomeTeam', 'AwayTeam']
    input_cols = ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1', 'importance2', 'BbAvH', 'BbAvA', 'BbAvD', 'BbAv>2.5', 'BbAv<2.5', 'BbAHh', 'BbAvAHH', 'BbAvAHA']
    output_cols = ['score1', 'score2', 'xg1', 'xg2', 'nsxg1', 'nsxg2', 'adj_score1', 'adj_score2']
    odds_cols_mapping = {'PSH': 'H', 'PSA': 'A', 'PSD': 'D', 'BbMx>2.5': 'over_2.5', 'BbMx<2.5': 'under_2.5', 'BbAHh': 'handicap', 'BbMxAHH': 'handicap_home', 'BbMxAHA': 'handicap_away'}
    
    # Load data
    data = {}
    for name in ('spi_historical', 'spi_fixtures', 'fd_historical', 'fd_fixtures', 'names_mapping'):
        parse_dates = ['date'] if name in ('spi_historical', 'spi_fixtures') else ['Date'] if name in ('fd_historical', 'fd_fixtures') else None
        data[name] = pd.read_sql('select * from %s' % name, DB_CONNECTION, parse_dates=parse_dates)

    # Rename teams
    for col in ['team1', 'team2']:
        for name in ('spi_historical', 'spi_fixtures'):
            data[name] = pd.merge(data[name], data['names_mapping'], left_on=col, right_on='left_team', how='left').drop(columns=[col, 'left_team']).rename(columns={'right_team': col})

    # Combine data
    historical = pd.merge(data['spi_historical'], data['fd_historical'], left_on=spi_keys, right_on=fd_keys).dropna(subset=odds_cols_mapping.keys(), how='any').reset_index(drop=True)
    fixtures = pd.merge(data['spi_fixtures'], data['fd_fixtures'], left_on=spi_keys, right_on=fd_keys)

    # Extract training, odds and fixtures
    X = historical.loc[:, ['season'] + spi_keys + input_cols]
    y = historical.loc[:, output_cols]
    odds = historical.loc[:, spi_keys + list(odds_cols_mapping.keys())].rename(columns=odds_cols_mapping)
    X_test = fixtures.loc[:, spi_keys + input_cols]
    odds_test = fixtures.loc[:, spi_keys + list(odds_cols_mapping.keys())].rename(columns=odds_cols_mapping)

    # Add average scores columns
    for ind in (1, 2):
        y['avg_score%s' % ind] =  y[['score%s' % ind, 'xg%s' % ind, 'nsxg%s' % ind]].mean(axis=1)

    # Add combined odds columns
    for target_type in TARGET_TYPES_MAPPING.keys():
        if '+' in target_type:
            target_types = target_type.split('+')
            odds = combine_odds(odds, target_types)
            odds_test = combine_odds(odds_test, target_types)

    # Feature extraction
    with np.errstate(divide='ignore',invalid='ignore'):
        for df in (X, X_test):
            df['quality'] = hmean(df[['spi1', 'spi2']], axis=1)
            df['importance'] = df[['importance1', 'importance2']].mean(axis=1)
            df['rating'] = df[['quality', 'importance']].mean(axis=1)
            df['sum_proj_score'] = df['proj_score1'] + df['proj_score2']

    # Save tables
    for name, df in zip(['X', 'y', 'odds', 'X_test', 'odds_test'], [X, y, odds, X_test, odds_test]):
        df.to_sql(name, DB_CONNECTION, index=False, if_exists='replace')


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

    # Adjust parameter
    leagues = args.leagues
    if len(leagues) == 1 and leagues[0] == 'all':
        leagues = leagues[0]

    # Create tables
    for ind, func in enumerate([create_spi_tables, create_fd_tables, create_names_mapping_table, create_modeling_tables]):
        func(leagues) if ind in (0, 1) else func()

    


