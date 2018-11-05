"""
Download and prepare soccer historical data from various leagues.
"""

from os.path import join
from itertools import product
from difflib import SequenceMatcher

import pandas as pd

LEAGUES_MAPPING = {
    'Argentina Primera Division': ('ARG', 'extra'),
    'Austrian T-Mobile Bundesliga': ('AUT', 'extra'),
    'Barclays Premier League': ('E0', 'main'),
    'Belgian Jupiler League': ('B1', 'main'),
    'Brasileiro Série A': ('BRA', 'extra'),
    'Chinese Super League': ('CHN', 'extra'),
    'Danish SAS-Ligaen': ('DNK', 'extra'),
    'Dutch Eredivisie': ('N1', 'main'),
    'English League Championship': ('E1', 'main'),
    'English League One': ('E2', 'main'),
    'English League Two': ('E3', 'main'),
    'French Ligue 1': ('F1', 'main'),
    'French Ligue 2': ('F2', 'main'),
    'German Bundesliga': ('D1', 'main'),
    'German 2. Bundesliga': ('D2', 'main'),
    'Greek Super League': ('G1', 'main'),
    'Italy Serie A': ('I1', 'main'),
    'Italy Serie B': ('I2', 'main'),
    'Japanese J League': ('JPN', 'extra'),
    'Major League Soccer': ('USA', 'extra'),
    'Norwegian Tippeligaen': ('NOR', 'extra'),
    'Portuguese Liga': ('P1', 'main'),
    'Russian Premier Liga': ('RUS', 'extra'),
    'Scottish Premiership': ('SC0', 'main'),
    'Spanish Primera Division': ('SP1', 'main'),
    'Spanish Segunda Division': ('SP2', 'main'),
    'Swedish Allsvenskan': ('SWE', 'extra'),
    'Swiss Raiffeisen Super League': ('SWZ', 'extra'),
    'Turkish Turkcell Super Lig': ('T1', 'main')
}


def _fetch_spi_data(leagues):
    """Fetch the data containing match-by-match SPI ratings."""

    # Define url
    url = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
    
    # Define columns mapping
    columns_mapping = {
        'league': 'League',
        'date': 'Date',
        'team1': 'HomeTeam',
        'team2': 'AwayTeam',
        'score1': 'HomeGoals',
        'score2': 'AwayGoals',
        'spi1': 'HomeSPI',
        'spi2': 'AwaySPI',
        'prob1': 'HomeSPIProb',
        'prob2': 'AwaySPIProb',
        'probtie': 'DrawSPIProb',
        'proj_score1': 'HomeSPIGoals',
        'proj_score2': 'AwaySPIGoals'
    }

    # Download data
    data = pd.read_csv(url)
    
    # Select and rename columns
    data = data.loc[:, columns_mapping.keys()]
    data.rename(columns=columns_mapping, inplace=True)

    # Filter and rename leagues
    data = data[data.League.isin(LEAGUES_MAPPING.keys())]
    data['League'] = data['League'].apply(lambda league: LEAGUES_MAPPING[league][0])

    # Filter matches
    if leagues == 'all':
        leagues = [league_id for league_id, _ in LEAGUES_MAPPING.values()]
    elif leagues == 'main':
        leagues = [league_id for league_id, league_type in LEAGUES_MAPPING.values() if league_type == 'main']
    else:
        leagues = [league_id for league_id, _ in LEAGUES_MAPPING.values() if league_id in leagues]
    data = data[data.League.isin(leagues)]

    return data
    

def _fetch_historical_spi_data(leagues):
    """Fetch historical data containing match-by-match SPI ratings."""

    # Fetch data
    data = _fetch_spi_data(leagues)

    # Filter historical data
    data = data[(~data.HomeGoals.isna()) & (~data.AwayGoals.isna())]

    # Cast columns
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data['HomeGoals'] = data['HomeGoals'].astype(int)
    data['AwayGoals'] = data['AwayGoals'].astype(int)

    # Sort data
    data = data.sort_values(['Date', 'League', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)
    
    return data


def _fetch_future_spi_data(leagues):
    """Fetch data containing future match-by-match SPI ratings."""

    # Fetch data
    data = _fetch_spi_data(leagues)

    # Filter future data
    data = data[(data.HomeGoals.isna()) & (data.AwayGoals.isna())]

    return data


def _fetch_historical_fd_data(leagues):
    """Fetch the data from football-data.co.uk."""

    # Define common parameters
    base_url = 'http://www.football-data.co.uk'
    url_part_main, url_part_extra = 'mmz4281', 'new'

    # Append leagues type
    if leagues == 'all':
        leagues_types = LEAGUES_MAPPING.values()
    elif leagues == 'main':
        leagues_types = [(league_id, league_type) for league_id, league_type in LEAGUES_MAPPING.values() if league_type == 'main']
    else:
        leagues_types = [(league_id, league_type) for league_id, league_type in LEAGUES_MAPPING.values() if league_id in leagues]

    # Define urls
    suffixes = []
    for league_id, league_type in leagues_types:
        if league_type == 'main':
            suffixes += [join(url_part_main, year, league_id) for year, league in product(['1617', '1718', '1819'], [league_id])]
        elif league_type == 'extra':
            suffixes.append(join(url_part_extra, league_id))
    urls = [join(base_url, suffix) for suffix in suffixes]

    # Define columns mapping
    columns = ['League', 'Season', 'Date', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals', 
               'HomeAverageOdd', 'AwayAverageOdd', 'DrawAverageOdd', 'HomeMaximumOdd', 
               'AwayMaximumOdd', 'DrawMaximumOdd']
    main_columns = ['Div', 'Season', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 
                    'BbAvH', 'BbAvA', 'BbAvD', 'BbMxH', 'BbMxA', 'BbMxD']
    extra_columns = ['League', 'Season', 'Date', 'Home', 'Away', 'HG', 'AG', 'AvgH',
                     'AvgA', 'AvgD', 'MaxH', 'MaxA', 'MaxD']
    columns_mapping = {'main': dict(zip(main_columns, columns)), 'extra': dict(zip(extra_columns, columns))}

    # Download data
    data = pd.DataFrame()
    for url in urls:

        # Create temporary dataframe
        partial_data = pd.read_csv(url)
        
        # Main leagues
        if url_part_main in url:
            season = url.split('/')[-2]
            partial_data['Season'] = season[:2] + '-' + season[2:]
            partial_data = partial_data.loc[:, columns_mapping['main'].keys()]
            partial_data.rename(columns=columns_mapping['main'], inplace=True)

        # Extra leagues
        elif url_part_extra in url:
            mask1 = partial_data.Season.isin([2016, 2017, 2018])
            mask2 = partial_data.Season.isin(['2016/2017', '2017/2018', '2018/2019'])
            partial_data = partial_data.loc[mask1 | mask2, :]
            partial_data['League'] = url.split('/')[-1]
            if mask2.sum() == 0:
                partial_data['Season'] = partial_data['Season'].apply(lambda year: str(year)[2:] + '-' + str(year + 1)[2:])
            elif mask1.sum() == 0:
                partial_data['Season'] = partial_data['Season'].apply(lambda season: '-'.join([year[2:] for year in  season.split('/')]))
            partial_data = partial_data.loc[:, columns_mapping['extra'].keys()]
            partial_data.rename(columns=columns_mapping['extra'], inplace=True)
        
        # Append data
        data = data.append(partial_data, ignore_index=True)

    # Filter only played matches
    data = data[(~data.HomeGoals.isna()) & (~data.AwayGoals.isna())]

    # Cast columns
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')
    data['HomeGoals'] = data['HomeGoals'].astype(int)
    data['AwayGoals'] = data['AwayGoals'].astype(int)

    # Sort data
    data = data.sort_values(['Date', 'League', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)
    
    return data


def _match_teams_names(spi_data, fd_data):
    """Match teams names between spi and fd data."""

    # Define merge keys
    keys = ['Date', 'League', 'HomeGoals', 'AwayGoals']

    # Define columns to select
    columns = ['HomeTeam_x', 'HomeTeam_y', 'AwayTeam_x', 'AwayTeam_y']

    # Merge data
    teams_names = pd.merge(spi_data, fd_data, on=keys, how='left').dropna().loc[:, columns].reset_index(drop=True)

    # Calculate similarity index
    similarity = teams_names.apply(lambda row: SequenceMatcher(None, row[0], row[1]).ratio() * SequenceMatcher(None, row[2], row[3]).ratio(), axis=1)

    # Append similarity index
    teams_names_similarity = pd.concat([teams_names, similarity], axis=1)

    # Filter correct matches
    indices = teams_names_similarity.groupby(['HomeTeam_x'])[0].idxmax().values
    teams_names_matching = teams_names.take(indices)

    # Generate mapping
    matching1 = teams_names_matching.iloc[:, 0:2].drop_duplicates()
    matching2 = teams_names_matching.iloc[:, 2:].drop_duplicates()
    matching1.columns, matching2.columns = ['x', 'y'], ['x', 'y']
    matching = matching1.append(matching2).drop_duplicates()
    mapping = dict(zip(matching.x, matching.y))

    return mapping
