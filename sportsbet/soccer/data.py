"""
Download and prepare soccer historical data from various leagues.
"""

from os.path import join
from urllib.request import urljoin
from itertools import product
from difflib import SequenceMatcher

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

LEAGUES_MAPPING = [
    ('Argentina Primera Division', ('ARG', 'extra'), ('argentina', 'superliga')),
    ('Austrian T-Mobile Bundesliga', ('AUT', 'extra'), ('austria', 'tipico-bundesliga')),
    ('Barclays Premier League', ('E0', 'main'), ('england', 'premier-league')),
    ('Belgian Jupiler League', ('B1', 'main'), ('belgium', 'jupiler-pro-league')),
    ('Brasileiro Série A', ('BRA', 'extra'), ('brazil', 'serie-a')),
    ('Chinese Super League', ('CHN', 'extra'), ('china' , 'super-league')),
    ('Danish SAS-Ligaen', ('DNK', 'extra'), ('denmark', 'superliga')),
    ('Dutch Eredivisie', ('N1', 'main'), ('netherlands', 'eerste-divisie')),
    ('English League Championship', ('E1', 'main'), ('england', 'championship')),
    ('English League One', ('E2', 'main'), ('england', 'league-1')),
    ('English League Two', ('E3', 'main'), ('england', 'league-2')),
    ('French Ligue 1', ('F1', 'main'), ('france', 'ligue-1')),
    ('French Ligue 2', ('F2', 'main'), ('france', 'ligue-2')),
    ('German Bundesliga', ('D1', 'main'), ('germany', 'bundesliga')),
    ('German 2. Bundesliga', ('D2', 'main'), ('germany', '2-bundesliga')),
    ('Greek Super League', ('G1', 'main'), ('greece', 'super-league')),
    ('Italy Serie A', ('I1', 'main'), ('italy', 'serie-a')),
    ('Italy Serie B', ('I2', 'main'), ('italy', 'serie-b')),
    ('Japanese J League', ('JPN', 'extra'), ('japan', 'j-league')),
    ('Major League Soccer', ('USA', 'extra'), ('usa', 'mls')),
    ('Norwegian Tippeligaen', ('NOR', 'extra'), ('norway', 'eliteserien')),
    ('Portuguese Liga', ('P1', 'main'), ('portugal', 'primeira-liga')),
    ('Russian Premier Liga', ('RUS', 'extra'), ('russia', 'premier-league')),
    ('Scottish Premiership', ('SC0', 'main'), ('scotland', 'premiership')),
    ('Spanish Primera Division', ('SP1', 'main'), ('spain', 'primera-division')),
    ('Spanish Segunda Division', ('SP2', 'main'), ('spain', 'segunda-division')),
    ('Swedish Allsvenskan', ('SWE', 'extra'), ('sweden', 'allsvenskan')),
    ('Swiss Raiffeisen Super League', ('SWZ', 'extra'), ('switzerland', 'super-league')),
    ('Turkish Turkcell Super Lig', ('T1', 'main'), ('turkey', 'super-lig'))
]


def _fetch_spi_data(leagues):
    """Fetch the data containing match-by-match SPI ratings."""

    # Define url
    url = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
    
    # Define mappings
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
    leagues_mapping = dict([(league, league_tpl) for league, league_tpl, *_ in LEAGUES_MAPPING])

    # Download data
    data = pd.read_csv(url)
    
    # Select and rename columns
    data = data.loc[:, columns_mapping.keys()]
    data.rename(columns=columns_mapping, inplace=True)

    # Filter and rename leagues
    data = data[data.League.isin(leagues_mapping.keys())]
    data['League'] = data['League'].apply(lambda league: leagues_mapping[league][0])

    # Extract leagues
    if leagues == 'all':
        leagues = [league_id for league_id, _ in leagues_mapping.values()]
    elif leagues == 'main':
        leagues = [league_id for league_id, league_type in leagues_mapping.values() if league_type == 'main']
    else:
        leagues = [league_id for league_id, _ in leagues_mapping.values() if league_id in leagues]
    
    # Filter matches
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

    # Define mapping
    leagues_mapping = dict([(league, league_tpl) for league, league_tpl, *_ in LEAGUES_MAPPING])

    # Extract leagues type
    if leagues == 'all':
        leagues_types = leagues_mapping.values()
    elif leagues == 'main':
        leagues_types = [(league_id, league_type) for league_id, league_type in leagues_mapping.values() if league_type == 'main']
    else:
        leagues_types = [(league_id, league_type) for league_id, league_type in leagues_mapping.values() if league_id in leagues]

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


def _scrape_bb_data(leagues):
    """Scrape upcoming matches data from BetBrain."""

    # Extract leagues
    leagues = [(league_tpl[0], join('football', *league_op)) for _, league_tpl, league_op, *_ in LEAGUES_MAPPING 
                if league_tpl[1] == 'main' and (league_tpl[0] in leagues if leagues not in ('main', 'all') else True)]

    # Define base url
    base_url = 'https://www.betbrain.com/'

    # Create driver
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    # Define data placeholder
    data = []

    for league_id, suffix_url in leagues:

        # Parse league data
        driver.get(urljoin(base_url, suffix_url))
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'MatchTitleLink'))
            )
        except TimeoutException:
            print('League %s was not scrapped.' % league_id)
            continue
        finally:
            parsed_data = BeautifulSoup(driver.page_source, 'html.parser').findAll('a', {'class': 'MatchTitleLink'})
        
        
        # Parse matches data
        matches_urls = [(match.attrs['title'].split(' - '), match.attrs['href']) for match in parsed_data if 'home-draw-away' in match.attrs['href']]
        matches_urls = [((home_team, away_team[:-5]), suffix_url) for (home_team, away_team), suffix_url in matches_urls]

        print('League %s. Scrapping %s odds.' % (league_id, len(matches_urls)))

        for (home_team, away_team), suffix_url in matches_urls:

            # Parse match data
            driver.get(urljoin(base_url, suffix_url))
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'IsAverage'))
                )
            except TimeoutException:
                print('Odds for match %s - %s from %s league were not scrapped.' % (home_team, away_team, league_id))  
                continue
            finally:
                parsed_data = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Define odds placeholder
            odds = []

            # Populate average and maximum odds
            for class_value in ['IsAverage', 'HighestOdds']:
                elements = parsed_data.findAll('li', {'class': class_value})[:3]
                parsed_odds = [float(element.text) for element in elements] 
                odds.append([parsed_odds[0], parsed_odds[2], parsed_odds[1]])

            # Append data
            data.append((league_id, home_team, away_team, odds[0], odds[1]))

    data = pd.DataFrame(data, columns=['League', 'HomeTeam', 'AwayTeam', 'AverageOdd', 'MaximumOdd'])
    
    return data


def _scrape_op_data(leagues):
    """Scrape upcoming matches data from Odds Portal."""

    if leagues == 'main':
        return []

    # Extract leagues
    leagues = [(league_tpl[0], join('soccer', *league_op)) for _, league_tpl, league_op, *_ in LEAGUES_MAPPING 
                if league_tpl[1] == 'extra' and (league_tpl[0] in leagues if leagues != 'all' else True)]

    # Define base url
    base_url = 'https://www.oddsportal.com'

    # Create driver
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    # Define data placeholder
    data = []
    
    for league_id, suffix_url in leagues:

        # Parse league data
        driver.get(urljoin(base_url, suffix_url))
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'table-participant'))
            )
        except TimeoutException:
            print('League %s was not scrapped.' % league_id)
            continue
        finally:
            parsed_data = BeautifulSoup(driver.page_source, 'html.parser').findAll('td', {'class': 'table-participant'})
        
        # Parse matches data
        matches = [list(match.children)[-1] for match in parsed_data]
        matches_urls = [(match.text.split(' - '), match.attrs['href']) for match in matches if match.name == 'a']

        print('League %s. Scrapping %s odds.' % (league_id, len(matches_urls)))

        for (home_team, away_team), suffix_url in matches_urls:

            # Parse match data
            driver.get(urljoin(base_url, suffix_url))
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'aver'))
                )
            except TimeoutException:
                print('Odds for match %s - %s from %s league were not scrapped.' % (home_team, away_team, league_id))  
                continue
            finally:
                parsed_data = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Define odds placeholder
            odds = []

            # Populate average and maximum odds
            for class_value in ['aver', 'highest']:
                elements = parsed_data.findAll('tr', {'class': class_value})[0]
                parsed_odds = [float(element.text) for element in elements.findAll(attrs={'class': 'right'})]
                odds.append([parsed_odds[0], parsed_odds[2], parsed_odds[1]])

            # Append data
            data.append((league_id, home_team, away_team, odds[0], odds[1]))
    
    data = pd.DataFrame(data, columns=['League', 'HomeTeam', 'AwayTeam', 'AverageOdd', 'MaximumOdd'])

    return data


def _match_teams_names_historical(spi_data, fd_data):
    """Match teams names between spi and fd historical data."""

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


def _match_teams_names_future(spi_future_data, scraped_data):
    """Match teams names between spi and fd historical data."""

    # Define merge keys
    keys = ['League', 'HomeTeam', 'AwayTeam']

    # Merge data
    teams_names = pd.merge(scraped_data[keys], spi_future_data[keys], how='outer', on='League')

    # Calculate similarity index
    similarity = teams_names.apply(lambda row: SequenceMatcher(None, row[1], row[3]).ratio() * SequenceMatcher(None, row[2], row[4]).ratio(), axis=1)

    # Append similarity index
    teams_names_similarity = pd.concat([teams_names, similarity], axis=1)

    # Filter correct matches
    indices = teams_names_similarity.groupby(['HomeTeam_x', 'AwayTeam_x'])[0].idxmax().values
    teams_names_matching = teams_names.take(indices)

    # Generate mapping
    mapping = {name1: name2 for name1, name2 in teams_names_matching[['HomeTeam_y', 'HomeTeam_x']].values}
    mapping.update({name1: name2 for name1, name2 in teams_names_matching[['AwayTeam_y', 'AwayTeam_x']].values})

    return mapping