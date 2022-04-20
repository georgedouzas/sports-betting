"""
Download and transform historical and fixtures data
for various leagues from Football-Data.co.uk.

Football-Data.co.uk: https://www.football-data.co.uk/data.php
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from urllib.request import urlopen, urljoin
from datetime import datetime
from os.path import join
from functools import lru_cache

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import ParameterGrid

from ._utils import OUTPUTS, _read_csv
from .._base import _BaseDataLoader

URL = 'https://www.football-data.co.uk'
BASE_URLS = [
    'englandm.php',
    'scotlandm.php',
    'germanym.php',
    'italym.php',
    'spainm.php',
    'francem.php',
    'netherlandsm.php',
    'belgiumm.php',
    'portugalm.php',
    'turkeym.php',
    'greecem.php',
    'Argentina.php',
    'Austria.php',
    'Brazil.php',
    'China.php',
    'Denmark.php',
    'Finland.php',
    'Ireland.php',
    'Japan.php',
    'Mexico.php',
    'Norway.php',
    'Poland.php',
    'Romania.php',
    'Russia.php',
    'Sweden.php',
    'Switzerland.php',
    'USA.php',
]
LEAGUES_MAPPING = {
    'England': ('E', '0', '1', '2', '3', 'C'),
    'Scotland': ('SC', '0', '1', '2', '3', 'C'),
    'Germany': ('D', '1', '2'),
    'Italy': ('I', '1', '2'),
    'Spain': ('SP', '1', '2'),
    'France': ('F', '1', '2'),
    'Netherlands': ('N', '1'),
    'Belgium': ('B', '1'),
    'Portugal': ('P', '1'),
    'Turkey': ('T', '1'),
    'Greece': ('G', '1'),
    'Argentina': ('ARG', '1'),
    'Austria': ('AUT', '1'),
    'Brazil': ('BRA', '1'),
    'China': ('CHN', '1'),
    'Denmark': ('DNK', '1'),
    'Finland': ('FIN', '1'),
    'Ireland': ('IRL', '1'),
    'Japan': ('JPN', '1'),
    'Mexico': ('MEX', '1'),
    'Norway': ('NOR', '1'),
    'Poland': ('POL', '1'),
    'Romania': ('ROU', '1'),
    'Russia': ('RUS', '1'),
    'Sweden': ('SWE', '1'),
    'Switzerland': ('SWZ', '1'),
    'USA': ('USA', '1'),
}
REMOVED_COLS = [
    'Div',
    'Country',
    'Season',
    'Time',
    'FTR',
    'Res',
    'Attendance',
    'Referee',
    'HTR',
    'BbAH',
    'Bb1X2',
    'BbOU',
    'League',
    'divisions',
]
COLS_MAPPING = {
    'HT': 'home_team',
    'Home': 'home_team',
    'AT': 'away_team',
    'Away': 'away_team',
    'LB': 'odds__ladbrokes__home_win__full_time_goals',
    'LB.1': 'odds__ladbrokes__draw__full_time_goals',
    'LB.2': 'odds__ladbrokes__away_win__full_time_goals',
    'PH': 'odds__pinnacle__home_win__full_time_goals',
    'PD': 'odds__pinnacle__draw__full_time_goals',
    'PA': 'odds__pinnacle__away_win__full_time_goals',
    'HomeTeam': 'home_team',
    'AwayTeam': 'away_team',
    'Date': 'date',
    'B365AH': 'odds__bet365__size_of_asian_handicap_home_team__full_time_goals',
    'LBAH': 'odds__ladbrokes__size_of_asian_handicap_home_team__full_time_goals',
    'BbAHh': 'odds__betbrain__size_of_asian_handicap_home_team__full_time_goals',
    'GBAH': 'odds__gamebookers__size_of_handicap_home_team__full_time_goals',
    'AHh': 'odds__market_average__size_of_handicap_home_team__full_time_goals',
    'AHCh': 'odds__market_average_closing__size_of_asian_handicap_home_team__full_time_goals',
    'B365H': 'odds__bet365__home_win__full_time_goals',
    'B365D': 'odds__bet365__draw__full_time_goals',
    'B365A': 'odds__bet365__away_win__full_time_goals',
    'B365>2.5': 'odds__bet365__over_2.5__full_time_goals',
    'B365<2.5': 'odds__bet365__under_2.5__full_time_goals',
    'B365AHH': 'odds__bet365__asian_handicap_home_team__full_time_goals',
    'B365AHA': 'odds__bet365__asian_handicap_away_team__full_time_goals',
    'B365CH': 'odds__bet365_closing__home_win__full_time_goals',
    'B365CD': 'odds__bet365_closing__draw__full_time_goals',
    'B365CA': 'odds__bet365_closing__away_win__full_time_goals',
    'B365C>2.5': 'odds__bet365_closing__over_2.5__full_time_goals',
    'B365C<2.5': 'odds__bet365_closing__under_2.5__full_time_goals',
    'B365CAHH': 'odds__bet365_closing__asian_handicap_home_team__full_time_goals',
    'B365CAHA': 'odds__bet365_closing__asian_handicap_away_team__full_time_goals',
    'BbMxH': 'odds__betbrain_maximum__home_win__full_time_goals',
    'BbMxD': 'odds__betbrain_maximum__draw__full_time_goals',
    'BbMxA': 'odds__betbrain_maximum__away_win__full_time_goals',
    'BbMx>2.5': 'odds__betbrain_maximum__over_2.5__full_time_goals',
    'BbMx<2.5': 'odds__betbrain_maximum__under_2.5__full_time_goals',
    'BbMxAHH': 'odds__betbrain_maximum__asian_handicap_home_team__full_time_goals',
    'BbMxAHA': 'odds__betbrain_maximum__asian_handicap_away_team__full_time_goals',
    'BbAvH': 'odds__betbrain_average__home_win__full_time_goals',
    'BbAvD': 'odds__betbrain_average__draw_win__full_time_goals',
    'BbAvA': 'odds__betbrain_average__away_win__full_time_goals',
    'BbAv>2.5': 'odds__betbrain_average__over_2.5__full_time_goals',
    'BbAv<2.5': 'odds__betbrain_average__under_2.5__full_time_goals',
    'BbAvAHH': 'odds__betbrain_average__asian_handicap_home_team__full_time_goals',
    'BbAvAHA': 'odds__betbrain_average__asian_handicap_away_team__full_time_goals',
    'BWH': 'odds__betwin__home_win__full_time_goals',
    'BWD': 'odds__betwin__draw__full_time_goals',
    'BWA': 'odds__betwin__away_win__full_time_goals',
    'BWCH': 'odds__betwin_closing__home_win__full_time_goals',
    'BWCD': 'odds__betwin_closing__draw__full_time_goals',
    'BWCA': 'odds__betwin_closing__away_win__full_time_goals',
    'BSH': 'odds__bluesquare__home_win__full_time_goals',
    'BSD': 'odds__bluesquare__draw__full_time_goals',
    'BSA': 'odds__bluesquare__away_win__full_time_goals',
    'GBH': 'odds__gamebookers__home_win__full_time_goals',
    'GBD': 'odds__gamebookers__draw__full_time_goals',
    'GBA': 'odds__gamebookers__away_win__full_time_goals',
    'GB>2.5': 'odds__gamebookers__over_2.5__full_time_goals',
    'GB<2.5': 'odds__gamebookers__under_2.5__full_time_goals',
    'GBAHH': 'odds__gamebookers__asian_handicap_home_team__full_time_goals',
    'GBAHA': 'odds__gamebookers__asian_handicap_away_team__full_time_goals',
    'IWH': 'odds__interwetten__home_win__full_time_goals',
    'IWD': 'odds__interwetten__draw__full_time_goals',
    'IWA': 'odds__interwetten__away_win__full_time_goals',
    'IWCH': 'odds__interwetten_closing__home_win__full_time_goals',
    'IWCD': 'odds__interwetten_closing__draw__full_time_goals',
    'IWCA': 'odds__interwetten_closing__away_win__full_time_goals',
    'LBH': 'odds__ladbrokes__home_win__full_time_goals',
    'LBD': 'odds__ladbrokes__draw__full_time_goals',
    'LBA': 'odds__ladbrokes__away_win__full_time_goals',
    'LBAHH': 'odds__ladbrokes__asian_handicap_home_team__full_time_goals',
    'LBAHA': 'odds__ladbrokes__asian_handicap_away_team__full_time_goals',
    'PSH': 'odds__pinnacle__home_win__full_time_goals',
    'PSD': 'odds__pinnacle__draw__full_time_goals',
    'PSA': 'odds__pinnacle__away_win__full_time_goals',
    'P>2.5': 'odds__pinnacle__over_2.5__full_time_goals',
    'P<2.5': 'odds__pinnacle__under_2.5__full_time_goals',
    'PAHH': 'odds__pinnacle__asian_handicap_home_team__full_time_goals',
    'PAHA': 'odds__pinnacle__asian_handicap_away_team__full_time_goals',
    'PSCH': 'odds__pinnacle_closing__home_win__full_time_goals',
    'PSCD': 'odds__pinnacle_closing__draw__full_time_goals',
    'PSCA': 'odds__pinnacle_closing__away_win__full_time_goals',
    'PC>2.5': 'odds__pinnacle_closing__over_2.5__full_time_goals',
    'PC<2.5': 'odds__pinnacle_closing__under_2.5__full_time_goals',
    'PCAHH': 'odds__pinnacle_closing__asian_handicap_home_team__full_time_goals',
    'PCAHA': 'odds__pinnacle_closing__asian_handicap_away_team__full_time_goals',
    'SOH': 'odds__sporting__home_win__full_time_goals',
    'SOD': 'odds__sporting__draw__full_time_goals',
    'SOA': 'odds__sporting__away_win__full_time_goals',
    'SBH': 'odds__sportingbet__home_win__full_time_goals',
    'SBD': 'odds__sportingbet__draw__full_time_goals',
    'SBA': 'odds__sportingbet__away_win__full_time_goals',
    'SJH': 'odds__stanjames__home_win__full_time_goals',
    'SJD': 'odds__stanjames__draw__full_time_goals',
    'SJA': 'odds__stanjames__away_win__full_time_goals',
    'SYH': 'odds__stanleybet__home_win__full_time_goals',
    'SYD': 'odds__stanleybet__draw__full_time_goals',
    'SYA': 'odds__stanleybet__away_win__full_time_goals',
    'VCH': 'odds__vcbet__home_win__full_time_goals',
    'VCD': 'odds__vcbet__draw__full_time_goals',
    'VCA': 'odds__vcbet__away_win__full_time_goals',
    'VCCH': 'odds__vcbet_closing__home_win__full_time_goals',
    'VCCD': 'odds__vcbet_closing__draw__full_time_goals',
    'VCCA': 'odds__vcbet_closing__away_win__full_time_goals',
    'WHH': 'odds__williamhill__home_win__full_time_goals',
    'WHD': 'odds__williamhill__draw__full_time_goals',
    'WHA': 'odds__williamhill__away_win__full_time_goals',
    'WHCH': 'odds__williamhill_closing__home_win__full_time_goals',
    'WHCD': 'odds__williamhill_closing__draw__full_time_goals',
    'WHCA': 'odds__williamhill_closing__away_win__full_time_goals',
    'MaxH': 'odds__market_maximum__home_win__full_time_goals',
    'MaxD': 'odds__market_maximum__draw__full_time_goals',
    'MaxA': 'odds__market_maximum__away_win__full_time_goals',
    'Max>2.5': 'odds__market_maximum__over_2.5__full_time_goals',
    'Max<2.5': 'odds__market_maximum__under_2.5__full_time_goals',
    'MaxAHH': 'odds__market_maximum__asian_handicap_home_team__full_time_goals',
    'MaxAHA': 'odds__market_maximum__asian_handicap_away_team__full_time_goals',
    'MaxCH': 'odds__market_maximum_closing__home_win__full_time_goals',
    'MaxCD': 'odds__market_maximum_closing__draw__full_time_goals',
    'MaxCA': 'odds__market_maximum_closing__away_win__full_time_goals',
    'MaxC>2.5': 'odds__market_maximum_closing__over_2.5__full_time_goals',
    'MaxC<2.5': 'odds__market_maximum_closing__under_2.5__full_time_goals',
    'MaxCAHH': 'odds__market_maximum_closing__asian_handicap_home_team__full_time_goals',
    'MaxCAHA': 'odds__market_maximum_closing__asian_handicap_away_team__full_time_goals',
    'AvgH': 'odds__market_average__home_win__full_time_goals',
    'AvgD': 'odds__market_average__draw__full_time_goals',
    'AvgA': 'odds__market_average__away_win__full_time_goals',
    'Avg>2.5': 'odds__market_average__over_2.5__full_time_goals',
    'Avg<2.5': 'odds__market_average__under_2.5__full_time_goals',
    'AvgAHH': 'odds__market_average__asian_handicap_home_team__full_time_goals',
    'AvgAHA': 'odds__market_average__asian_handicap_away_team__full_time_goals',
    'AvgCH': 'odds__market_average_closing__home_win__full_time_goals',
    'AvgCD': 'odds__market_average_closing__draw__full_time_goals',
    'AvgCA': 'odds__market_average_closing__away_win__full_time_goals',
    'AvgC>2.5': 'odds__market_average_closing__over_2.5__full_time_goals',
    'AvgC<2.5': 'odds__market_average_closing__under_2.5__full_time_goals',
    'AvgCAHH': 'odds__market_average_closing__asian_handicap_home_team__full_time_goals',
    'AvgCAHA': 'odds__market_average_closing__asian_handicap_away_team__full_time_goals',
    'HG': 'target__home_team__full_time_goals',
    'AG': 'target__away_team__full_time_goals',
    'FTHG': 'target__home_team__full_time_goals',
    'FTAG': 'target__away_team__full_time_goals',
    'HTHG': 'target__home_team__half_time_goals',
    'HTAG': 'target__away_team__half_time_goals',
    'HS': 'target__home_team__shots',
    'AS': 'target__away_team__shots',
    'HST': 'target__home_team__shots_on_target',
    'AST': 'target__away_team__shots_on_target',
    'HHW': 'target__home_team__hit_woodork',
    'AHW': 'target__away_team__hit_woodork',
    'HC': 'target__home_team__corners',
    'AC': 'target__away_team__corners',
    'HF': 'target__home_team__fouls_committed',
    'AF': 'target__away_team__fouls_committed',
    'HFKC': 'target__home_team__free_kicks_conceded',
    'AFKC': 'target__away_team__free_kicks_conceded',
    'HO': 'target__home_team__offsides',
    'AO': 'target__away_team__offsides',
    'HY': 'target__home_team__yellow_cards',
    'AY': 'target__away_team__yellow_cards',
    'HR': 'target__home_team__red_cards',
    'AR': 'target__away_team__red_cards',
    'HBP': 'target__home_team__bookings_points',
    'ABP': 'target__away_team__bookings_points',
}


def _convert_base_url_to_league(base_url):
    league = base_url.replace('.php', '')
    if base_url[0].islower():
        league = league[:-1].capitalize()
    return league


def _extract_csv_urls(base_url):
    html = urlopen(urljoin(URL, base_url))
    bsObj = BeautifulSoup(html.read(), features='html.parser')
    return {
        el.get('href') for el in bsObj.find_all('a') if el.get('href').endswith('csv')
    }


def _param_grid_to_csv_urls(param_grid):
    urls = []
    for params in param_grid:
        in_main_leagues = f'{params["league"].lower()}m.php' in BASE_URLS
        encoded_league, *divisions = LEAGUES_MAPPING[params['league']]
        if in_main_leagues:
            year = f'{str(params["year"] - 1)[2:]}{str(params["year"])[2:]}'
            if '0' in divisions:
                division = (
                    str(params['division'] - 1) if params['division'] != 5 else 'C'
                )
            else:
                division = str(params['division'])
            urls.append(
                (params, join(URL, 'mmz4281', year, f'{encoded_league}{division}.csv'))
            )
        else:
            urls.append((params, join(URL, 'new', f'{encoded_league}.csv')))
    return urls


@lru_cache
def _get_params():
    full_param_grid = []
    for base_url in BASE_URLS:
        league = _convert_base_url_to_league(base_url)
        divisions = LEAGUES_MAPPING[league][1:]
        urls = _extract_csv_urls(base_url)
        for url in urls:
            if base_url[0].islower():
                _, year, division = url.split('/')
                year = datetime.strptime(year[2:], '%y').year
                division = division.replace('.csv', '')[-1]
                param_grid = {
                    'league': [league],
                    'division': [
                        int(division) + int('0' in divisions) if division != 'C' else 5
                    ],
                    'year': [year],
                }
            else:
                years = _read_csv(urljoin(URL, url))['Season']
                years = list(
                    {
                        season + 1
                        if type(season) is not str
                        else int(season.split('/')[-1])
                        for season in years.unique()
                    }
                )
                param_grid = {'league': [league], 'division': [1], 'year': years}
            full_param_grid.append(param_grid)
    return ParameterGrid(full_param_grid)


class _FDSoccerDataLoader(_BaseDataLoader):
    """Dataloader for Football-Data.co.uk soccer data.

    It downloads historical and fixtures data from
    `Football-Data.co.uk <https://www.football-data.co.uk/data.php>`_.
    """

    SCHEMA = [
        ('league', object),
        ('division', int),
        ('year', int),
        ('home_team', object),
        ('away_team', object),
        ('date', np.datetime64),
        ('odds__bet365__home_win__full_time_goals', float),
        ('odds__bet365__draw__full_time_goals', float),
        ('odds__bet365__away_win__full_time_goals', float),
        ('odds__bet365__over_2.5__full_time_goals', float),
        ('odds__bet365__under_2.5__full_time_goals', float),
        ('odds__bet365__asian_handicap_home_team__full_time_goals', float),
        ('odds__bet365__asian_handicap_away_team__full_time_goals', float),
        ('odds__bet365_closing__home_win__full_time_goals', float),
        ('odds__bet365_closing__draw__full_time_goals', float),
        ('odds__bet365_closing__away_win__full_time_goals', float),
        ('odds__bet365_closing__over_2.5__full_time_goals', float),
        ('odds__bet365_closing__under_2.5__full_time_goals', float),
        ('odds__bet365_closing__asian_handicap_home_team__full_time_goals', float),
        ('odds__bet365_closing__asian_handicap_away_team__full_time_goals', float),
        ('odds__bet365__size_of_asian_handicap_home_team__full_time_goals', object),
        ('odds__betbrain_maximum__home_win__full_time_goals', float),
        ('odds__betbrain_maximum__draw__full_time_goals', float),
        ('odds__betbrain_maximum__away_win__full_time_goals', float),
        ('odds__betbrain_maximum__over_2.5__full_time_goals', float),
        ('odds__betbrain_maximum__under_2.5__full_time_goals', float),
        ('odds__betbrain_maximum__asian_handicap_home_team__full_time_goals', float),
        ('odds__betbrain_maximum__asian_handicap_away_team__full_time_goals', float),
        ('odds__betbrain_average__home_win__full_time_goals', float),
        ('odds__betbrain_average__draw_win__full_time_goals', float),
        ('odds__betbrain_average__away_win__full_time_goals', float),
        ('odds__betbrain_average__over_2.5__full_time_goals', float),
        ('odds__betbrain_average__under_2.5__full_time_goals', float),
        ('odds__betbrain_average__asian_handicap_home_team__full_time_goals', float),
        ('odds__betbrain_average__asian_handicap_away_team__full_time_goals', float),
        ('odds__betbrain__size_of_asian_handicap_home_team__full_time_goals', object),
        ('odds__betwin__home_win__full_time_goals', float),
        ('odds__betwin__draw__full_time_goals', float),
        ('odds__betwin__away_win__full_time_goals', float),
        ('odds__betwin_closing__home_win__full_time_goals', float),
        ('odds__betwin_closing__draw__full_time_goals', float),
        ('odds__betwin_closing__away_win__full_time_goals', float),
        ('odds__bluesquare__home_win__full_time_goals', float),
        ('odds__bluesquare__draw__full_time_goals', float),
        ('odds__bluesquare__away_win__full_time_goals', float),
        ('odds__gamebookers__home_win__full_time_goals', float),
        ('odds__gamebookers__draw__full_time_goals', float),
        ('odds__gamebookers__away_win__full_time_goals', float),
        ('odds__gamebookers__over_2.5__full_time_goals', float),
        ('odds__gamebookers__under_2.5__full_time_goals', float),
        ('odds__gamebookers__asian_handicap_home_team__full_time_goals', float),
        ('odds__gamebookers__asian_handicap_away_team__full_time_goals', float),
        ('odds__gamebookers__size_of_handicap_home_team__full_time_goals', object),
        ('odds__interwetten__home_win__full_time_goals', float),
        ('odds__interwetten__draw__full_time_goals', float),
        ('odds__interwetten__away_win__full_time_goals', float),
        ('odds__interwetten_closing__home_win__full_time_goals', float),
        ('odds__interwetten_closing__draw__full_time_goals', float),
        ('odds__interwetten_closing__away_win__full_time_goals', float),
        ('odds__ladbrokes__home_win__full_time_goals', float),
        ('odds__ladbrokes__draw__full_time_goals', float),
        ('odds__ladbrokes__away_win__full_time_goals', float),
        ('odds__ladbrokes__asian_handicap_home_team__full_time_goals', float),
        ('odds__ladbrokes__asian_handicap_away_team__full_time_goals', float),
        ('odds__ladbrokes__size_of_asian_handicap_home_team__full_time_goals', object),
        ('odds__pinnacle__home_win__full_time_goals', float),
        ('odds__pinnacle__draw__full_time_goals', float),
        ('odds__pinnacle__away_win__full_time_goals', float),
        ('odds__pinnacle__over_2.5__full_time_goals', float),
        ('odds__pinnacle__under_2.5__full_time_goals', float),
        ('odds__pinnacle__asian_handicap_home_team__full_time_goals', float),
        ('odds__pinnacle__asian_handicap_away_team__full_time_goals', float),
        ('odds__pinnacle_closing__home_win__full_time_goals', float),
        ('odds__pinnacle_closing__draw__full_time_goals', float),
        ('odds__pinnacle_closing__away_win__full_time_goals', float),
        ('odds__pinnacle_closing__over_2.5__full_time_goals', float),
        ('odds__pinnacle_closing__under_2.5__full_time_goals', float),
        ('odds__pinnacle_closing__asian_handicap_home_team__full_time_goals', float),
        ('odds__pinnacle_closing__asian_handicap_away_team__full_time_goals', float),
        ('odds__sporting__home_win__full_time_goals', float),
        ('odds__sporting__draw__full_time_goals', float),
        ('odds__sporting__away_win__full_time_goals', float),
        ('odds__sportingbet__home_win__full_time_goals', float),
        ('odds__sportingbet__draw__full_time_goals', float),
        ('odds__sportingbet__away_win__full_time_goals', float),
        ('odds__stanjames__home_win__full_time_goals', float),
        ('odds__stanjames__draw__full_time_goals', float),
        ('odds__stanjames__away_win__full_time_goals', float),
        ('odds__stanleybet__home_win__full_time_goals', float),
        ('odds__stanleybet__draw__full_time_goals', float),
        ('odds__stanleybet__away_win__full_time_goals', float),
        ('odds__vcbet__home_win__full_time_goals', float),
        ('odds__vcbet__draw__full_time_goals', float),
        ('odds__vcbet__away_win__full_time_goals', float),
        ('odds__vcbet_closing__home_win__full_time_goals', float),
        ('odds__vcbet_closing__draw__full_time_goals', float),
        ('odds__vcbet_closing__away_win__full_time_goals', float),
        ('odds__williamhill__home_win__full_time_goals', float),
        ('odds__williamhill__draw__full_time_goals', float),
        ('odds__williamhill__away_win__full_time_goals', float),
        ('odds__williamhill_closing__home_win__full_time_goals', float),
        ('odds__williamhill_closing__draw__full_time_goals', float),
        ('odds__williamhill_closing__away_win__full_time_goals', float),
        ('odds__market_maximum__home_win__full_time_goals', float),
        ('odds__market_maximum__draw__full_time_goals', float),
        ('odds__market_maximum__away_win__full_time_goals', float),
        ('odds__market_maximum__over_2.5__full_time_goals', float),
        ('odds__market_maximum__under_2.5__full_time_goals', float),
        ('odds__market_maximum__asian_handicap_home_team__full_time_goals', float),
        ('odds__market_maximum__asian_handicap_away_team__full_time_goals', float),
        ('odds__market_maximum_closing__home_win__full_time_goals', float),
        ('odds__market_maximum_closing__draw__full_time_goals', float),
        ('odds__market_maximum_closing__away_win__full_time_goals', float),
        ('odds__market_maximum_closing__over_2.5__full_time_goals', float),
        ('odds__market_maximum_closing__under_2.5__full_time_goals', float),
        (
            'odds__market_maximum_closing__asian_handicap_home_team__full_time_goals',
            float,
        ),
        (
            'odds__market_maximum_closing__asian_handicap_away_team__full_time_goals',
            float,
        ),
        ('odds__market_average__home_win__full_time_goals', float),
        ('odds__market_average__draw__full_time_goals', float),
        ('odds__market_average__away_win__full_time_goals', float),
        ('odds__market_average__over_2.5__full_time_goals', float),
        ('odds__market_average__under_2.5__full_time_goals', float),
        ('odds__market_average__asian_handicap_home_team__full_time_goals', float),
        ('odds__market_average__asian_handicap_away_team__full_time_goals', float),
        ('odds__market_average_closing__home_win__full_time_goals', float),
        ('odds__market_average_closing__draw__full_time_goals', float),
        ('odds__market_average_closing__away_win__full_time_goals', float),
        ('odds__market_average_closing__over_2.5__full_time_goals', float),
        ('odds__market_average_closing__under_2.5__full_time_goals', float),
        (
            'odds__market_average_closing__asian_handicap_home_team__full_time_goals',
            float,
        ),
        (
            'odds__market_average_closing__asian_handicap_away_team__full_time_goals',
            float,
        ),
        ('odds__market_average__size_of_handicap_home_team__full_time_goals', object),
        (
            'odds__market_average_closing__size_of_asian_handicap_home_team__full_time_goals',
            object,
        ),
        ('target__home_team__full_time_goals', int),
        ('target__away_team__full_time_goals', int),
        ('target__home_team__half_time_goals', int),
        ('target__away_team__half_time_goals', int),
        ('target__home_team__shots', int),
        ('target__away_team__shots', int),
        ('target__home_team__shots_on_target', int),
        ('target__away_team__shots_on_target', int),
        ('target__home_team__hit_woodork', int),
        ('target__away_team__hit_woodork', int),
        ('target__home_team__corners', int),
        ('target__away_team__corners', int),
        ('target__home_team__fouls_committed', int),
        ('target__away_team__fouls_committed', int),
        ('target__home_team__free_kicks_conceded', int),
        ('target__away_team__free_kicks_conceded', int),
        ('target__home_team__offsides', int),
        ('target__away_team__offsides', int),
        ('target__home_team__yellow_cards', int),
        ('target__away_team__yellow_cards', int),
        ('target__home_team__red_cards', int),
        ('target__away_team__red_cards', int),
        ('target__home_team__bookings_points', float),
        ('target__away_team__bookings_points', float),
    ]
    OUTPUTS = OUTPUTS

    @classmethod
    @property
    def PARAMS(cls):
        return _get_params()

    @lru_cache
    def _get_data(self):

        # Training data
        data_container = []
        urls = _param_grid_to_csv_urls(self.param_grid_)
        for params, url in urls:

            data = _read_csv(url).replace('#REF!', np.nan)
            try:
                data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
            except ValueError:
                data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)

            if url.split('/')[-2] != 'new':
                data = data.assign(
                    league=params['league'],
                    division=params['division'],
                    year=params['year'],
                    fixtures=False,
                )
            else:
                data = data.assign(
                    league=params['league'], division=params['division'], fixtures=False
                )
                data['year'] = data['Season'].apply(
                    lambda season: season + 1
                    if type(season) is not str
                    else int(season.split('/')[-1])
                )
                data = data[data.year == params['year']]

            data = data.drop(
                columns=[
                    col
                    for col in data.columns
                    if 'Unnamed' in col or col in REMOVED_COLS
                ],
            ).rename(columns=COLS_MAPPING)
            data_container.append(data)

        # Fixtures data
        data = _read_csv(join(URL, 'fixtures.csv'))
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
        data = data.dropna(axis=0, how='any', subset=['Div', 'HomeTeam', 'AwayTeam'])
        data['fixtures'] = True
        inv_leagues_mapping = {v[0]: k for k, v in LEAGUES_MAPPING.items()}
        data['league'] = data['Div'].apply(lambda div: inv_leagues_mapping[div[:-1]])
        data['division'] = data['Div'].apply(lambda div: div[-1])
        data['divisions'] = data['league'].apply(
            lambda league: LEAGUES_MAPPING[league][1:]
        )
        data['division'] = (
            data[['division', 'divisions']]
            .apply(
                lambda row: row[0]
                if 'C' not in row[1]
                else (row[0] - 1 if isinstance(row[0], int) else 4),
                axis=1,
            )
            .astype(int)
        )
        years = (
            pd.DataFrame(self.PARAMS).groupby(['league', 'division']).max()
        ).reset_index()
        data = pd.merge(data, years, how='left')
        data = data.drop(
            columns=[
                col for col in data.columns if 'Unnamed' in col or col in REMOVED_COLS
            ]
        ).rename(columns=COLS_MAPPING)
        data_container.append(data)

        # Combine data
        data = pd.concat(data_container, ignore_index=True)

        return data.sort_values(['league', 'division', 'year'], ignore_index=True)
