"""
Download and transform historical and fixtures data
for various leagues from Football-Data.co.uk.

Football-Data.co.uk: http://www.football-data.co.uk/data.php
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
from rich.progress import track
from sklearn.model_selection import ParameterGrid

from . import OUTCOMES
from .._base import _BaseDataLoader, _read_csv

URL = 'http://www.football-data.co.uk'
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


class FDSoccerDataLoader(_BaseDataLoader):
    """Dataloader for Football-Data.co.uk soccer data.

    Read more in the :ref:`user guide <user_guide>`.

    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such parameter, default=None
        The parameter grid to explore, as a dictionary mapping data parameters
        to sequences of allowed values. An empty dict signifies default
        parameters. A sequence of dicts signifies a sequence of grids to search,
        and is useful to avoid exploring parameter combinations that do not
        exist. The default value corresponds to all parameters.

    Examples
    --------
    >>>
    """

    _removed_cols = [
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
    _cols_mapping = {
        'HT': 'home_team',
        'Home': 'home_team',
        'AT': 'away_team',
        'Away': 'away_team',
        'LB': 'ladbrokes__home_win__odds',
        'LB.1': 'ladbrokes__draw__odds',
        'LB.2': 'ladbrokes__away_win__odds',
        'PH': 'pinnacle__home_win__odds',
        'PD': 'pinnacle__draw__odds',
        'PA': 'pinnacle__away_win__odds',
        'HG': 'home_team__full_time_goals',
        'AG': 'away_team__full_time_goals',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'Date': 'date',
        'B365AH': 'bet365__size_of_asian_handicap_home_team__odds',
        'LBAH': 'ladbrokes__size_of_asian_handicap_home_team__odds',
        'BbAHh': 'betbrain__size_of_asian_handicap_home_team__odds',
        'GBAH': 'gamebookers__size_of_handicap_home_team__odds',
        'AHh': 'market_average__size_of_handicap_home_team__odds',
        'AHCh': 'market_average_closing__size_of_asian_handicap_home_team__odds',
        'B365H': 'bet365__home_win__odds',
        'B365D': 'bet365__draw__odds',
        'B365A': 'bet365__away_win__odds',
        'B365>2.5': 'bet365__over_2.5__odds',
        'B365<2.5': 'bet365__under_2.5__odds',
        'B365AHH': 'bet365__asian_handicap_home_team__odds',
        'B365AHA': 'bet365__asian_handicap_away_team__odds',
        'B365CH': 'bet365_closing__home_win__odds',
        'B365CD': 'bet365_closing__draw__odds',
        'B365CA': 'bet365_closing__away_win__odds',
        'B365C>2.5': 'bet365_closing__over_2.5__odds',
        'B365C<2.5': 'bet365_closing__under_2.5__odds',
        'B365CAHH': 'bet365_closing__asian_handicap_home_team__odds',
        'B365CAHA': 'bet365_closing__asian_handicap_away_team__odds',
        'BbMxH': 'betbrain_maximum__home_win__odds',
        'BbMxD': 'betbrain_maximum__draw__odds',
        'BbMxA': 'betbrain_maximum__away_win__odds',
        'BbMx>2.5': 'betbrain_maximum__over_2.5__odds',
        'BbMx<2.5': 'betbrain_maximum__under_2.5__odds',
        'BbMxAHH': 'betbrain_maximum__asian_handicap_home_team__odds',
        'BbMxAHA': 'betbrain_maximum__asian_handicap_away_team__odds',
        'BbAvH': 'betbrain_average__home_win__odds',
        'BbAvD': 'betbrain_average__draw_win__odds',
        'BbAvA': 'betbrain_average__away_win__odds',
        'BbAv>2.5': 'betbrain_average__over_2.5__odds',
        'BbAv<2.5': 'betbrain_average__under_2.5__odds',
        'BbAvAHH': 'betbrain_average__asian_handicap_home_team__odds',
        'BbAvAHA': 'betbrain_average__asian_handicap_away_team__odds',
        'BWH': 'betwin__home_win__odds',
        'BWD': 'betwin__draw__odds',
        'BWA': 'betwin__away_win__odds',
        'BWCH': 'betwin_closing__home_win__odds',
        'BWCD': 'betwin_closing__draw__odds',
        'BWCA': 'betwin_closing__away_win__odds',
        'BSH': 'bluesquare__home_win__odds',
        'BSD': 'bluesquare__draw__odds',
        'BSA': 'bluesquare__away_win__odds',
        'GBH': 'gamebookers__home_win__odds',
        'GBD': 'gamebookers__draw__odds',
        'GBA': 'gamebookers__away_win__odds',
        'GB>2.5': 'gamebookers__over_2.5__odds',
        'GB<2.5': 'gamebookers__under_2.5__odds',
        'GBAHH': 'gamebookers__asian_handicap_home_team__odds',
        'GBAHA': 'gamebookers__asian_handicap_away_team__odds',
        'IWH': 'interwetten__home_win__odds',
        'IWD': 'interwetten__draw__odds',
        'IWA': 'interwetten__away_win__odds',
        'IWCH': 'interwetten_closing__home_win__odds',
        'IWCD': 'interwetten_closing__draw__odds',
        'IWCA': 'interwetten_closing__away_win__odds',
        'LBH': 'ladbrokes__home_win__odds',
        'LBD': 'ladbrokes__draw__odds',
        'LBA': 'ladbrokes__away_win__odds',
        'LBAHH': 'ladbrokes__asian_handicap_home_team__odds',
        'LBAHA': 'ladbrokes__asian_handicap_away_team__odds',
        'PSH': 'pinnacle__home_win__odds',
        'PSD': 'pinnacle__draw__odds',
        'PSA': 'pinnacle__away_win__odds',
        'P>2.5': 'pinnacle__over_2.5__odds',
        'P<2.5': 'pinnacle__under_2.5__odds',
        'PAHH': 'pinnacle__asian_handicap_home_team__odds',
        'PAHA': 'pinnacle__asian_handicap_away_team__odds',
        'PSCH': 'pinnacle_closing__home_win__odds',
        'PSCD': 'pinnacle_closing__draw__odds',
        'PSCA': 'pinnacle_closing__away_win__odds',
        'PC>2.5': 'pinnacle_closing__over_2.5__odds',
        'PC<2.5': 'pinnacle_closing__under_2.5__odds',
        'PCAHH': 'pinnacle_closing__asian_handicap_home_team__odds',
        'PCAHA': 'pinnacle_closing__asian_handicap_away_team__odds',
        'SOH': 'sporting__home_win__odds',
        'SOD': 'sporting__draw__odds',
        'SOA': 'sporting__away_win__odds',
        'SBH': 'sportingbet__home_win__odds',
        'SBD': 'sportingbet__draw__odds',
        'SBA': 'sportingbet__away_win__odds',
        'SJH': 'stanjames__home_win__odds',
        'SJD': 'stanjames__draw__odds',
        'SJA': 'stanjames__away_win__odds',
        'SYH': 'stanleybet__home_win__odds',
        'SYD': 'stanleybet__draw__odds',
        'SYA': 'stanleybet__away_win__odds',
        'VCH': 'vcbet__home_win__odds',
        'VCD': 'vcbet__draw__odds',
        'VCA': 'vcbet__away_win__odds',
        'VCCH': 'vcbet_closing__home_win__odds',
        'VCCD': 'vcbet_closing__draw__odds',
        'VCCA': 'vcbet_closing__away_win__odds',
        'WHH': 'williamhill__home_win__odds',
        'WHD': 'williamhill__draw__odds',
        'WHA': 'williamhill__away_win__odds',
        'WHCH': 'williamhill_closing__home_win__odds',
        'WHCD': 'williamhill_closing__draw__odds',
        'WHCA': 'williamhill_closing__away_win__odds',
        'MaxH': 'market_maximum__home_win__odds',
        'MaxD': 'market_maximum__draw__odds',
        'MaxA': 'market_maximum__away_win__odds',
        'Max>2.5': 'market_maximum__over_2.5__odds',
        'Max<2.5': 'market_maximum__under_2.5__odds',
        'MaxAHH': 'market_maximum__asian_handicap_home_team__odds',
        'MaxAHA': 'market_maximum__asian_handicap_away_team__odds',
        'MaxCH': 'market_maximum_closing__home_win__odds',
        'MaxCD': 'market_maximum_closing__draw__odds',
        'MaxCA': 'market_maximum_closing__away_win__odds',
        'MaxC>2.5': 'market_maximum_closing__over_2.5__odds',
        'MaxC<2.5': 'market_maximum_closing__under_2.5__odds',
        'MaxCAHH': 'market_maximum_closing__asian_handicap_home_team__odds',
        'MaxCAHA': 'market_maximum_closing__asian_handicap_away_team__odds',
        'AvgH': 'market_average__home_win__odds',
        'AvgD': 'market_average__draw__odds',
        'AvgA': 'market_average__away_win__odds',
        'Avg>2.5': 'market_average__over_2.5__odds',
        'Avg<2.5': 'market_average__under_2.5__odds',
        'AvgAHH': 'market_average__asian_handicap_home_team__odds',
        'AvgAHA': 'market_average__asian_handicap_away_team__odds',
        'AvgCH': 'market_average_closing__home_win__odds',
        'AvgCD': 'market_average_closing__draw__odds',
        'AvgCA': 'market_average_closing__away_win__odds',
        'AvgC>2.5': 'market_average_closing__over_2.5__odds',
        'AvgC<2.5': 'market_average_closing__under_2.5__odds',
        'AvgCAHH': 'market_average_closing__asian_handicap_home_team__odds',
        'AvgCAHA': 'market_average_closing__asian_handicap_away_team__odds',
        'FTHG': 'home_team__full_time_goals',
        'FTAG': 'away_team__full_time_goals',
        'HTHG': 'home_team__half_time_goals',
        'HTAG': 'away_team__half_time_goals',
        'HS': 'home_team__shots',
        'AS': 'away_team__shots',
        'HST': 'home_team__shots_on_target',
        'AST': 'away_team__shots_on_target',
        'HHW': 'home_team__hit_woodork',
        'AHW': 'away_team__hit_woodork',
        'HC': 'home_team__corners',
        'AC': 'away_team__corners',
        'HF': 'home_team__fouls_committed',
        'AF': 'away_team__fouls_committed',
        'HFKC': 'home_team__free_kicks_conceded',
        'AFKC': 'away_team__free_kicks_conceded',
        'HO': 'home_team__offsides',
        'AO': 'away_team__offsides',
        'HY': 'home_team__yellow_cards',
        'AY': 'away_team__yellow_cards',
        'HR': 'home_team__red_cards',
        'AR': 'away_team__red_cards',
        'HBP': 'home_team__bookings_points',
        'ABP': 'away_team__bookings_points',
    }

    @classmethod
    def _get_schema(cls):
        return [
            ('league', object),
            ('division', int),
            ('year', int),
            ('home_team', object),
            ('away_team', object),
            ('date', np.datetime64),
            ('bet365__size_of_asian_handicap_home_team__odds', object),
            ('ladbrokes__size_of_asian_handicap_home_team__odds', object),
            ('betbrain__size_of_asian_handicap_home_team__odds', object),
            ('gamebookers__size_of_handicap_home_team__odds', object),
            ('market_average__size_of_handicap_home_team__odds', object),
            ('market_average_closing__size_of_asian_handicap_home_team__odds', object),
            ('bet365__home_win__odds', float),
            ('bet365__draw__odds', float),
            ('bet365__away_win__odds', float),
            ('bet365__over_2.5__odds', float),
            ('bet365__under_2.5__odds', float),
            ('bet365__asian_handicap_home_team__odds', float),
            ('bet365__asian_handicap_away_team__odds', float),
            ('bet365_closing__home_win__odds', float),
            ('bet365_closing__draw__odds', float),
            ('bet365_closing__away_win__odds', float),
            ('bet365_closing__over_2.5__odds', float),
            ('bet365_closing__under_2.5__odds', float),
            ('bet365_closing__asian_handicap_home_team__odds', float),
            ('bet365_closing__asian_handicap_away_team__odds', float),
            ('betbrain_maximum__home_win__odds', float),
            ('betbrain_maximum__draw__odds', float),
            ('betbrain_maximum__away_win__odds', float),
            ('betbrain_maximum__over_2.5__odds', float),
            ('betbrain_maximum__under_2.5__odds', float),
            ('betbrain_maximum__asian_handicap_home_team__odds', float),
            ('betbrain_maximum__asian_handicap_away_team__odds', float),
            ('betbrain_average__home_win__odds', float),
            ('betbrain_average__draw_win__odds', float),
            ('betbrain_average__away_win__odds', float),
            ('betbrain_average__over_2.5__odds', float),
            ('betbrain_average__under_2.5__odds', float),
            ('betbrain_average__asian_handicap_home_team__odds', float),
            ('betbrain_average__asian_handicap_away_team__odds', float),
            ('betwin__home_win__odds', float),
            ('betwin__draw__odds', float),
            ('betwin__away_win__odds', float),
            ('betwin_closing__home_win__odds', float),
            ('betwin_closing__draw__odds', float),
            ('betwin_closing__away_win__odds', float),
            ('bluesquare__home_win__odds', float),
            ('bluesquare__draw__odds', float),
            ('bluesquare__away_win__odds', float),
            ('gamebookers__home_win__odds', float),
            ('gamebookers__draw__odds', float),
            ('gamebookers__away_win__odds', float),
            ('gamebookers__over_2.5__odds', float),
            ('gamebookers__under_2.5__odds', float),
            ('gamebookers__asian_handicap_home_team__odds', float),
            ('gamebookers__asian_handicap_away_team__odds', float),
            ('interwetten__home_win__odds', float),
            ('interwetten__draw__odds', float),
            ('interwetten__away_win__odds', float),
            ('interwetten_closing__home_win__odds', float),
            ('interwetten_closing__draw__odds', float),
            ('interwetten_closing__away_win__odds', float),
            ('ladbrokes__home_win__odds', float),
            ('ladbrokes__draw__odds', float),
            ('ladbrokes__away_win__odds', float),
            ('ladbrokes__asian_handicap_home_team__odds', float),
            ('ladbrokes__asian_handicap_away_team__odds', float),
            ('pinnacle__home_win__odds', float),
            ('pinnacle__draw__odds', float),
            ('pinnacle__away_win__odds', float),
            ('pinnacle__over_2.5__odds', float),
            ('pinnacle__under_2.5__odds', float),
            ('pinnacle__asian_handicap_home_team__odds', float),
            ('pinnacle__asian_handicap_away_team__odds', float),
            ('pinnacle_closing__home_win__odds', float),
            ('pinnacle_closing__draw__odds', float),
            ('pinnacle_closing__away_win__odds', float),
            ('pinnacle_closing__over_2.5__odds', float),
            ('pinnacle_closing__under_2.5__odds', float),
            ('pinnacle_closing__asian_handicap_home_team__odds', float),
            ('pinnacle_closing__asian_handicap_away_team__odds', float),
            ('sporting__home_win__odds', float),
            ('sporting__draw__odds', float),
            ('sporting__away_win__odds', float),
            ('sportingbet__home_win__odds', float),
            ('sportingbet__draw__odds', float),
            ('sportingbet__away_win__odds', float),
            ('stanjames__home_win__odds', float),
            ('stanjames__draw__odds', float),
            ('stanjames__away_win__odds', float),
            ('stanleybet__home_win__odds', float),
            ('stanleybet__draw__odds', float),
            ('stanleybet__away_win__odds', float),
            ('vcbet__home_win__odds', float),
            ('vcbet__draw__odds', float),
            ('vcbet__away_win__odds', float),
            ('vcbet_closing__home_win__odds', float),
            ('vcbet_closing__draw__odds', float),
            ('vcbet_closing__away_win__odds', float),
            ('williamhill__home_win__odds', float),
            ('williamhill__draw__odds', float),
            ('williamhill__away_win__odds', float),
            ('williamhill_closing__home_win__odds', float),
            ('williamhill_closing__draw__odds', float),
            ('williamhill_closing__away_win__odds', float),
            ('market_maximum__home_win__odds', float),
            ('market_maximum__draw__odds', float),
            ('market_maximum__away_win__odds', float),
            ('market_maximum__over_2.5__odds', float),
            ('market_maximum__under_2.5__odds', float),
            ('market_maximum__asian_handicap_home_team__odds', float),
            ('market_maximum__asian_handicap_away_team__odds', float),
            ('market_maximum_closing__home_win__odds', float),
            ('market_maximum_closing__draw__odds', float),
            ('market_maximum_closing__away_win__odds', float),
            ('market_maximum_closing__over_2.5__odds', float),
            ('market_maximum_closing__under_2.5__odds', float),
            ('market_maximum_closing__asian_handicap_home_team__odds', float),
            ('market_maximum_closing__asian_handicap_away_team__odds', float),
            ('market_average__home_win__odds', float),
            ('market_average__draw__odds', float),
            ('market_average__away_win__odds', float),
            ('market_average__over_2.5__odds', float),
            ('market_average__under_2.5__odds', float),
            ('market_average__asian_handicap_home_team__odds', float),
            ('market_average__asian_handicap_away_team__odds', float),
            ('market_average_closing__home_win__odds', float),
            ('market_average_closing__draw__odds', float),
            ('market_average_closing__away_win__odds', float),
            ('market_average_closing__over_2.5__odds', float),
            ('market_average_closing__under_2.5__odds', float),
            ('market_average_closing__asian_handicap_home_team__odds', float),
            ('market_average_closing__asian_handicap_away_team__odds', float),
            ('home_team__full_time_goals', int),
            ('away_team__full_time_goals', int),
            ('home_team__half_time_goals', int),
            ('away_team__half_time_goals', int),
            ('home_team__shots', int),
            ('away_team__shots', int),
            ('home_team__shots_on_target', int),
            ('away_team__shots_on_target', int),
            ('home_team__hit_woodork', int),
            ('away_team__hit_woodork', int),
            ('home_team__corners', int),
            ('away_team__corners', int),
            ('home_team__fouls_committed', int),
            ('away_team__fouls_committed', int),
            ('home_team__free_kicks_conceded', int),
            ('away_team__free_kicks_conceded', int),
            ('home_team__offsides', int),
            ('away_team__offsides', int),
            ('home_team__yellow_cards', int),
            ('away_team__yellow_cards', int),
            ('home_team__red_cards', int),
            ('away_team__red_cards', int),
            ('home_team__bookings_points', float),
            ('away_team__bookings_points', float),
        ]

    @classmethod
    def _get_outcomes(cls):
        return OUTCOMES

    @classmethod
    @lru_cache
    def _get_params(cls):
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
                            int(division) + int('0' in divisions)
                            if division != 'C'
                            else 5
                        ],
                        'year': [year],
                    }
                else:
                    years = _read_csv(urljoin(URL, url), parse_dates='Date')['Season']
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

    @lru_cache
    def _get_data(self):

        # Training data
        data_container = []
        urls = _param_grid_to_csv_urls(self.param_grid_)
        for params, url in track(urls, description='Football-Data.co.uk:'):

            data = _read_csv(url, parse_dates='Date').replace('#REF!', np.nan)

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
                    if 'Unnamed' in col or col in self._removed_cols
                ],
            ).rename(columns=self._cols_mapping)
            data_container.append(data)

        # Fixtures data
        data = _read_csv(join(URL, 'fixtures.csv'), parse_dates='Date')
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
            pd.DataFrame(self.get_all_params()).groupby(['league', 'division']).max()
        ).reset_index()
        data = pd.merge(data, years, how='left')
        data = data.drop(
            columns=[
                col
                for col in data.columns
                if 'Unnamed' in col or col in self._removed_cols
            ]
        ).rename(columns=self._cols_mapping)
        data_container.append(data)

        # Combine data
        data = pd.concat(data_container, ignore_index=True)

        return data.sort_values(['league', 'division', 'year'], ignore_index=True)
