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

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from bs4 import BeautifulSoup
from rich.progress import track

from . import TARGETS
from .._utils import _DataLoader

URL = 'http://www.football-data.co.uk'
REMOVED = [
    ('Div', None, None),
    ('Season', None, None),
    ('League', None, None),
    ('Country', None, None),
    ('Time', None, None),
    ('FTR', None, None),
    ('Res', None, None),
    ('Attendance', None, None),
    ('Referee', None, None),
    ('HTR', None, None),
    ('BbAH', None, None),
    ('Bb1X2', None, None),
    ('BbOU', None, None)
]
CREATED = [
    (None, 'league', object),
    (None, 'division', int),
    (None, 'year', int)
]
RENAMED = [
    ('HomeTeam', 'home_team', object),
    ('AwayTeam', 'away_team', object),
    ('Date', 'date', np.datetime64)
]
BOOKMAKERS_MAPPING = {
    'B365': 'bet365',
    'LB': 'ladbrokers',
    'GB': 'gamebookers',
    'BbMx': 'betbrain_maximum',
    'BbAv': 'betbrain_average',
    'BW': 'betwin',
    'BS': 'bluesquare',
    


}
CONFIG = [

    ##############
    # Input data #
    ##############

    # Removed

    

    # Created

    

    # Converted

    
    
    ('B365AH', 'bet365_size_of_handicap_home_team', object),
    ('LBAH', 'ladbrokes_size_of_handicap_home_team', object),
    ('BbAHh', 'betbrain_size_of_handicap_home_team', object),
    ('GBAH', 'gamebookers_size_of_handicap_home_team', object),
    ('AHh', 'market_size_of_handicap_home_team', object),
    ('AHCh', 'market_closing_size_of_handicap_home_team', object),
    
    ('B365H', 'bet365_home_win_odds', float),
    ('B365D', 'bet365_draw_odds', float),
    ('B365A', 'bet365_away_win_odds', float),
    ('B365>2.5', 'bet365_over_2.5_odds', float),
    ('B365<2.5', 'bet365_under_2.5_odds', float),
    ('B365AHH', 'bet365_asian_handicap_home_team_odds', float),
    ('B365AHA', 'bet365_asian_handicap_away_team_odds', float),
    ('B365CH', 'bet365_closing_home_win_odds', float),
    ('B365CD', 'bet365_closing_draw_odds', float),
    ('B365CA', 'bet365_closing_away_win_odds', float),
    ('B365C>2.5', 'bet365_closing_over_2.5_odds', float),
    ('B365C<2.5', 'bet365_closing_under_2.5_odds', float),
    ('B365CAHH', 'bet365_closing_asian_handicap_home_team_odds', float),
    ('B365CAHA', 'bet365_closing_asian_handicap_away_team_odds', float),
    
    ('BbMxH', 'betbrain_maximum_home_win_odds', float),
    ('BbMxD', 'betbrain_maximum_draw_odds', float),
    ('BbMxA', 'betbrain_maximum_away_win_odds', float),
    ('BbMx>2.5', 'betbrain_maximum_over_2.5_odds', float),
    ('BbMx<2.5', 'betbrain_maximum_under_2.5_odds', float),
    ('BbMxAHH', 'betbrain_maximum_asian_handicap_home_team_odds', float),
    ('BbMxAHA', 'betbrain_maximum_asian_handicap_away_team_odds', float),
    ('BbAvH', 'betbrain_average_home_win_odds', float),
    ('BbAvD', 'betbrain_average_draw_win_odds', float),
    ('BbAvA', 'betbrain_average_away_win_odds', float),
    ('BbAv>2.5', 'betbrain_average_over_2.5_odds', float),
    ('BbAv<2.5', 'betbrain_average_under_2.5_odds', float),
    ('BbAvAHH', 'betbrain_average_asian_handicap_home_team_odds', float),
    ('BbAvAHA', 'betbrain_average_asian_handicap_away_team_odds', float),
    
    ('BWH', 'bet_win_home_win_odds', float),
    ('BWD', 'bet_win_draw_odds', float),
    ('BWA', 'bet_win_away_win_odds', float),
    ('BWCH', 'bet_win_closing_home_win_odds', float),
    ('BWCD', 'bet_win_closing_draw_odds', float),
    ('BWCA', 'bet_win_closing_away_win_odds', float),

    ('BSH', 'blue_square_home_win_odds', float),
    ('BSD', 'blue_square_draw_odds', float),
    ('BSA', 'blue_square_away_win_odds', float),
    
    ('GBH', 'gamebookers_home_win_odds', float),
    ('GBD', 'gamebookers_draw_odds', float),
    ('GBA', 'gamebookers_away_win_odds', float),
    ('GB>2.5', 'gamebookers_over_2.5_odds', float),
    ('GB<2.5', 'gamebookers_under_2.5_odds', float),
    ('GBAHH', 'gamebookers_asian_handicap_home_team_odds', float),
    ('GBAHA', 'gamebookers_asian handicap_away_team_odds', float),
    
    ('IWH', 'interwetten_home_win_odds', float),
    ('IWD', 'interwetten_draw_odds', float),
    ('IWA', 'interwetten_away_win_odds', float),
    ('IWCH', 'interwetten_closing_home_win_odds', float),
    ('IWCD', 'interwetten_closing_draw_odds', float),
    ('IWCA', 'interwetten_closing_away_win_odds', float),
    
    ('LBH', 'ladbrokes_home_win_odds', float),
    ('LBD', 'ladbrokes_draw_odds', float),
    ('LBA', 'ladbrokes_away_win_odds', float),
    ('LBAHH', 'ladbrokes_asian_handicap_home_team_odds', float),
    ('LBAHA', 'ladbrokes_asian_handicap_away_team_odds', float),
    
    ('PSH', 'pinnacle_home_win_odds', float),
    ('PSD', 'pinnacle_draw_odds', float),
    ('PSA', 'pinnacle_away_win_odds', float),
    ('P>2.5', 'pinnacle_over_2.5_odds', float),
    ('P<2.5', 'pinnacle_under_2.5_odds', float),
    ('PAHH', 'pinnacle_asian_handicap_home_team_odds', float),
    ('PAHA', 'pinnacle_asian_handicap_away_team_odds', float),
    ('PSCH', 'pinnacle_closing_home_win_odds', float),
    ('PSCD', 'pinnacle_closing_draw_odds', float),
    ('PSCA', 'pinnacle_closing_away_win_odds', float),
    ('PC>2.5', 'pinnacle_closing_over_2.5_odds', float),    
    ('PC<2.5', 'pinnacle_closing_under_2.5_odds', float),
    ('PCAHH', 'pinnacle_closing_asian_handicap_home_team_odds', float),
    ('PCAHA', 'pinnacle_closing_asian_handicap_away_team_odds', float),

    ('SOH', 'sporting_odds_home_win_odds', float),
    ('SOD', 'sporting_odds_draw_odds', float),
    ('SOA', 'sporting_odds_away_win_odds', float),
    
    ('SBH', 'sportingbet_home_win_odds', float),
    ('SBD', 'sportingbet_draw_odds', float),
    ('SBA', 'sportingbet_away_win_odds', float),
    
    ('SJH', 'stan_james_home_win_odds', float),
    ('SJD', 'stan_james_draw_odds', float),
    ('SJA', 'stan_james_away_win_odds', float),
    
    ('SYH', 'stanleybet_home_win_odds', float),
    ('SYD', 'stanleybet_draw_odds', float),
    ('SYA', 'stanleybet_away_win_odds', float),
    
    ('VCH', 'vc_bet_home_win_odds', float),
    ('VCD', 'vc_bet_draw_odds', float),
    ('VCA', 'vc_bet_away_win_odds', float),
    ('VCCH', 'vc_bet_closing_home_win_odds', float),
    ('VCCD', 'vc_bet_closing_draw_odds', float),
    ('VCCA', 'vc_bet_closing_away_win_odds', float),

    ('WHH', 'william_hill_home_win_odds', float),
    ('WHD', 'william_hill_draw_odds', float),
    ('WHA', 'william_hill_away_win_odds', float),
    ('WHCH', 'william_hill_closing_home_win_odds', float),
    ('WHCD', 'william_hill_closing_draw_odds', float),
    ('WHCA', 'william_hill_closing_away_win_odds', float),
    
    ('MaxH', 'market_maximum_home_win_odds', float),
    ('MaxD', 'market_maximum_draw_odds', float),
    ('MaxA', 'market_maximum_away_win_odds', float),
    ('Max>2.5', 'market_maximum_over_2.5_odds', float),
    ('Max<2.5', 'market_maximum_under_2.5_odds', float),
    ('MaxAHH', 'market_maximum_asian_handicap_home_team_odds', float),
    ('MaxAHA', 'market_maximum_asian_handicap_away_team_odds', float),
    ('MaxCH', 'market_closing_maximum_home_win_odds', float),
    ('MaxCD', 'market_closing_maximum_draw_odds', float),
    ('MaxCA', 'market_closing_maximum_away_win_odds', float),
    ('MaxC>2.5', 'market_closing_maximum_over_2.5_odds', float),
    ('MaxC<2.5', 'market_closing_maximum_under_2.5_odds', float),
    ('MaxCAHH', 'market_closing_maximum_asian_handicap_home_team_odds', float),
    ('MaxCAHA', 'market_closing_maximum_asian_handicap_away_team_odds', float),

    ('AvgH', 'market_average_home_win_odds', float),
    ('AvgD', 'market_average_draw_odds', float),
    ('AvgA', 'market_average_away_win_odds', float),    
    ('Avg>2.5', 'market_average_over_2.5_odds', float),
    ('Avg<2.5', 'market_average_under_2.5_odds', float),
    ('AvgAHH', 'market_average_asian_handicap_home_team_odds', float),
    ('AvgAHA', 'market_average_asian_handicap_away_team_odds', float),
    ('AvgCH', 'market_closing_average_home_win_odds', float),
    ('AvgCD', 'market_closing_average_draw_odds', float),
    ('AvgCA', 'market_closing_average_away_win_odds', float),
    ('AvgC>2.5', 'market_closing_average_over_2.5_odds', float),
    ('AvgC<2.5', 'market_closing_average_under_2.5_odds', float),
    ('AvgCAHH', 'market_closing_average_asian_handicap_home_team_odds', float),
    ('AvgCAHA', 'market_closing_average_asian_handicap_away_team_odds', float),

    ###############
    # Output data #
    ###############

    # Goals

    ('FTHG', 'home_team__full_time_goals', int),
    ('FTAG', 'away_team__full_time_goals', int),
    ('HTHG', 'home_team__half_time_goals', int),
    ('HTAG', 'away_team__half_time_goals', int),
    
    # Shots

    ('HS', 'home_team__shots', int),
    ('AS', 'away_team__shots', int),
    ('HST', 'home_team__shots_on_target', int),
    ('AST', 'away_team__shots_on_target', int),

    # Woodwork

    ('HHW', 'home_team__hit_woodork', int),
    ('AHW', 'away_team__hit_woodork', int),

    # Corners

    ('HC', 'home_team__corners', int),
    ('AC', 'away_team__corners', int),
    
    # Fouls

    ('HF', 'home_team__fouls_committed', int),
    ('AF', 'away_team__fouls_committed', int),

    # Free kicks
    ('HFKC', 'home_team__free_kicks_conceded', int),
    ('AFKC', 'away_team__free_kicks_conceded', int),
    
    # Offsides

    ('HO', 'home_team__offsides', int),
    ('AO', 'away_team__offsides', int),
    
    # Cards

    ('HY', 'home_team__yellow_cards', int),
    ('AY', 'away_team__yellow_cards', int),
    ('HR', 'home_team__red_cards', int),
    ('AR', 'away_team__red_cards', int),

    # Bookings points
    ('HBP', 'home_team__bookings_points', float),
    ('ABP', 'away_team__bookings_points', float)

]


def _extract_leagues_urls(leagues_type):
    """Extract Football-Data.co.uk urls
    for a league type."""
    html = urlopen(urljoin(URL, 'data.php'))
    bsObj = BeautifulSoup(html.read(), features='html.parser')
    return [
        el.get('href') for el in bsObj.find(text=leagues_type).find_next().find_all('a')
    ]


def _extract_main_leagues_param_grid():
    """Extract parameter grid of main leagues."""

    # Extract urls
    urls = _extract_leagues_urls('Main Leagues')
    main_leagues_urls = {}
    for url in urls:
        html = urlopen(urljoin(URL, url))
        bsObj = BeautifulSoup(html.read(), features='html.parser')
        league = url.replace('m.php', '').capitalize()
        main_leagues_urls[league] = [el.get('href') for el in bsObj.find_all('a') if el.get('href').endswith('csv')]
    
    # Extract parameter grid
    main_leagues_param_grid = []
    for league, urls in main_leagues_urls.items():
        league_param_grid = []
        divisions = []
        for url in urls:
            _, year, division = url.split('/')
            year = datetime.strptime(year[2:], '%y').year
            div = division.replace('.csv', '')
            division = div[-1]
            param_grid = {'league': [league], 'division': division, 'year': [year], 'url': [url], 'Div': [div]}
            league_param_grid.append(param_grid)
            divisions.append(division)
        div_offset = int('0' in divisions)
        for param_grid in league_param_grid:
            param_grid['division'] = [int(param_grid['division']) + div_offset] if param_grid['division'] != 'C' else [5]
        main_leagues_param_grid += league_param_grid
    
    return ParameterGrid(main_leagues_param_grid)


def _extract_extra_leagues_param_grid():
    """Extract parameter grid of extra leagues."""
    
    # Extract urls
    urls = _extract_leagues_urls('Extra Leagues')
    extra_leagues_urls = {}
    for url in urls:
        html = urlopen(urljoin(URL, url))
        bsObj = BeautifulSoup(html.read(), features='html.parser')
        league = url.replace('.php', '')
        extra_leagues_urls[league] = list({el.get('href') for el in bsObj.find_all('a') if el.get('href').endswith('csv')})
    
    # Extract parameter grid
    extra_leagues_param_grid = []
    for league, urls in extra_leagues_urls.items():
        years = pd.read_csv(urljoin(URL, urls[0]), usecols=['Season'])['Season']
        years = list({s if type(s) is not str else int(s.split('/')[-1]) for s in years.unique()})
        extra_leagues_param_grid.append({'league': [league], 'division': [1], 'year': years, 'url': urls})
    
    return ParameterGrid(extra_leagues_param_grid)


class _FDDataLoader(_DataLoader):
    """Data loader for Football-Data.co.uk data."""

    _cols_mapping = {
        'HT': 'HomeTeam',
        'Home': 'HomeTeam',
        'AT': 'AwayTeam',
        'Away': 'AwayTeam',
        'LB': 'LBH',
        'LB.1': 'LBD',
        'LB.2': 'LBA',
        'PH': 'PSH',
        'PD': 'PSD',
        'PA': 'PSA',
        'HG': 'FTHG',
        'AG': 'FTAG'
    }

    def _fetch_full_param_grid(self):
        self.main_leagues_param_grid_ = _extract_main_leagues_param_grid()
        self.extra_leagues_param_grid_ = _extract_extra_leagues_param_grid()
        self.full_param_grid_ = []
        full_param_grid = ParameterGrid(self.main_leagues_param_grid_.param_grid + self.extra_leagues_param_grid_.param_grid)
        for param in full_param_grid:
            self.full_param_grid_.append({'division': [param['division']], 'league': [param['league']], 'year': [param['year']]})
        self.full_param_grid_ = ParameterGrid(self.full_param_grid_)
        return self

    def _fetch_data(self, odds_input):

        # Main train data
        params = pd.merge(pd.DataFrame(self.main_leagues_param_grid_), pd.DataFrame(self.param_grid_)).to_records(False)
        data_train_main = []
        for _, division, league, url, year in track(params, description='Football-Data.co.uk | Download training data of main leagues:'):
            url = urljoin(URL, url)
            names = pd.read_csv(url, nrows=0).columns
            try:
                data = pd.read_csv(url, names=names, skiprows=1)
            except (UnicodeDecodeError, pd.errors.ParserError):
                data = pd.read_csv(url, names=names, skiprows=1, encoding='ISO-8859-1')
            
            # Extract columns
            data.replace('#REF!', np.nan, inplace=True)
            data.drop(columns=[col for col in data.columns if 'Unnamed' in col], inplace=True)
            data = data.assign(league=league, division=division, year=year, test=False)
            data.rename(columns=self._cols_mapping, inplace=True)
            
            # Append data to main data
            data_train_main.append(data)

        data_train_main = pd.concat(data_train_main, ignore_index=True) if data_train_main else pd.DataFrame()
        
        # Extra train data
        params = pd.merge(pd.DataFrame(self.extra_leagues_param_grid_), pd.DataFrame(self.param_grid_)).drop(columns='year').drop_duplicates().to_records(False)
        data_train_extra = []
        for division, league, url in track(params, description='Football-Data.co.uk | Download training data of extra leagues:'):
            data = pd.read_csv(urljoin(URL, url))
            data = data.assign(league=league, division=division, year=lambda s: s.Season if type(s.Season) is not str else s.Season.split('/')[-1], test=False)
            data.rename(columns=self._cols_mapping, inplace=True)
            data_train_extra.append(data)
        data_train_extra = pd.concat(data_train_extra, ignore_index=True) if data_train_extra else pd.DataFrame()
        
        # Test data
        params = pd.DataFrame(self.main_leagues_param_grid_).drop(columns='url').groupby(['league', 'division'], as_index=False).max()
        data_test = pd.read_csv(join(URL, 'fixtures.csv'))
        data_test.drop(columns=[col for col in data_test.columns if 'Unnamed' in col], inplace=True)
        data_test = pd.merge(data_test, params[['Div', 'league', 'division', 'year']].dropna().drop_duplicates(), how='left')
        data_test['test'] = True
        data_test.rename(columns=self._cols_mapping, inplace=True)

        # Combine data
        self.data_ = pd.concat([data_train_main, data_train_extra, data_test], ignore_index=True)
        
        # Select odds types
        cols_odds = []
        for col in self.data_.columns:
            for odds_type in odds_input:
                if col.startswith(odds_type) or col.startswith(f'{self.odds_type_}_'):
                    cols_odds.append(col)
        
        return self


def load_from_football_data_soccer_data(
    param_grid=None,
    drop_na_cols=None,
    drop_na_rows=None,
    odds_input=None,
    odds_type=None,
    testing_duration=None,
    return_only_params=False
):
    """Load and return Football-Data.co.uk soccer data for model training and testing.

    parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such parameter, default=None
        The parameter grid to explore, as a dictionary mapping data parameters
        to sequences of allowed values. An empty dict signifies default
        parameters. A sequence of dicts signifies a sequence of grids to search,
        and is useful to avoid exploring parameter combinations that do not
        exist. The default value corresponds to all parameters.
    
    drop_na_cols : float, default=None
        The threshold of input columns with missing values to drop. It is a
        float in the [0.0, 1.0] range. The default value ``None``
        corresponds to ``0.0`` i.e. all columns are kept while the value
        ``1.0`` keeps only columns with non missing values.
    
    drop_na_rows : bool, default=None
        The threshold of rows with missing values to drop. It is a
        float in the [0.0, 1.0] range. The default value ``None``
        corresponds to ``0.0`` i.e. all rows are kept while the value
        ``1.0`` keeps only rows with non missing values.

    odds_input : list, default=['market_average', 'market_maximum']
        A list of the prefixes of the odds columns to be included in the input 
        data. The default value returns the market average and maximum odds data.
        
    odds_type : str, default='market_average'
        The prefix of the odds column to be used for generating the odds
        data. The default value returns the market average odds data.
    
    testing_duration : int, default=None
        The number of future weeks to include in the testing data. The
        default value corresponds to one week.
    
    return_only_params : bool, default=False
        When set to ``True`` only the available parameter grid is returned.

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        training : (X, Y, O) tuple
            A tuple of 'X' and 'Y', both as pandas
            DataFrames, that represent the training input data and 
            multi-output targets, respectively.
        testing : (X, None, O) tuple
            A pandas DataFrame that represents the testing input data.
        removed : :class:`sklearn.utils.Bunch`
            The dropped columns and rows as attributes.
        params : :class:`sklearn.utils.Bunch`
            The selected and available parameter grids as pandas DataFrames.
    """
    data_loader = _FDDataLoader(config=CONFIG, targets=TARGETS, param_grid=param_grid, drop_na_cols=drop_na_cols, drop_na_rows=drop_na_rows, odds_type=odds_type, testing_duration=testing_duration)
    return data_loader.load(return_only_params, odds_input=odds_input)
