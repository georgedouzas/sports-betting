from __future__ import annotations

import asyncio
import io
from datetime import datetime
import warnings
from pathlib import Path

import aiohttp
from prefect import flow, task
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from git import Repo
from prefect_github import GitHubCredentials

DATA_PATH = Path(__file__).parent / 'data' / 'soccer'
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
SCHEMA = [
    ('league', object),
    ('division', np.int64),
    ('year', np.int64),
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
    ('target__home_team__full_time_goals', np.int64),
    ('target__away_team__full_time_goals', np.int64),
    ('target__home_team__half_time_goals', np.int64),
    ('target__away_team__half_time_goals', np.int64),
    ('target__home_team__shots', np.int64),
    ('target__away_team__shots', np.int64),
    ('target__home_team__shots_on_target', np.int64),
    ('target__away_team__shots_on_target', np.int64),
    ('target__home_team__hit_woodork', np.int64),
    ('target__away_team__hit_woodork', np.int64),
    ('target__home_team__corners', np.int64),
    ('target__away_team__corners', np.int64),
    ('target__home_team__fouls_committed', np.int64),
    ('target__away_team__fouls_committed', np.int64),
    ('target__home_team__free_kicks_conceded', np.int64),
    ('target__away_team__free_kicks_conceded', np.int64),
    ('target__home_team__offsides', np.int64),
    ('target__away_team__offsides', np.int64),
    ('target__home_team__yellow_cards', np.int64),
    ('target__away_team__yellow_cards', np.int64),
    ('target__home_team__red_cards', np.int64),
    ('target__away_team__red_cards', np.int64),
    ('target__home_team__bookings_points', float),
    ('target__away_team__bookings_points', float),
]
CONNECTIONS_LIMIT = 20


async def _read_url_content_async(client: aiohttp.ClientSession, url: str) -> str:
    """Read asynchronously the URL content."""
    async with client.get(url) as response:
        with io.StringIO(await response.text(encoding='ISO-8859-1')) as text_io:
            return text_io.getvalue()


async def _read_urls_content_async(urls: list[str]) -> list[str]:
    """Read asynchronously the URLs content."""
    async with aiohttp.ClientSession(
        raise_for_status=True,
        connector=aiohttp.TCPConnector(limit=CONNECTIONS_LIMIT),
    ) as client:
        futures = [_read_url_content_async(client, url) for url in urls]
        return await asyncio.gather(*futures)


def _read_urls_content(urls: list[str]) -> list[str]:
    """Read the URLs content."""
    return asyncio.run(_read_urls_content_async(urls))


def _read_csvs(urls: list[str]) -> list[pd.DataFrame]:
    """Read the CSVs."""
    urls_content = _read_urls_content(urls)
    csvs = []
    for content in urls_content:
        names = pd.read_csv(
            io.StringIO(content), nrows=0, encoding='ISO-8859-1'
        ).columns.to_list()
        csv = pd.read_csv(
            io.StringIO(content),
            names=names,
            skiprows=1,
            encoding='ISO-8859-1',
            on_bad_lines='skip',
        )
        csvs.append(csv)
    return csvs


def _preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    data = data.drop(
        columns=[
            col for col in data.columns if 'Unnamed' in col or col in REMOVED_COLS
        ],
    ).rename(columns=COLS_MAPPING)
    schema_cols = [col for col, _ in SCHEMA]
    data = data.merge(pd.DataFrame(columns=schema_cols), how='outer')
    data = data[schema_cols]
    data = data.set_index('date').sort_values('date')
    data = data[~data.index.isna()]
    return data


def _convert_data_types(data: pd.DataFrame) -> pd.DataFrame:
    """Cast the data type of columns."""
    data_types = {data_type for _, data_type in SCHEMA}
    for data_type in data_types:
        converted_cols = list(
            {
                col
                for col, selected_data_type in SCHEMA
                if selected_data_type is data_type and col in data.columns
            },
        )
        if converted_cols:
            data_converted_cols = data[converted_cols]
            if data_type is float or data_type is np.int64:
                data_converted_cols = data_converted_cols.infer_objects(
                    copy=False
                ).replace(('-', '`', 'x'), np.nan)
                data_converted_cols = data_converted_cols.infer_objects().fillna(
                    -1 if data_type is np.int64 else np.nan,
                )
            data[converted_cols] = (
                data_converted_cols.to_numpy().astype(data_type)
                if data_type is not np.datetime64
                else pd.to_datetime(data_converted_cols.iloc[:, 0])
            )
    return data


def _get_output_cols_mapping(home: bool, cols: pd.Index) -> dict:
    """Get the names mapping of output columns."""
    suffix1 = "for" if home else "against"
    suffix2 = "against" if home else "for"
    output_cols_mapping = {
        col: f'{col.split("__")[-1]}_{suffix1 if "home" in col else suffix2}'.replace(
            'full_time_', ''
        )
        for col in cols
        if col not in ('home_team', 'away_team')
    }
    output_cols_mapping.update({'home_team': 'team', 'away_team': 'team'})
    return output_cols_mapping


def _extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """Extract high level features for modelling data."""

    # Columns
    team_cols = ['home_team', 'away_team']
    target_cols = [
        col
        for col in data.columns
        if col.startswith('target')
        and col.endswith(
            (
                'full_time_goals',
                'shots',
                'shots_on_target',
                'corners',
                'fouls_commited',
                'cards',
            )
        )
    ]
    odds_cols = [
        col
        for col in data.columns
        if col.startswith('odds__market')
        and 'closing' not in col
        and col.endswith(
            (
                'home_win__full_time_goals',
                'draw__full_time_goals',
                'away_win__full_time_goals',
                'over_2.5__full_time_goals',
                'under_2.5__full_time_goals',
            )
        )
    ]

    # Features data
    features_data = data[team_cols + target_cols].copy()
    home_features_data = features_data.drop(columns='away_team').rename(
        columns=_get_output_cols_mapping(True, features_data.columns)
    )
    away_features_data = features_data.drop(columns='home_team').rename(
        columns=_get_output_cols_mapping(False, features_data.columns)
    )
    features_data = (
        pd.concat([home_features_data, away_features_data])
        .reset_index()
        .set_index(['team', 'date'])
        .sort_index()
    )
    features_data['adj_goals_for'] = (
        features_data['goals_for']
        + features_data['red_cards_for']
        + features_data['yellow_cards_for'] / 20
        + features_data['shots_on_target_for'] / 10
        + features_data['shots_on_target_for'] / 50
        + features_data['corners_for'] / 100
    )
    features_data['adj_goals_against'] = (
        features_data['goals_against']
        + features_data['red_cards_against']
        + features_data['yellow_cards_against'] / 20
        + features_data['shots_on_target_against'] / 10
        + features_data['shots_on_target_against'] / 50
        + features_data['corners_against'] / 100
    )
    features_data['points'] = 3 * (
        features_data['goals_for'] > features_data['goals_against']
    ) + (features_data['goals_for'] == features_data['goals_against'])
    features_data['adj_points'] = (
        3 * (features_data['adj_goals_for'] > features_data['adj_goals_against'] + 0.25)
        + 1.0
        * (
            np.abs(features_data['adj_goals_for'] - features_data['adj_goals_against'])
            <= 0.25
        )
    ).astype(int)
    features_data = features_data[
        [
            'points',
            'adj_points',
            'goals_for',
            'goals_against',
            'adj_goals_for',
            'adj_goals_against',
        ]
    ]

    features_cols = features_data.columns
    features_avg_cols = [f'{col}__avg' for col in features_cols]
    features_latest_avg_cols = [f'{col}__latest_avg' for col in features_cols]
    features_data[features_avg_cols] = (
        features_data.groupby('team')[features_cols].expanding().mean().to_numpy()
    )
    features_data[features_avg_cols] = features_data.groupby('team')[
        features_avg_cols
    ].shift(1)
    features_data[features_latest_avg_cols] = (
        features_data.groupby('team')[features_cols]
        .rolling(window=3, min_periods=1)
        .mean()
        .to_numpy()
    )
    features_data[features_latest_avg_cols] = features_data.groupby('team')[
        features_latest_avg_cols
    ].shift(1)
    features_data = features_data.drop(columns=features_cols).reset_index()

    # Input data
    input_data = data[team_cols + odds_cols].copy()
    input_data = (
        input_data.reset_index()
        .merge(
            features_data.rename(
                columns={
                    col: f'home__{col}'
                    for col in features_data.columns
                    if col.endswith('avg')
                }
            ),
            left_on=['date', 'home_team'],
            right_on=['date', 'team'],
        )
        .drop(columns='team')
        .set_index('date')
    )
    input_data = (
        input_data.reset_index()
        .merge(
            features_data.rename(
                columns={
                    col: f'away__{col}'
                    for col in features_data.columns
                    if col.endswith('avg')
                }
            ),
            left_on=['date', 'away_team'],
            right_on=['date', 'team'],
        )
        .drop(columns='team')
        .set_index('date')
    )

    # Output data
    output_data = data[target_cols].copy()

    # Data
    data = pd.concat([input_data, output_data], axis=1)

    return data


@task(
    description='Extract the URLs to download the raw training data',
    tags=['raw', 'training', 'download'],
    retries=10,
    retry_delay_seconds=5,
)
def extract_raw_training_urls() -> list[tuple[str, str, str]]:
    """Extract the URLs to download the raw training data."""
    base_urls = ['/'.join([URL, base_url]) for base_url in BASE_URLS]
    bsObjs = [
        BeautifulSoup(html, features='html.parser')
        for html in _read_urls_content(base_urls)
    ]
    training_urls = []
    for base_url, bsObj in zip(BASE_URLS, bsObjs):
        urls = {
            el.get('href')
            for el in bsObj.find_all('a')
            if el.get('href').endswith('csv')
        }
        league = base_url.replace('.php', '')
        if base_url[0].islower():
            league = league[:-1].capitalize()
            for url in urls:
                *_, year, division = url.split('/')
                year = datetime.strptime(year[2:], '%y').year
                division = division.replace('.csv', '')[-1]
                if 'C' in LEAGUES_MAPPING[league][1:]:
                    division = int(division) + 1 if division != 'C' else 5
                else:
                    division = int(division)
                training_urls.append((league, division, year, '/'.join([URL, url])))
        else:
            url, *_ = urls
            training_urls.append((league, 1, None, '/'.join([URL, url])))
    return training_urls


@task(
    description='Download the raw training data',
    tags=['raw', 'training', 'download'],
    retries=10,
    retry_delay_seconds=5,
)
def download_raw_training_data(
    training_urls: list[tuple[str, str, str]],
) -> list[tuple[str, int, int, pd.DataFrame]]:
    """Download the raw training data."""
    data = _read_csvs([url for *_, url in training_urls])
    data = [
        (league, division, year, league_data)
        for (league, division, year, _), league_data in zip(training_urls, data)
    ]
    return data


@task(
    description='Download the raw fixtures data',
    tags=['raw', 'fixtures', 'download'],
    retries=10,
    retry_delay_seconds=5,
)
def download_raw_fixtures_data() -> pd.DataFrame:
    """Download the raw fixtures data."""
    data = _read_csvs(['/'.join([URL, 'fixtures.csv'])])[0]
    return data


@task(description='Process the raw training data', tags=['raw', 'training', 'process'])
def process_raw_training_data(
    data: list[tuple[str, int, int, pd.DataFrame]],
) -> list[tuple[str, int, int, pd.DataFrame]]:
    """Process the raw training data."""
    processed_data = []
    for league, division, year, league_data in data:
        league_data = league_data.replace('^#', np.nan, regex=True)
        try:
            league_data['Date'] = pd.to_datetime(league_data['Date'], format='%d/%m/%Y')
        except ValueError:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                league_data['Date'] = pd.to_datetime(
                    league_data['Date'], infer_datetime_format=True
                )
        if year is not None:
            league_data = league_data.assign(
                league=league, division=division, year=year
            )
            league_data = _preprocess_data(league_data)
            league_data = _convert_data_types(league_data)
            processed_data.append((league, division, year, league_data))
        else:
            league_data = league_data.assign(league=league, division=division)
            league_data['year'] = league_data['Season'].apply(
                lambda season: (
                    season
                    if not isinstance(season, str)
                    else int(season.split('/')[-1])
                ),
            )
            for league_year in league_data['year'].unique():
                league_year_data = league_data[league_data['year'] == league_year]
                league_year_data = _preprocess_data(league_year_data)
                league_year_data = _convert_data_types(league_year_data)
                processed_data.append((league, division, league_year, league_year_data))
    return processed_data


@task(description='Process the raw fixtures data', tags=['raw', 'fixtures', 'process'])
def process_raw_fixtures_data(
    data: pd.DataFrame, processed_training_data: pd.DataFrame
) -> pd.DataFrame:
    """Process the raw fixtures data."""
    data = data.rename(columns={'ï»¿Div': 'Div'})
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data = data.dropna(axis=0, how='any', subset=['Div', 'HomeTeam', 'AwayTeam'])
    inv_leagues_mapping = {v[0]: k for k, v in LEAGUES_MAPPING.items()}
    data['league'] = data['Div'].apply(lambda div: inv_leagues_mapping[div[:-1]])
    data['division'] = data['Div'].apply(lambda div: div[-1])
    data['divisions'] = data['league'].apply(lambda league: LEAGUES_MAPPING[league][1:])
    data['division'] = (
        data[['division', 'divisions']]
        .apply(
            lambda row: (
                row.iloc[0]
                if 'C' not in row.iloc[1]
                else (int(row.iloc[0]) + 1 if row.iloc[0] != 'C' else 5)
            ),
            axis=1,
        )
        .astype(int)
    )
    years = (
        pd.DataFrame(
            [key for *key, _ in processed_training_data],
            columns=['league', 'division', 'year'],
        )
        .groupby(['league', 'division'])
        .max()
        .reset_index()
    )
    data = data.merge(years, how='left')
    data = _preprocess_data(data)
    data = _convert_data_types(data)
    data = data.loc[
        data.index >= pd.Timestamp(pd.to_datetime('today').date()),
        [col for col in data.columns if not col.startswith('target')],
    ]
    return data


@task(
    description='Transform the processed data',
    tags=['processed', 'training', 'fixtures' 'transform'],
)
def transform_processed_data(
    processed_training_data: list[tuple[str, int, int, pd.DataFrame]],
    processed_fixtures_data: pd.DataFrame,
) -> list[tuple[str, int, int, pd.DataFrame]]:
    """Transform the processed data."""
    modelling_data = []
    for league, division, year, training_data in processed_training_data:
        mask = (
            (processed_fixtures_data['league'] == league)
            & (processed_fixtures_data['division'] == division)
            & (processed_fixtures_data['year'] == year)
        )
        data = pd.concat([training_data, processed_fixtures_data[mask]])
        features_data = _extract_features(data).assign(
            league=league, division=division, year=year
        )
        cols = ['league', 'division', 'year']
        features_data = features_data[
            cols + [col for col in features_data.columns if col not in cols]
        ]
        modelling_data.append(
            (league, division, year, training_data.shape[0], features_data)
        )
    return modelling_data


@task(
    description='Extract the modelling training data',
    tags=['modelling', 'training', 'extract'],
)
def extract_modelling_training_data(modelling_data: pd.DataFrame) -> pd.DataFrame:
    """Extract the training modelling data."""
    modelling_training_data = [
        (league, division, year, data[:n])
        for league, division, year, n, data in modelling_data
    ]
    return modelling_training_data


@task(
    description='Extract the modelling fixtures data',
    tags=['modelling', 'fixtures', 'extract'],
)
def extract_modelling_fixtures_data(modelling_data: pd.DataFrame) -> pd.DataFrame:
    """Extract the modelling fixtures data."""
    modelling_fixtures_data = pd.concat([data[n:] for *_, n, data in modelling_data])
    modelling_fixtures_data = modelling_fixtures_data.sort_values(
        ['date', 'league', 'division']
    )
    return modelling_fixtures_data


@task(
    description='Save the processed training data',
    tags=['processed', 'training', 'save'],
)
def save_processed_training_data(
    data: list[tuple[str, int, int, pd.DataFrame]],
) -> None:
    """Save the processed training data."""
    (DATA_PATH / 'processed').mkdir(parents=True, exist_ok=True)
    for league, division, year, league_data in data:
        league_data.to_csv(DATA_PATH / 'processed' / f'{league}_{division}_{year}.csv')


@task(
    description='Save the modelling training data',
    tags=['modelling', 'training', 'save'],
)
def save_modelling_training_data(
    data: list[tuple[str, int, int, pd.DataFrame]],
) -> None:
    """Save the modelling training data."""
    (DATA_PATH / 'modelling').mkdir(parents=True, exist_ok=True)
    for league, division, year, league_data in data:
        league_data.to_csv(DATA_PATH / 'modelling' / f'{league}_{division}_{year}.csv')


@task(
    description='Save the processed fixtures data',
    tags=['processed', 'fixtures', 'save'],
)
def save_processed_fixtures_data(data: pd.DataFrame) -> None:
    """Save the processed fixtures data."""
    (DATA_PATH / 'processed').mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH / 'processed' / 'fixtures.csv')


@task(
    description='Save the modelling fixtures data',
    tags=['modelling', 'fixtures', 'save'],
)
def save_modelling_fixtures_data(data: pd.DataFrame) -> None:
    """Save the modelling fixtures data."""
    (DATA_PATH / 'modelling').mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH / 'modelling' / 'fixtures.csv')


@task(description='Sync the repository')
def sync_repo() -> None:
    """Sync the repository."""
    block = GitHubCredentials.load('sports-betting-data-block')
    repo = Repo(Path(__file__).parent)
    if repo.is_dirty() or repo.untracked_files:
        repo.index.add([DATA_PATH])
        repo.index.commit(f'Update soccer data {str(datetime.now().date())}')
        remote = repo.remote()
        remote.set_url(
            f'https://{block.token.get_secret_value()}@{remote.url.split("://")[-1]}',
            remote.url,
        )
        repo.git.pull('origin', 'data', '--rebase')
        repo.git.push('-u', 'origin', 'data')


@flow(description='Update the training and fixtures data')
def update_data():
    """Update the training and fixtures data."""

    # Download raw data
    urls = extract_raw_training_urls()
    raw_training_data = download_raw_training_data(urls)
    raw_fixtures_data = download_raw_fixtures_data()

    # Processed training data
    processed_training_data = process_raw_training_data(raw_training_data)
    save_processed_training_data(processed_training_data)

    # Processed fixtures data
    processed_fixtures_data = process_raw_fixtures_data(
        raw_fixtures_data, processed_training_data
    )
    save_processed_fixtures_data(processed_fixtures_data)

    # Modelling data
    modelling_data = transform_processed_data(
        processed_training_data, processed_fixtures_data
    )

    # Modelling training data
    modelling_training_data = extract_modelling_training_data(modelling_data)
    save_modelling_training_data(modelling_training_data)

    # Modelling fixtures data
    modelling_fixtures_data = extract_modelling_fixtures_data(modelling_data)
    save_modelling_fixtures_data(modelling_fixtures_data)

    sync_repo()


if __name__ == '__main__':
    update_data()
