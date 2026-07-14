"""Implements the sources backed by the football-data.co.uk feed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar, Self

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .. import ParamGrid
from ._base import BaseOddsSource, BaseSource, BaseStatsSource, RawItem, RawPayload
from ._fetch import ENCODING, read_csv_content
from ._utils import market_outcomes

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
SNAPSHOT_IDENTITY = ['date', 'league', 'division', 'year', 'home_team', 'away_team']
MARKETS = ['home_win', 'draw', 'away_win', 'over_2.5', 'under_2.5']
PROVIDERS = ['market_average', 'market_maximum']
HALF_TIME_COLS = {
    'target__home_team__half_time_goals': 'home_half_goals',
    'target__away_team__half_time_goals': 'away_half_goals',
}
CHAMPIONSHIP_DIVISION = 5
CENTURY_PIVOT = 68
FEED_TIMEZONE = 'Europe/London'
KEY_PARTS = 3
DRAW_MARGIN = 0.25
FIXTURES_KEY = 'fixtures'


def _preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data.

    Each non-closing odds column is back-filled from its closing twin when it is absent or empty, before the closing
    columns are dropped, so the historical odds are not lost.
    """
    data = data.drop(
        columns=[col for col in data.columns if 'Unnamed' in col or col in REMOVED_COLS],
    ).rename(columns=COLS_MAPPING)
    backfilled = {}
    for col in data.columns:
        if 'closing' in col:
            non_closing_col = col.replace('_closing', '')
            if non_closing_col not in data.columns or np.all(data[non_closing_col].isna()):
                backfilled[non_closing_col] = data[col]
    if backfilled:
        replaced = [col for col in backfilled if col in data.columns]
        data = pd.concat([data.drop(columns=replaced), pd.DataFrame(backfilled)], axis=1)
    data = data.drop(columns=[col for col in data.columns if 'closing' in col])
    schema_cols = [col for col, _ in SCHEMA]
    data = data.merge(pd.DataFrame(columns=schema_cols), how='outer')
    data = data[schema_cols]
    data = data.set_index('date').sort_values('date')
    return data[~data.index.isna()]


def _convert_data_types(data: pd.DataFrame) -> pd.DataFrame:
    """Cast the data type of columns.

    Missing integers become `-1`, which is the sentinel the snapshots read to decide whether a match has been played.
    """
    data_types = {data_type for _, data_type in SCHEMA}
    for data_type in data_types:
        converted_cols = list(
            {col for col, selected_data_type in SCHEMA if selected_data_type is data_type and col in data.columns},
        )
        if converted_cols:
            data_converted_cols = data[converted_cols]
            if data_type is float or data_type is np.int64:
                data_converted_cols = data_converted_cols.infer_objects().replace(('-', '`', 'x'), np.nan)
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
    suffix1 = 'for' if home else 'against'
    suffix2 = 'against' if home else 'for'
    output_cols_mapping = {
        col: f'{col.split("__")[-1]}_{suffix1 if "home" in col else suffix2}'.replace('full_time_', '')
        for col in cols
        if col not in ('home_team', 'away_team')
    }
    output_cols_mapping.update({'home_team': 'team', 'away_team': 'team'})
    return output_cols_mapping


def _rename_modelling_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Rename verbose columns to the concise modelling convention."""
    mapping = {}
    for col in data.columns:
        if col.startswith('odds__') and col.endswith('__full_time_goals'):
            _, provider, market, _ = col.split('__')
            mapping[col] = f'{provider}__{market}'
        elif col.startswith('target__home_team__'):
            stat = col.removeprefix('target__home_team__')
            mapping[col] = 'home_goals' if stat == 'full_time_goals' else f'home_{stat}'
        elif col.startswith('target__away_team__'):
            stat = col.removeprefix('target__away_team__')
            mapping[col] = 'away_goals' if stat == 'full_time_goals' else f'away_{stat}'
        elif col.startswith(('home__', 'away__')):
            mapping[col] = col.replace('__', '_')
    return data.rename(columns=mapping)


def _extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """Extract high level features for modelling data.

    The expanding and rolling means run over the season frame with the fixtures rows already concatenated onto it, then
    shift by one, so an upcoming match carries the form of the matches before it and never its own outcome.
    """
    team_cols = ['home_team', 'away_team']
    target_cols = [
        col
        for col in data.columns
        if col.startswith('target')
        and col.endswith(('full_time_goals', 'shots', 'shots_on_target', 'corners', 'fouls_commited', 'cards'))
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
            ),
        )
    ]

    features_data = data[team_cols + target_cols].copy()
    home_features_data = features_data.drop(columns='away_team').rename(
        columns=_get_output_cols_mapping(True, features_data.columns),
    )
    away_features_data = features_data.drop(columns='home_team').rename(
        columns=_get_output_cols_mapping(False, features_data.columns),
    )
    features_data = pd.concat([home_features_data, away_features_data]).reset_index().set_index(['team', 'date'])
    features_data = features_data.sort_index()
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
    features_data['points'] = 3 * (features_data['goals_for'] > features_data['goals_against']) + (
        features_data['goals_for'] == features_data['goals_against']
    )
    features_data['adj_points'] = (
        3 * (features_data['adj_goals_for'] > features_data['adj_goals_against'] + DRAW_MARGIN)
        + 1.0 * (np.abs(features_data['adj_goals_for'] - features_data['adj_goals_against']) <= DRAW_MARGIN)
    ).astype(int)
    features_data = features_data[
        ['points', 'adj_points', 'goals_for', 'goals_against', 'adj_goals_for', 'adj_goals_against']
    ]

    features_cols = features_data.columns
    features_avg_cols = [f'{col}__avg' for col in features_cols]
    features_latest_avg_cols = [f'{col}__latest_avg' for col in features_cols]
    features_data[features_avg_cols] = features_data.groupby('team')[features_cols].expanding().mean().to_numpy()
    features_data[features_avg_cols] = features_data.groupby('team')[features_avg_cols].shift(1)
    features_data[features_latest_avg_cols] = (
        features_data.groupby('team')[features_cols].rolling(window=3, min_periods=1).mean().to_numpy()
    )
    features_data[features_latest_avg_cols] = features_data.groupby('team')[features_latest_avg_cols].shift(1)
    features_data = features_data.drop(columns=features_cols).reset_index()

    input_data = data[team_cols + odds_cols].copy()
    input_data = (
        input_data.reset_index()
        .merge(
            features_data.rename(
                columns={col: f'home__{col}' for col in features_data.columns if col.endswith('avg')},
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
                columns={col: f'away__{col}' for col in features_data.columns if col.endswith('avg')},
            ),
            left_on=['date', 'away_team'],
            right_on=['date', 'team'],
        )
        .drop(columns='team')
        .set_index('date')
    )

    output_data = data[target_cols].copy()
    data = pd.concat([input_data, output_data], axis=1)
    return _rename_modelling_columns(data)


def _to_snapshots(modelling: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explode a wide modelling frame into long `stats` and `odds` snapshots.

    A match only gets its in-play and post-play snapshots once its goals are known, so an upcoming fixture keeps just
    its pre-play snapshot.
    """
    modelling = modelling.reset_index() if modelling.index.name == 'date' else modelling.copy()
    feature_cols = [col for col in modelling.columns if col.endswith('avg')]
    odds_cols = [col for col in modelling.columns if col.count('__') == 1]

    preplay = modelling[SNAPSHOT_IDENTITY + feature_cols].assign(event_status='preplay', event_time=0)

    stats_frames = [preplay]
    if {'home_half_goals', 'away_half_goals'}.issubset(modelling.columns):
        half_mask = modelling['home_half_goals'].ge(0) & modelling['away_half_goals'].ge(0)
        inplay = modelling.loc[half_mask, SNAPSHOT_IDENTITY].assign(
            home_goals=modelling.loc[half_mask, 'home_half_goals'].astype(int),
            away_goals=modelling.loc[half_mask, 'away_half_goals'].astype(int),
            event_status='inplay',
            event_time=45,
        )
        outcomes = market_outcomes(inplay['home_goals'], inplay['away_goals'], MARKETS)
        inplay = pd.concat([inplay, outcomes], axis=1)
        stats_frames.append(inplay)

    full_mask = modelling['home_goals'].ge(0) & modelling['away_goals'].ge(0)
    postplay = modelling.loc[full_mask, SNAPSHOT_IDENTITY].assign(
        home_goals=modelling.loc[full_mask, 'home_goals'].astype(int),
        away_goals=modelling.loc[full_mask, 'away_goals'].astype(int),
        event_status='postplay',
        event_time=0,
    )
    outcomes = market_outcomes(postplay['home_goals'], postplay['away_goals'], MARKETS)
    postplay = pd.concat([postplay, outcomes], axis=1)
    stats_frames.append(postplay)

    stats = pd.concat(stats_frames, ignore_index=True)
    stats_order = [
        'event_status',
        'event_time',
        *SNAPSHOT_IDENTITY,
        'home_goals',
        'away_goals',
        *MARKETS,
        *feature_cols,
    ]
    stats = stats.reindex(columns=[col for col in stats_order if col in stats.columns])

    odds_frames = []
    for provider in PROVIDERS:
        mapping = {f'{provider}__{market}': market for market in MARKETS if f'{provider}__{market}' in odds_cols}
        if not mapping:
            continue
        provider_odds = modelling[SNAPSHOT_IDENTITY + list(mapping)].rename(columns=mapping)
        provider_odds = provider_odds.assign(provider=provider, event_status='preplay', event_time=0)
        odds_frames.append(provider_odds)
    odds = pd.concat(odds_frames, ignore_index=True)
    odds = odds.reindex(columns=['event_status', 'event_time', *SNAPSHOT_IDENTITY, 'provider', *MARKETS])
    return stats, odds


def _division(league: str, code: str) -> int:
    """Map a feed division code to the division number."""
    if 'C' in LEAGUES_MAPPING[league][1:]:
        return int(code) + 1 if code != 'C' else CHAMPIONSHIP_DIVISION
    return int(code)


def _current_year() -> int:
    """Return the current year, above which a season can still change upstream."""
    return datetime.now(tz=UTC).year


def _season_years(data: pd.DataFrame) -> pd.Series:
    """Return the ending year of each season of a whole-history file."""
    return data['Season'].apply(lambda season: season if not isinstance(season, str) else int(season.split('/')[-1]))


def _key_params(key: str) -> tuple[str, int, int | None]:
    """Return the league, division and year an item key encodes."""
    parts = key.split('_')
    year = int(parts[2]) if len(parts) == KEY_PARTS else None
    return parts[0], int(parts[1]), year


def _kickoff(data: pd.DataFrame, date_format: str) -> pd.Series:
    """Return the kick-off instant in UTC.

    The feed publishes every league's kick-off in UK time, whatever the country the match is played in, so it is
    converted here. A source resolves its own time zone at its own boundary and never emits a local or naive instant,
    which is what lets `date + event_time` address a moment of a match in wall-clock time.

    Older seasons carry no kick-off time. Their matches fall back to midnight, and they predate every time-stamped odds
    source, so they can never be joined to one.
    """
    date = pd.to_datetime(data['Date'], format=date_format, dayfirst=True)
    if 'Time' in data.columns:
        time = pd.to_timedelta(data['Time'].astype(str) + ':00', errors='coerce')
        date = date + time.fillna(pd.Timedelta(0))
    date = date.dt.tz_localize(FEED_TIMEZONE, ambiguous=True, nonexistent='shift_forward')
    return date.dt.tz_convert('UTC').dt.tz_localize(None)


def _process_training(payloads: list[RawPayload]) -> list[tuple[str, int, int, pd.DataFrame]]:
    """Process the raw season payloads into typed, per-season frames."""
    processed = []
    for payload in payloads:
        data = read_csv_content(payload.content)
        data = data.replace('^#', np.nan, regex=True).copy()
        data['Date'] = _kickoff(data, 'mixed')
        league, division, year = _key_params(payload.item.key)
        if year is not None:
            season = data.assign(league=league, division=division, year=year)
            processed.append((league, division, year, _convert_data_types(_preprocess_data(season))))
        else:
            data = data.assign(league=league, division=division)
            data['year'] = _season_years(data)
            for league_year in data['year'].unique():
                season = data[data['year'] == league_year]
                processed.append((league, division, int(league_year), _convert_data_types(_preprocess_data(season))))
    return processed


def _latest_years(catalogue: list[tuple[str, int, int, str]], processed: list) -> pd.DataFrame:
    """Return the latest published year of each league and division.

    It comes from the whole feed catalogue rather than the selection, so an upcoming match belongs to the season it is
    actually part of and never to an older one the user happened to select.
    """
    years = [{'league': league, 'division': division, 'year': year} for league, division, year, _ in catalogue]
    years.extend({'league': league, 'division': division, 'year': year} for league, division, year, _ in processed)
    return pd.DataFrame(years).groupby(['league', 'division']).max().reset_index()


def _process_fixtures(content: bytes, latest_years: pd.DataFrame) -> pd.DataFrame:
    """Process the raw fixtures payload into a typed frame of upcoming matches."""
    data = read_csv_content(content)
    data = data.rename(columns={'ï»¿Div': 'Div'}).copy()
    data['Date'] = _kickoff(data, '%d/%m/%Y')
    data = data.dropna(axis=0, how='any', subset=['Div', 'HomeTeam', 'AwayTeam'])
    leagues_mapping = {value[0]: key for key, value in LEAGUES_MAPPING.items()}
    data = data.assign(
        league=data['Div'].apply(lambda div: leagues_mapping[div[:-1]]),
        division=data['Div'].apply(lambda div: _division(leagues_mapping[div[:-1]], div[-1])),
    )
    data = data.merge(latest_years, how='left')
    data = _convert_data_types(_preprocess_data(data))
    today = pd.Timestamp(pd.to_datetime('today').date())
    return data.loc[data.index >= today, [col for col in data.columns if not col.startswith('target')]]


def _transform(
    processed: list[tuple[str, int, int, pd.DataFrame]],
    fixtures: pd.DataFrame,
) -> list[tuple[int, pd.DataFrame]]:
    """Build the wide modelling frame of each season, with its upcoming matches appended.

    The fixtures rows are concatenated before the features are extracted, so an upcoming match carries the form of the
    season that precedes it.
    """
    modelling = []
    for league, division, year, training in processed:
        mask = (
            (fixtures['league'] == league) & (fixtures['division'] == division) & (fixtures['year'] == year)
            if not fixtures.empty
            else pd.Series(dtype=bool)
        )
        data = pd.concat([training, fixtures[mask]]) if not fixtures.empty else training
        data = data.copy()
        features = _extract_features(data).assign(league=league, division=division, year=year)
        params_cols = ['league', 'division', 'year']
        features = features[params_cols + [col for col in features.columns if col not in params_cols]]
        half_time = data.reset_index()[['date', 'home_team', 'away_team', *HALF_TIME_COLS]].rename(
            columns=HALF_TIME_COLS,
        )
        features = features.reset_index().merge(half_time, on=['date', 'home_team', 'away_team'], how='left')
        modelling.append((training.shape[0], features))
    return modelling


def _modelling_snapshots(payloads: list[RawPayload], catalogue: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the feed transform and return the long `stats` and `odds` snapshots."""
    fixtures_payloads = [payload for payload in payloads if payload.item.key == FIXTURES_KEY]
    season_payloads = [payload for payload in payloads if payload.item.key != FIXTURES_KEY]
    processed = _process_training(season_payloads)
    fixtures = pd.DataFrame()
    if fixtures_payloads:
        fixtures = _process_fixtures(fixtures_payloads[0].content, _latest_years(catalogue, processed))
    modelling = _transform(processed, fixtures)

    train_stats, train_odds, upcoming_stats, upcoming_odds = [], [], [], []
    for n_train, data in modelling:
        stats, odds = _to_snapshots(data.iloc[:n_train])
        train_stats.append(stats)
        train_odds.append(odds)
        upcoming = data.iloc[n_train:]
        if len(upcoming):
            stats, odds = _to_snapshots(upcoming)
            upcoming_stats.append(stats)
            upcoming_odds.append(odds)

    def _combine(train: list[pd.DataFrame], upcoming: list[pd.DataFrame]) -> pd.DataFrame:
        frames = list(train)
        if upcoming:
            frames.append(pd.concat(upcoming, ignore_index=True).sort_values(['date', 'league', 'division']))
        non_empty = [frame for frame in frames if not frame.empty]
        return pd.concat(non_empty, ignore_index=True) if non_empty else frames[0].copy()

    return _combine(train_stats, upcoming_stats), _combine(train_odds, upcoming_odds)


def _year(season: str) -> int:
    """Map a two-digit feed season token to its year, pivoting the century at 68 as the feed does."""
    year = int(season[2:])
    return year + (2000 if year <= CENTURY_PIVOT else 1900)


MAIN_LEAGUES = [base_url.replace('.php', '')[:-1].capitalize() for base_url in BASE_URLS if base_url[0].islower()]


def _selected_leagues(selection: ParamGrid | None) -> set[str] | None:
    """Return the leagues a selection names, or `None` when it names none and so every one of them is wanted.

    A selection is one grid or a list of them, and a grid that names no league names all of them, so one such grid is
    enough for every league to be wanted. Answering with too few would leave a league that was asked for undownloaded,
    which is the one mistake here that is worse than downloading too much.
    """
    if not selection:
        return None
    grids = selection if isinstance(selection, list) else [selection]
    leagues: set[str] = set()
    for grid in grids:
        if 'league' not in grid:
            return None
        leagues.update(str(league) for league in grid['league'])
    return leagues or None


HISTORY_LEAGUES = [base_url.replace('.php', '') for base_url in BASE_URLS if not base_url[0].islower()]
INDEX_PREFIX = 'index'


def _index_url(league: str) -> str:
    """Return the index page of a league published one file per season."""
    return f'{URL}/{league.lower()}m.php'


def _history_url(league: str) -> str:
    """Return the whole-history file of a league published as a single file."""
    return f'{URL}/new/{LEAGUES_MAPPING[league][0]}.csv'


def _parse_index(league: str, content: bytes) -> list[tuple[str, int, int, str]]:
    """Parse an index page for the division, year and URL of every season it publishes.

    The upstream is the only authority on what it publishes, so no combination is fabricated from a cartesian product.
    """
    page = BeautifulSoup(content.decode(ENCODING), features='html.parser')
    hrefs = {
        element.get('href')
        for element in page.find_all('a')
        if element.get('href') and element.get('href').endswith('csv')
    }
    seasons = []
    for href in hrefs:
        *_, season, division = href.split('/')
        seasons.append((league, _division(league, division.replace('.csv', '')[-1]), _year(season), f'{URL}/{href}'))
    return seasons


class _FootballDataSource(BaseSource):
    """A source backed by the football-data.co.uk feed.

    The statistics and the odds come from the same upstream file, so both declare the same items and the file is
    downloaded once rather than twice.
    """

    sport: ClassVar[str | None] = 'soccer'
    name: ClassVar[str] = 'football_data'

    def __init__(self: Self) -> None:
        self._catalogue: list[tuple[str, int, int, str]] = []

    def index_items(self: Self, selection: ParamGrid | None = None) -> list[RawItem]:
        """Return the index pages and the whole-history files, from which the catalogue is read.

        A main league lists its seasons on an index page, which is small. A league published as one file of its whole
        history has no index, so its seasons can only be read out of the file, and reading it means downloading it. All
        sixteen of them together are twenty times the size of a season of the league that was actually asked for, and
        the index is re-read on every preparation, so they were paid for again every time.

        A league that was not selected is therefore not asked about. Discovery still asks about all of them, because
        nothing can be selected before it is known what exists.
        """
        leagues = _selected_leagues(selection)
        items = [
            RawItem(source=self.name, key=f'{INDEX_PREFIX}_{league}', url=_index_url(league), volatile=True)
            for league in MAIN_LEAGUES
            if leagues is None or league in leagues
        ]
        items.extend(
            RawItem(source=self.name, key=f'{league}_1', url=_history_url(league), volatile=True)
            for league in HISTORY_LEAGUES
            if leagues is None or league in leagues
        )
        return items

    def catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the combinations the feed publishes.

        The leagues published as a single whole-history file carry their seasons in a column, so their years are read
        from the file itself rather than guessed. The catalogue is kept, so the items and the snapshots are built
        against what the feed actually publishes.
        """
        catalogue: list[tuple[str, int, int, str]] = []
        for payload in payloads:
            key = payload.item.key
            if key.startswith(f'{INDEX_PREFIX}_'):
                catalogue.extend(_parse_index(key.removeprefix(f'{INDEX_PREFIX}_'), payload.content))
            else:
                league, _ = key.rsplit('_', 1)
                data = read_csv_content(payload.content)
                catalogue.extend(
                    (league, 1, int(year), _history_url(league)) for year in sorted(_season_years(data).unique())
                )
        self._catalogue = catalogue
        params = [{'league': league, 'division': division, 'year': year} for league, division, year, _ in catalogue]
        return sorted(params, key=lambda param: (param['league'], param['division'], param['year']))

    def required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return one item per selected season file, plus the fixtures of the feed.

        The URLs are looked up in the catalogue rather than constructed, so a combination the feed does not publish is
        never fabricated. A whole-history league yields the same item for each of its seasons, so it is downloaded once
        however many are selected.
        """
        seasons = {(league, division, year): url for league, division, year, url in self._catalogue}
        items: list[RawItem] = []
        for param in params:
            league, division, year = param['league'], param['division'], param['year']
            url = seasons.get((league, division, year))
            if url is None:
                continue
            if league in HISTORY_LEAGUES:
                item = RawItem(source=self.name, key=f'{league}_{division}', url=url, volatile=True)
            else:
                key = f'{league}_{division}_{year}'
                item = RawItem(source=self.name, key=key, url=url, volatile=year >= _current_year())
            if item not in items:
                items.append(item)
        items.append(RawItem(source=self.name, key=FIXTURES_KEY, url=f'{URL}/fixtures.csv', volatile=True))
        return items

    def to_snapshots(self: Self, payloads: list[RawPayload]) -> pd.DataFrame:
        """Transform the raw feed payloads into the long snapshots of this source."""
        stats, odds = _modelling_snapshots(payloads, self._catalogue)
        return stats if self.kind == 'stats' else odds


class FootballDataStats(_FootballDataSource, BaseStatsSource):
    """The statistics of the football-data.co.uk feed.

    It downloads the feed on your own machine and transforms it locally, so no data is redistributed. It needs no key.

    Read more in the [user guide][user-guide].

    Examples:
        >>> from sportsbet.dataloaders import DataLoader
        >>> from sportsbet.sources import FootballDataOdds, FootballDataStats
        >>> source = FootballDataStats()
        >>> source.name, source.kind, source.sport
        ('football_data', 'stats', 'soccer')
        >>> # It declares what it would read to learn what it publishes, and reads nothing.
        >>> [item.key for item in source.index_items({'league': ['Italy']})]
        ['index_Italy']
        >>> # Hand it to a dataloader, together with wherever the odds come from.
        >>> dataloader = DataLoader(
        ...     param_grid={'league': ['Italy'], 'division': [1], 'year': [2024]},
        ...     stats=source,
        ...     odds=FootballDataOdds(),
        ... )
        >>> dataloader.sport
        'soccer'
    """


class FootballDataOdds(_FootballDataSource, BaseOddsSource):
    """The odds of the football-data.co.uk feed.

    It carries the closing odds of the market average and the market maximum. They are pre-match prices, so an in-play
    bet cannot be backtested against them; a source with time-stamped prices is needed for that.

    Read more in the [user guide][user-guide].

    Examples:
        >>> from sportsbet.sources import FootballDataOdds
        >>> source = FootballDataOdds()
        >>> source.name, source.kind, source.sport
        ('football_data', 'odds', 'soccer')
        >>> # It reads the same upstream files as the statistics, so declaring the same items means fetching them once.
        >>> from sportsbet.sources import FootballDataStats
        >>> source.index_items({'league': ['Italy']}) == FootballDataStats().index_items({'league': ['Italy']})
        True
    """
