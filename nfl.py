from pathlib import Path
from pprint import pprint

import requests
from bs4 import BeautifulSoup
import nflscraPy
import nfl_data_py as nfl
import pandas as pd
from sbrscrape import Scoreboard

DATA_PATH = Path(__file__).parent / 'data' / 'nfl'
COL_DROP = ['status',
            'tm_nano',
            'opp_nano',
            'old_game_id',
            'home_coach',
            'referee',
            'stadium_id',
            'stadium',
            'week_day',
            'weekday',
            'gametime',
            'roof',
            'surface',
            'roof',
            'surface',
            'temp',
            'wind',
            'away_qb_id',
            'home_qb_id',
            'away_qb_name',
            'home_qb_name',
            'away_coach',
            'location',
            'nfl_detail_id',
            'div_game',
            'ftn',
            'overtime',
            'pff',
            'gsis',
            'pfr',
            'espn',
            'away_rest',
            'home_rest'
            ]
ODDS_COLS = [
    'home_moneyline_odds_5Dimes',
    'home_moneyline_odds_sportbet',
    'home_moneyline_odds_bet365',
    'home_moneyline_odds_BETOnline',
    'home_moneyline_odds_SPORTS-BETTING',
    'home_moneyline_odds_BETOnline',
    'home_moneyline_odds_BOVADA',
    'home_moneyline_odds_bodog',
    'home_moneyline_odds_BOOKMAKER',
    'home_moneyline_odds_BetCRIS',
    'home_moneyline_odds_JAZZ',
    'home_moneyline_odds_JUSTBET',
    'home_moneyline_odds_MATCHBOOK',
    'home_moneyline_odds_PINNACLE',
    'home_moneyline_odds_intertops',
    'home_moneyline_odds_Sports-Interaction',
    'home_moneyline_odds_youwager.eu',
    'home_moneyline_odds_HERITAGE',
    'home_moneyline_odds_SBR',
    'away_moneyline_odds_5Dimes',
    'away_moneyline_odds_sportbet',
    'away_moneyline_odds_bet365',
    'away_moneyline_odds_BETOnline',
    'away_moneyline_odds_SPORTS-BETTING',
    'away_moneyline_odds_BETOnline',
    'away_moneyline_odds_BOVADA',
    'away_moneyline_odds_bodog',
    'away_moneyline_odds_BOOKMAKER',
    'away_moneyline_odds_BetCRIS',
    'away_moneyline_odds_JAZZ',
    'away_moneyline_odds_JUSTBET',
    'away_moneyline_odds_MATCHBOOK',
    'away_moneyline_odds_PINNACLE',
    'away_moneyline_odds_intertops',
    'away_moneyline_odds_Sports-Interaction',
    'away_moneyline_odds_youwager.eu',
    'away_moneyline_odds_HERITAGE',
    'away_moneyline_odds_SBR',
    'home_spread_odds_5Dimes',
    'home_spread_odds_sportbet',
    'home_spread_odds_bet365',
    'home_spread_odds_BETOnline',
    'home_spread_odds_SPORTS-BETTING',
    'home_spread_odds_BETOnline',
    'home_spread_odds_BOVADA',
    'home_spread_odds_bodog',
    'home_spread_odds_BOOKMAKER',
    'home_spread_odds_BetCRIS',
    'home_spread_odds_JAZZ',
    'home_spread_odds_JUSTBET',
    'home_spread_odds_MATCHBOOK',
    'home_spread_odds_PINNACLE',
    'home_spread_odds_intertops',
    'home_spread_odds_Sports-Interaction',
    'home_spread_odds_youwager.eu',
    'home_spread_odds_HERITAGE',
    'home_spread_odds_SBR',
    'away_spread_odds_5Dimes',
    'away_spread_odds_sportbet',
    'away_spread_odds_bet365',
    'away_spread_odds_BETOnline',
    'away_spread_odds_SPORTS-BETTING',
    'away_spread_odds_BETOnline',
    'away_spread_odds_BOVADA',
    'away_spread_odds_bodog',
    'away_spread_odds_BOOKMAKER',
    'away_spread_odds_BetCRIS',
    'away_spread_odds_JAZZ',
    'away_spread_odds_JUSTBET',
    'away_spread_odds_MATCHBOOK',
    'away_spread_odds_PINNACLE',
    'away_spread_odds_intertops',
    'away_spread_odds_Sports-Interaction',
    'away_spread_odds_youwager.eu',
    'away_spread_odds_HERITAGE',
    'away_spread_odds_SBR',
    'total_under_odds_5Dimes',
    'total_under_odds_sportbet',
    'total_under_odds_bet365',
    'total_under_odds_BETOnline',
    'total_under_odds_SPORTS-BETTING',
    'total_under_odds_BETOnline',
    'total_under_odds_BOVADA',
    'total_under_odds_bodog',
    'total_under_odds_BOOKMAKER',
    'total_under_odds_BetCRIS',
    'total_under_odds_JAZZ',
    'total_under_odds_JUSTBET',
    'total_under_odds_MATCHBOOK',
    'total_under_odds_PINNACLE',
    'total_under_odds_intertops',
    'total_under_odds_Sports-Interaction',
    'total_under_odds_youwager.eu',
    'total_under_odds_HERITAGE',
    'total_under_odds_SBR',
    'total_over_odds_5Dimes',
    'total_over_odds_sportbet',
    'total_over_odds_bet365',
    'total_over_odds_BETOnline',
    'total_over_odds_SPORTS-BETTING',
    'total_over_odds_BETOnline',
    'total_over_odds_BOVADA',
    'total_over_odds_bodog',
    'total_over_odds_BOOKMAKER',
    'total_over_odds_BetCRIS',
    'total_over_odds_JAZZ',
    'total_over_odds_JUSTBET',
    'total_over_odds_MATCHBOOK',
    'total_over_odds_PINNACLE',
    'total_over_odds_intertops',
    'total_over_odds_Sports-Interaction',
    'total_over_odds_youwager.eu',
    'total_over_odds_HERITAGE',
    'total_over_odds_SBR'
]

YEARS = list(range(2000, 2025))


def save_proccessed_data_to_csv(data: pd.DataFrame, year: int, src: str) -> pd.DataFrame:
    (DATA_PATH / 'processed').mkdir(parents=True, exist_ok=True)
    # print(data.columns)
    try:
        clean_date = data.drop(columns=COL_DROP, errors='ignore')
        clean_date.to_csv(DATA_PATH / 'processed' / f'{src}_nfl_season_{year}.csv', index=False)
    except PermissionError as e:
        print(f"Error saving data: {e}")
        # Handle the error, like suggesting a different location
    return clean_date


def get_historical_nfl_data(year: int) -> list[pd.DataFrame]:
    nlf_games_win_totals: pd.DataFrame = save_proccessed_data_to_csv(data=nfl.import_win_totals([year]),
                                                                     year=year, src='nfl_data_win_totals')

    nlf_games_sc_lines: pd.DataFrame = save_proccessed_data_to_csv(data=nfl.import_schedules([year]),
                                                                   year=year, src='nfl_data_sc_lines')

    return [nlf_games_win_totals, nlf_games_sc_lines]


def create_modelling_data(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Creates a DataFrame containing betting odds for a game based on team abbreviations.

    Args:
        dfs: A list of DataFrames. The first DataFrame (dfs[0]) should contain
            a column named 'abbr' with team abbreviations, and the second DataFrame
            (dfs[1]) should contain columns named 'home_team' and 'away_team'.

        dfs[0] = odds_book data
        dfs[1] = nfl week stats data

    Returns:
        None. The function modifies the `df_betting_odds` DataFrame in-place.
    """

    df_nfl_odds = dfs[1].copy()

    for df_odds_index in range(len(dfs[1])):  # Iterate over each row in dfs[1] (assuming game data)

        if 'game_id' in dfs[1].columns:
            # Proceed with extracting game_id if the column exists
            dfs_nfl_data_game_id = dfs[1]['game_id'].iloc[df_odds_index]
        else:
            # Handle the case where 'game_id' is missing (e.g., print a message)
            print("Error: 'game_id' column not found in dfs[1]")

        home_team = dfs[1]['home_team'].iloc[df_odds_index]
        away_team = dfs[1]['away_team'].iloc[df_odds_index]

        odds_data = dfs[0].query('game_id == @dfs_nfl_data_game_id')

        for i in range(len(odds_data)):
            odds_game_id = odds_data['game_id'].iloc[i]
            abbr = odds_data['abbr'].iloc[i]
            market_type = odds_data['market_type'].iloc[i]
            book = odds_data['book'].iloc[i]
            odds = odds_data['odds'].iloc[i]

            odds_as_float = float(odds)

            if (abbr == home_team or
                    # different abbreviation in the data for JAX/JAC
                    (abbr == 'JAC' and home_team == 'JAX') or
                    # San Diego chargers moved to LA
                    (abbr == 'LAC' and home_team == 'SD') or
                    # St. Louis Rams moved to LA
                    (abbr == 'LA' and home_team == 'STL') or
                    # Oakland Raiders moved to Las Vegas
                    (abbr == 'OAK' and home_team == 'LV')):

                # if f'home_team_{market_type}_{book}' not in df_nfl_odds.columns:
                #     df_nfl_odds[f'home_team_{market_type}_{book}'] = float
                df_nfl_odds.loc[df_odds_index, f'home_team_{market_type}_{book}'] = odds_as_float

                # Get a copy of the row data
                # row_data = df_nfl_odds.iloc[df_odds_index].to_dict()

                # Add the new column and value to the row data dictionary
                # row_data[f'home_team_{market_type}_{book}'] = odds_as_float

                # Update the row in the DataFrame using the modified row data
                # df_nfl_odds.iloc[df_odds_index] = row_data
            elif (abbr == away_team or
                  # different abbreviation in the data for JAX/JAC
                  (abbr == 'JAC' and away_team == 'JAX') or
                  # San Diego chargers moved to LA
                  (abbr == 'LAC' and away_team == 'SD') or
                  # St. Louis Rams moved to LA
                  (abbr == 'LA' and away_team == 'STL') or
                  # Oakland Raiders moved to Las Vegas
                  (abbr == 'OAK' and away_team == 'LV')):

                # if f'away_team_{market_type}_{book}' not in df_nfl_odds.columns:
                #     df_nfl_odds[f'home_team_{market_type}_{book}'] = float

                df_nfl_odds.loc[df_odds_index, f'away_team_{market_type}_{book}'] = odds_as_float

                # Get a copy of the row data
                # row_data = df_nfl_odds.iloc[df_odds_index].to_dict()

                # Add the new column and value to the row data dictionary
                # row_data[f'away_team_{market_type}_{book}'] = odds_as_float

                # Update the row in the DataFrame using the modified row data
                # df_nfl_odds.iloc[df_odds_index] = row_data
            else:
                if odds_game_id == dfs_nfl_data_game_id and (abbr == 'under' or abbr == 'over'):
                    # if f'{abbr}_{market_type}_{book}' not in df_nfl_odds.columns:
                    #     df_nfl_odds[f'home_team_{market_type}_{book}'] = float

                    df_nfl_odds.loc[df_odds_index, f'{abbr}_{market_type}_{book}'] = odds_as_float

                    # Get a copy of the row data
                    # row_data = df_nfl_odds.iloc[df_odds_index].to_dict()

                    # Add the new column and value to the row data dictionary
                    # row_data[f'{abbr}_{market_type}_{book}'] = odds_as_float

                    # Update the row in the DataFrame using the modified row data
                    # df_nfl_odds.iloc[df_odds_index] = row_data
                else:
                    print(f'NFL spread data has wrong abbreviation: {abbr}\n '
                          f'and/or nfl data has incorrect home team: {home_team} or away team: {away_team}')

    return df_nfl_odds


def save_modelling_data_to_csv(data: pd.DataFrame, year: int, src: str) -> pd.DataFrame:
    (DATA_PATH / 'modelling').mkdir(parents=True, exist_ok=True)
    try:
        data.to_csv(DATA_PATH / 'modelling' / f'{src}_nfl_season_{year}.csv', index=False)
    except PermissionError as e:
        print(f"Error saving data: {e}")
        # Handle the error, like suggesting a different location
    return data


def update_data():
    for year in YEARS:
        dfs: list[pd.DataFrame] = get_historical_nfl_data(year=year)
        df_odds = create_modelling_data(dfs=dfs)
        save_modelling_data_to_csv(data=df_odds, year=year, src='odds')


update_data()
