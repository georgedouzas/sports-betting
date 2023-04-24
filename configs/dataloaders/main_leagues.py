"""Configuration file for dataloader of main leagues."""

from sportsbet.datasets import SoccerDataLoader

MAIN = {
    'dataloader': SoccerDataLoader,
    'path': './data/main_leagues.pkl'
}
OPTIONAL = {
    'param_grid': {
       'league': ['England', 'Scotland', 'Germany', 'Italy', 'Spain', 'France', 'Netherlands', 'Belgium', 'Portugal', 'Turkey', 'Greece'],
        'year': [2017, 2018, 2019, 2020, 2021, 2022, 2023]
    },
    'drop_na_thres': 0.8,
    'odds_type': 'market_maximum'
}
