"""Example of configuration file to get the training data."""

from sportsbet.datasets import SoccerDataLoader

DATALOADER = SoccerDataLoader(
    param_grid={
        'league': ['England', 'Italy'],
        'year': [2023],
    },
)
DROP_NA_THRES = 0.8
ODDS_TYPE = 'market_average'
