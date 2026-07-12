"""Example of configuration file to get the fixtures data."""

from sportsbet.datasets import SoccerDataLoader

DATALOADER = SoccerDataLoader(
    param_grid={
        'league': ['England', 'Italy', 'Greece'],
        'year': [2023],
    },
)
ODDS_TYPE = 'market_maximum'
