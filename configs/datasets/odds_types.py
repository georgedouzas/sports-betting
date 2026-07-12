"""Example of configuration file to get the odds types."""

from sportsbet.datasets import SoccerDataLoader

DATALOADER = SoccerDataLoader(
    param_grid={
        'league': ['England'],
        'year': [2019, 2022],
    },
)
