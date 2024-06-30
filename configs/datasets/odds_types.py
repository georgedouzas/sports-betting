"""Example of configuration file to get the odds types."""

from sportsbet.datasets import SoccerDataLoader

DATALOADER_CLASS = SoccerDataLoader
PARAM_GRID = {
    'league': ['England'],
    'year': [2019, 2022],
}
