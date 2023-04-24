"""Dataloader configuration file for tests."""

from sportsbet.datasets import DummySoccerDataLoader

MAIN = {'dataloader': DummySoccerDataLoader, 'path': './dataloader.pkl'}
OPTIONAL = {
    'param_grid': {
        'league': ['England', 'Greece'],
    },
    'drop_na_thres': 0.8,
    'odds_type': 'interwetten',
}
