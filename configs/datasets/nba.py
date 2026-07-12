"""Example of a configuration whose source needs a key.

The key is read from the environment, so the file itself can be committed. Nothing the command line does could express
this before: it was handed a class, and a class carries no sources, so no source could carry a key.
"""

import os

from sportsbet.datasets import BasketballDataLoader, NBAStats, OddsApi

DATALOADER = BasketballDataLoader(
    param_grid={
        'league': ['NBA'],
        'year': [2026],
    },
    stats=NBAStats(),
    odds=OddsApi(key=os.environ['ODDS_API_KEY'], markets=['h2h']),
)
