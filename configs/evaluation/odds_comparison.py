"""Example of configuration file for odds comparison bettor."""

from sklearn.model_selection import TimeSeriesSplit

from sportsbet.datasets import SoccerDataLoader
from sportsbet.evaluation import OddsComparisonBettor

# Data extraction
DATALOADER_CLASS = SoccerDataLoader
PARAM_GRID = {
    'year': [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
}
ODDS_TYPE = 'market_maximum'

# Betting process
BETTOR = OddsComparisonBettor()
CV = TimeSeriesSplit(3)
