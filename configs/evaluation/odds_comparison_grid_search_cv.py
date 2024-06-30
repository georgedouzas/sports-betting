"""Example of configuration file for a bettor based on odds comparison and a grid search of its hyperparameters."""

from sportsbet.datasets import SoccerDataLoader
from sportsbet.evaluation import BettorGridSearchCV, OddsComparisonBettor

# Data extraction
DATALOADER_CLASS = SoccerDataLoader
PARAM_GRID = {
    'year': [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
}
ODDS_TYPE = 'market_maximum'

# Betting process
BETTOR = BettorGridSearchCV(
    estimator=OddsComparisonBettor(),
    param_grid={
        'alpha': [0.03, 0.04, 0.05, 0.06, 0.07],
        'betting_markets': [
            ['home_win__full_time_goals', 'draw__full_time_goals'],
            ['home_win__full_time_goals', 'away_win__full_time_goals'],
            ['home_win__full_time_goals', 'over_2.5__full_time_goals'],
            ['home_win__full_time_goals', 'under_2.5__full_time_goals'],
            ['away_win__full_time_goals', 'draw__full_time_goals'],
            ['away_win__full_time_goals', 'over_2.5__full_time_goals'],
            ['away_win__full_time_goals', 'under_2.5__full_time_goals'],
            ['draw__full_time_goals', 'over_2.5__full_time_goals'],
            ['draw__full_time_goals', 'under_2.5__full_time_goals'],
            ['over_2.5__full_time_goals', 'under_2.5__full_time_goals'],
            ['home_win__full_time_goals', 'draw__full_time_goals', 'away_win__full_time_goals'],
            ['home_win__full_time_goals', 'draw__full_time_goals', 'over_2.5__full_time_goals'],
            ['home_win__full_time_goals', 'draw__full_time_goals', 'under_2.5__full_time_goals'],
            ['draw__full_time_goals', 'away_win__full_time_goals', 'over_2.5__full_time_goals'],
            ['draw__full_time_goals', 'away_win__full_time_goals', 'under_2.5__full_time_goals'],
            None,
        ],
    },
)
