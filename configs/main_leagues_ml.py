"""Configuration file for bettor based on Gradient Boosting Classifier."""

from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sportsbet.datasets import SoccerDataLoader
from sportsbet.evaluation import ClassifierBettor

CONFIG = {
    'data': {
        'dataloader': SoccerDataLoader,
        'param_grid': {
            'league': [
                'England',
                'Scotland',
                'Germany',
                'Italy',
                'Spain',
                'France',
                'Netherlands',
                'Belgium',
                'Portugal',
                'Turkey',
                'Greece',
            ],
            'year': [2017, 2018, 2019, 2020, 2021, 2022, 2023],
        },
        'drop_na_thres': 0.8,
        'odds_type': 'market_maximum',
    },
    'betting': {
        'bettor': ClassifierBettor,
        'classifier': make_pipeline(
            make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']),
                remainder='passthrough',
            ),
            SimpleImputer(),
            MultiOutputClassifier(GradientBoostingClassifier(random_state=5)),
        ),
        'tscv': TimeSeriesSplit(5),
        'init_cash': 1000,
    },
}
