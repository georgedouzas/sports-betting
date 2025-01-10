"""Example of configuration file for a classifier-based bettor and a grid search of its hyperparameters."""

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from sportsbet.datasets import SoccerDataLoader
from sportsbet.evaluation import BettorGridSearchCV, ClassifierBettor

# Data extraction
DATALOADER_CLASS = SoccerDataLoader
PARAM_GRID={
    'league': ['Germany', 'Italy', 'France', 'England', 'Spain'],
    'year': [2019, 2020, 2021, 2022, 2023, 2024],
    'division': [1, 2],
}
ODDS_TYPE = 'market_maximum'

# Betting process
BETTOR = BettorGridSearchCV(
    estimator=ClassifierBettor(
        classifier=make_pipeline(
            make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']), remainder='passthrough',
            ),
            SimpleImputer(),
            MultiOutputClassifier(
                LogisticRegression(solver='liblinear', random_state=7),
            ),
        ),
        init_cash=10000.0,
        stake=50.0,
    ),
    param_grid={
        'classifier__multioutputclassifier__estimator__C': [0.01, 0.1, 1.0, 10.0, 50.0],
        'classifier__multioutputclassifier__estimator__class_weight': [None, 'balanced'],
        'betting_markets': [
            ['home_win__full_time_goals'],
            ['draw__full_time_goals'],
            ['home_win__full_time_goals', 'draw__full_time_goals'],
            ['away_win__full_time_goals', 'draw__full_time_goals'],
            None,
        ],
    },
)
CV = TimeSeriesSplit(3)
N_JOBS = -1
VERBOSE = 1
