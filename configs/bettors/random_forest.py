"""Configuration file for bettor based on Random Forest classifier."""

from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sportsbet.evaluation import ClassifierBettor

MAIN = {
    'bettor': ClassifierBettor,
    'path': './data/random_forest.pkl',
}
OPTIONAL = {
    'classifier': make_pipeline(
        make_column_transformer(
            (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']),
            remainder='passthrough',
        ),
        SimpleImputer(),
        RandomForestClassifier(),
    ),
    'tscv': TimeSeriesSplit(8),
    'init_cash': 1000
}
