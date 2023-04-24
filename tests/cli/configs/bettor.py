"""Bettor configuration file for tests."""

from sklearn.model_selection import TimeSeriesSplit
from sportsbet.evaluation import OddsComparisonBettor

MAIN = {'bettor': OddsComparisonBettor, 'path': './bettor.pkl'}
OPTIONAL = {'alpha': 0.03, 'tscv': TimeSeriesSplit(2)}
