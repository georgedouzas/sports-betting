from collections import OrderedDict, Counter
from os.path import join
from itertools import product
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from category_encoders import OrdinalEncoder
from xgboost import XGBClassifier


def _calculate_profit(y_true, y_pred, odds, generate_weights):
        """Calculate mean profit."""
        correct_bets = (y_true == y_pred)
        if correct_bets.size == 0:
            return 0.0
        profit = correct_bets * (odds - 1)
        profit[profit == 0] = -1
        if generate_weights is not None:
            profit = np.average(profit, weights=generate_weights(odds))
        else:
            profit = profit.mean()
        return profit


class _ProfitScorer:
    """Return mean profit scorer."""

    PROBABILITIES_MAPPING = OrderedDict({
        'H': 'ProbHomeTeamFD',
        'D': 'ProbDrawFD',
        'A': 'ProbAwayTeamFD'
    })

    def __init__(self, predicted_result, generate_weights, odds_threshold):
        self.predicted_result = predicted_result
        self.generate_weights = generate_weights
        self.odds_threshold = odds_threshold
    
    def __call__(self, estimator, X, y_true):
        odds_index = X.columns.tolist().index(self.PROBABILITIES_MAPPING[self.predicted_result])
        odds = 1 / X.iloc[:, odds_index]
        y_pred = estimator.predict(X)
        mask = y_pred.astype(bool) & (odds > self.odds_threshold)
        y_true, y_pred, odds = y_true[mask], y_pred[mask], odds[mask]
        return _calculate_profit(y_true, y_pred, odds=odds, generate_weights=self.generate_weights)


class _Ratio:
    """Return dictionary for the ratio parameter of oversamplers."""

    def __init__(self, ratio):
        self.ratio = ratio
    
    def __call__(self, y):
        return {1: int(self.ratio * Counter(y)[0])}


# Parameters mappings
TEAMS_MAPPING = OrderedDict({
    'Borussia Monchengladbach': "M'gladbach",
    'Brighton and Hove Albion': 'Brighton',
    'Deportivo La Coruña': 'La Coruna',
    'Internazionale': 'Inter',
    'Chievo Verona': 'Chievo',
    'Wolverhampton': 'Wolves',
    'Cardiff City': 'Cardiff'
})
LEAGUES_MAPPING = OrderedDict({
    'Barclays Premier League': 'E0',
    'German Bundesliga': 'D1',
    'Italy Serie A': 'I1',
    'Spanish Primera Division': 'SP1'
})
SPI_FEATURES_MAPPING = OrderedDict({
    'date': 'Date',
    'league': 'League',
    'team1': 'HomeTeam',
    'team2': 'AwayTeam',
    'spi1': 'HomeTeamSPI',
    'spi2': 'AwayTeamSPI',
    'prob1': 'ProbHomeTeamSPI',
    'prob2': 'ProbAwayTeamSPI',
    'probtie': 'ProbDrawSPI'
})
FD_FEATURES_MAPPING = OrderedDict({
    'Div': 'League',
    'FTR': 'Target'
})
RESULTS_MAPPING = OrderedDict({
    'H': 0, 
    'D': 1, 
    'A': 2
})
SCORERS_MAPPING = OrderedDict({
    'H': {('profit_%s_%s' % (round(threshold, 2), 'none' if wts is None else 'exp')):_ProfitScorer('H', wts, threshold) for 
                             wts, threshold in product([None, np.exp], np.arange(1.0, 2.0, 0.05))},
    'D': {('profit_%s_%s' % (round(threshold, 2), 'none' if wts is None else 'exp')):_ProfitScorer('D', wts, threshold) for 
                             wts, threshold in product([None, np.exp], np.arange(3.0, 4.5, 0.05))},
    'A': {('profit_%s_%s' % (round(threshold, 2), 'none' if wts is None else 'exp')):_ProfitScorer('A', wts, threshold) for 
                             wts, threshold in product([None, np.exp], np.arange(1.5, 2.5, 0.05))}
})

# General parameters
MIN_N_MATCHES = 20
YEARS = ['1617', '1718', '1819']

# URLs
SPI_URL = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
FD_URL = 'http://www.football-data.co.uk/mmz4281'
FD_SUFFIX = [join(year, league) for year, league in product(YEARS, LEAGUES_MAPPING.values())]
FD_URLS = [join(FD_URL, suffix) for suffix in FD_SUFFIX]

# SPI data features
SPI_DATA_FEATURES = ['date', 'league', 'team1', 'team2', 'spi1', 'spi2', 'prob1', 'prob2', 'probtie']

# Football data features
FD_MAX_ODDS, FD_AVG_ODDS = 'BbMx', 'BbAv'
FD_MAX_ODDS_FEATURES = [FD_MAX_ODDS + result for result in RESULTS_MAPPING.keys()]
FD_AVG_ODDS_FEATURES = [FD_AVG_ODDS + result for result in RESULTS_MAPPING.keys()]
FD_KEYS_FEATURES = ['Div', 'Date', 'Season', 'HomeTeam', 'AwayTeam']
FD_DATA_FEATURES = FD_MAX_ODDS_FEATURES + FD_AVG_ODDS_FEATURES + FD_KEYS_FEATURES + ['FTR']

# Training data features
TRAIN_SPI_FEATURES = ['HomeTeamSPI', 'AwayTeamSPI']
TRAIN_PROB_SPI_FEATURES = ['ProbHomeTeamSPI', 'ProbDrawSPI', 'ProbAwayTeamSPI']
TRAIN_PROB_FD_FEATURES = ['ProbHomeTeamFD', 'ProbDrawFD', 'ProbAwayTeamFD']
TRAIN_KEYS_FEATURES = ['Day', 'Season', 'League', 'HomeTeam', 'AwayTeam', 'Target']
TRAIN_FEATURES = TRAIN_SPI_FEATURES + TRAIN_PROB_SPI_FEATURES + TRAIN_PROB_FD_FEATURES + TRAIN_KEYS_FEATURES

# Various features
KEYS_FEATURES = ['Date', 'League', 'HomeTeam', 'AwayTeam']
RESULTS_FEATURES = ['Days', 'Profit', 'Total profit', 'Precision', 'Bets precision',
                   'Bets precision ratio', 'Predictions ratio', 'Threshold', 'Hyperparameters']

# Modelling parameters
ODDS_THRESHOLDS = OrderedDict({
    'H': 1.0,
    'D': 3.75,
    'A': 1.85
})
GENERATE_WEIGHTS = np.exp
ESTIMATORS = OrderedDict({
    'H': make_pipeline(OrdinalEncoder(return_df=False), XGBClassifier()),
    'D': make_pipeline(OrdinalEncoder(return_df=False), SMOTE(), XGBClassifier()),
    'A': make_pipeline(OrdinalEncoder(return_df=False), SMOTE(), XGBClassifier())
})
PARAM_GRIDS = OrderedDict({
    'H': dict(xgbclassifier__max_depth=[3], xgbclassifier__n_estimators=[1000], xgbclassifier__learning_rate=[0.01], xgbclassifier__reg_lambda=[1.0]),
    'D': dict(smote__k_neighbors=[2], smote__ratio=[_Ratio(1.2)], xgbclassifier__max_depth=[3], xgbclassifier__n_estimators=[1000], xgbclassifier__learning_rate=[0.01], xgbclassifier__reg_lambda=[1.0]),
    'A': dict(smote__k_neighbors=[2], smote__ratio=['auto'], xgbclassifier__max_depth=[3], xgbclassifier__n_estimators=[1000], xgbclassifier__learning_rate=[0.02], xgbclassifier__reg_lambda=[1.0])
}) 
FIT_PARAMS = OrderedDict({
    'H': dict(test_size=0.05, xgbclassifier__eval_metric='map', xgbclassifier__early_stopping_rounds=10, xgbclassifier__verbose=False),
    'D': dict(test_size=0.05, xgbclassifier__eval_metric='map', xgbclassifier__early_stopping_rounds=10, xgbclassifier__verbose=False),
    'A': dict(test_size=0.05, xgbclassifier__eval_metric='map', xgbclassifier__early_stopping_rounds=10, xgbclassifier__verbose=False)
})
