from warnings import filterwarnings

from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearnext.over_sampling import GeometricSMOTE
from sportsbet.soccer import BettingAgent
from sportsbet.soccer.optimization import Ratio
from sportsbet.soccer.data import LEAGUES_MAPPING

filterwarnings('ignore')

# Backtesting
betting_agent = BettingAgent(leagues='all')
estimator = make_pipeline(
    MinMaxScaler(), 
    GeometricSMOTE(k_neighbors=2, ratio=Ratio(1.2), random_state=0), 
    LogisticRegression()
)
betting_agent.backtest(predicted_result='D', test_year=2, max_day_range=6, estimator=estimator)
betting_agent.calculate_statistics(odds_threshold=0.3, factor=2.5, credit_limit=10.0)