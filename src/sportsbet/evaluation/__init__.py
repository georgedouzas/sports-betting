"""Evaluate the performance of predictive models."""

from ._base import (
    BaseBettor,
    derive_complementary_events,
    derive_market_base,
    find_latest_odds_column,
    load_bettor,
    save_bettor,
)
from ._classifier import ClassifierBettor
from ._factory import build_bettor
from ._model_selection import BettorGridSearchCV, backtest
from ._rules import OddsComparisonBettor

__all__: list[str] = [
    'BaseBettor',
    'BettorGridSearchCV',
    'ClassifierBettor',
    'OddsComparisonBettor',
    'backtest',
    'build_bettor',
    'derive_complementary_events',
    'derive_market_base',
    'find_latest_odds_column',
    'load_bettor',
    'save_bettor',
]
