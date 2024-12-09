"""It provides the tools to evaluate the performance of predictive models."""

from __future__ import annotations

from ._base import load_bettor, save_bettor
from ._classifier import ClassifierBettor
from ._model_selection import BettorGridSearchCV, backtest
from ._rules import OddsComparisonBettor

__all__: list[str] = [
    'BettorGridSearchCV',
    'ClassifierBettor',
    'OddsComparisonBettor',
    'backtest',
    'load_bettor',
    'save_bettor',
]
