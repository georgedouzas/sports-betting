"""It provides the tools to evaluate the performance of predictive models."""

from __future__ import annotations

from ._classifier import ClassifierBettor
from ._rules import OddsComparisonBettor

__all__: list[str] = ['ClassifierBettor', 'OddsComparisonBettor']
