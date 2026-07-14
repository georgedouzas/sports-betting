"""Test evaluation module."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from sportsbet import Data
from sportsbet.dataloaders import DataLoader
from sportsbet.evaluation._base import BaseBettor
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

_loader = DataLoader(stats=SampleSoccerStats(), odds=SampleSoccerOdds())
X_train, Y_train, O_train = _loader.extract_train_data(odds_type='market_average', download=True)
X_fix, _, O_fix = _loader.extract_fixtures_data()


class TestBettor(BaseBettor):
    """Test bettor class."""

    __test__ = False

    def _predict_proba(self: Self, X: pd.DataFrame) -> Data:
        return np.repeat(0.6, X.shape[0] * len(self.feature_names_out_)).reshape(-1, len(self.feature_names_out_))

    def _fit(self: Self, X: pd.DataFrame, Y: pd.DataFrame, O: pd.DataFrame | None) -> Self:
        return self
