"""Test evaluation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing_extensions import Self

from sportsbet import Data
from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation._base import BaseBettor

X_train, Y_train, O_train = DummySoccerDataLoader().extract_train_data(odds_type='williamhill')


class TestBettor(BaseBettor):
    """Test bettor class."""

    __test__ = False

    def _predict_proba(self: Self, X: pd.DataFrame) -> Data:
        return np.repeat(0.6, X.shape[0] * len(self.feature_names_out_)).reshape(-1, len(self.feature_names_out_))

    def _fit(self: Self, X: pd.DataFrame, Y: pd.DataFrame, O: pd.DataFrame | None) -> Self:
        return self
