"""
Test the data module.
"""

import requests
import numpy as np
import pandas as pd
import pytest

from sportsbet.soccer.data import (
    LEAGUES_MAPPING,
    combine_odds
)

def test_combine_odds():
    """Test the generation of names mapping."""
    
