"""
Test the data module.
"""

import numpy as np
import pandas as pd
import pytest

from sportsbet.soccer.data import (
    LEAGUES_MAPPING,
    combine_odds
)


# @pytest.mark.parametrize('y,label', [
#     (y_results, 'H'), 
#     (y_results, 'A'), 
#     (y_results, 'D'),
#     (y_results, 'O'), 
#     (y_results, 'U')
# ])
def test_combine_odds():
    """Test the generation of combined odds."""
    

    
