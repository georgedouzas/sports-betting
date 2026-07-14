"""
A dataloader of your own
========================

This example illustrates [`BaseDataLoader`][sportsbet.dataloaders.BaseDataLoader], for data that is already on your
machine and never came from a source.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt
import pandas as pd

from sportsbet.dataloaders import BaseDataLoader
from sportsbet.sources import market_outcomes

# %%
# One method to implement
# -----------------------
#
# A dataloader reads long snapshots and shapes them. Where they come from is the only thing it does not know, so that is
# the only thing you have to tell it: implement `_snapshots` and everything else follows.

MATCHES = [('2024-08-16', 'Arsenal', 'Chelsea', 2, 0), ('2024-08-23', 'Everton', 'Spurs', 1, 2)]
MARKETS = ['home_win', 'draw', 'away_win']


class MyDataLoader(BaseDataLoader):
    """A dataloader of snapshots I already hold."""

    def _snapshots(self):
        stats, odds = [], []
        for date, home, away, home_goals, away_goals in MATCHES:
            identity = {
                'date': date,
                'league': 'England',
                'division': 1,
                'year': 2025,
                'home_team': home,
                'away_team': away,
            }
            outcomes = market_outcomes(pd.Series([home_goals]), pd.Series([away_goals]), MARKETS).iloc[0]
            stats += [
                {**identity, 'event_status': 'preplay', 'event_time': pd.Timedelta('0min'), 'home_points_avg': 2.1},
                {**identity, 'event_status': 'postplay', 'event_time': pd.Timedelta('0min'), **outcomes},
            ]
            odds.append(
                dict(
                    **identity,
                    event_status='preplay',
                    event_time=pd.Timedelta('0min'),
                    provider='acme',
                    home_win=1.7,
                    draw=3.6,
                    away_win=4.8,
                ),
            )
        return pd.DataFrame(stats), pd.DataFrame(odds)


# %%
# Nothing is downloaded, because there is nothing to download.

dataloader = MyDataLoader()
dataloader.get_odds_types()

# %%
# The providers, the markets, the features and the moments are all derived from the data. Nothing was registered.

X, Y, O = dataloader.extract_train_data(odds_type='acme')
X

# %%
Y

# %%
O

# %%
# A picture of it
# ---------------

outcomes = Y.sum()
outcomes.index = outcomes.index.str.split('__').str[0]

fig, ax = plt.subplots()
ax.bar(outcomes.index, outcomes.to_numpy())
ax.set_title('Outcomes of my own two matches')
ax.set_ylabel('matches')
