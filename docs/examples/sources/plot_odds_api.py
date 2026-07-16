"""
The Odds API
============

This example illustrates OddsApi, the paid odds source that carries time-stamped prices.

Nothing here is bought and nothing is downloaded: the example shows how the source is configured and how your key is
handled, which is what you want to understand before spending anything.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt
import numpy as np

from sportsbet.sources import NBAStats, OddsApi, RawItem

# %%
# Configuring it
# --------------
#
# It carries prices with a timestamp, so an in-play bet can be backtested against the odds that were actually on offer
# at the minute it would have been placed. The free feeds cannot do that: they publish the closing price only.
#
# Every market, every region and every moment is a separate request, so each one multiplies the work.

odds = OddsApi(key='your-key', markets=['h2h'], regions=['eu'])
odds.name, odds.kind

# %%
# It sells every sport, so it carries none of its own and takes the sport of the statistics it is paired with.

{'carries no sport of its own': odds.sport is None}

# %%
# Your key never reaches the data
# -------------------------------
#
# The key is added to a request at the moment the request is made. It is never part of a `RawItem`, so it is never
# written to disk.

item = RawItem(source='odds_api', key='snapshot', url='https://api.the-odds-api.com/v4/sports?all=true')
{'key in the item': 'your-key' in item.url}

# %%
odds.request_url(item)

# %%
# Using it
# --------
#
# Pair it with free statistics, here the NBA, and read your key from the environment rather than writing it into a
# file that could be committed:
#
# ```python
# import os
#
# from sportsbet.dataloaders import DataLoader
# from sportsbet.sources import NBAStats, OddsApi
#
# dataloader = DataLoader(
#     param_grid={'league': ['NBA'], 'year': [2026]},
#     stats=NBAStats(),
#     odds=OddsApi(key=os.environ['ODDS_API_KEY'], markets=['h2h']),
# )
# X, Y, O = dataloader.extract_train_data(odds_type='pinnacle')
# ```
#
# Extracting is what spends: every market, region and moment is its own request, and what those requests cost is
# between you and the vendor. Ask the source what it would fetch first, and price it before you commit.

NBAStats().sport, OddsApi(key='your-key').sport

# %%
# What a price is saying
# ----------------------

odds_range = np.linspace(1.05, 10, 200)

fig, ax = plt.subplots()
ax.plot(odds_range, 1 / odds_range)
ax.set_title('The probability a price implies')
ax.set_xlabel('decimal odds')
ax.set_ylabel('implied probability')
