"""
Lumify
======

This example shows LumifyOdds, a keyed odds source for live and upcoming prices from
`Lumify <https://lumify.ai>`_.

Nothing here is bought and nothing is downloaded. The example shows how you configure the source and how it handles
your key.
"""

# Licence: MIT

import os

import matplotlib.pyplot as plt
import numpy as np

from sportsbet.sources import LumifyOdds, NBAStats, RawItem

# The source reads the key from the named variable. Set a placeholder, so the example runs without a real key and never
# writes yours into the docs.
os.environ['LUMIFY_API_KEY'] = 'your-key'

# %%
# Configuring it
# --------------
#
# Lumify prices the current slate. It does not sell historical closing lines the way The Odds API does. Use it for
# fixtures and the season in progress. Instant trial keys (no signup): https://lumify.ai/docs/ai

odds = LumifyOdds(key_env='LUMIFY_API_KEY', markets=['h2h'], bookmakers=['pinnacle'])
odds.name, odds.kind

# %%
# It sells several sports. So it carries no sport of its own and takes the sport of the statistics you pair it with.

{'carries no sport of its own': odds.sport is None}

# %%
# Your key never reaches the data
# -------------------------------
#
# The source adds the key to a request header at the moment it makes the request. The key is never part of a `RawItem`,
# so it is never written to disk.

item = RawItem(source='lumify', key='nba__-__2026__live', url='https://lumify.ai/v1/events?sport=nba')
{'key in the item': 'your-key' in item.url}

# %%
odds.request_headers(item)['Authorization']

# %%
# Using it
# --------
#
# Pair it with free statistics, here the NBA. Read your key from the environment rather than write it into a file that
# could be committed:
#
# ```python
# import os
#
# from sportsbet.dataloaders import DataLoader
# from sportsbet.sources import LumifyOdds, NBAStats
#
# dataloader = DataLoader(
#     param_grid={'league': ['NBA'], 'year': [2026]},
#     stats=NBAStats(),
#     odds=LumifyOdds(key_env='LUMIFY_API_KEY', markets=['h2h']),
# )
# X, Y, O = dataloader.extract_fixtures_data(odds_type='pinnacle')
# ```
#
# Extracting is what spends credits. List calls are cheap; each event with ``include_odds`` adds credits per Lumify's
# pricing. Prefer a single bookmaker (default ``pinnacle``) unless you need a multi-book view.

NBAStats().sport, LumifyOdds(key_env='LUMIFY_API_KEY').sport

# %%
# What a price implies
# --------------------

odds_range = np.linspace(1.05, 10, 200)

fig, ax = plt.subplots()
ax.plot(odds_range, 1 / odds_range)
ax.set_title('The probability a price implies')
ax.set_xlabel('decimal odds')
ax.set_ylabel('implied probability')
