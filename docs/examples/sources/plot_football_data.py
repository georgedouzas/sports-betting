"""Football-Data soccer feed =========================

This example illustrates [`FootballDataStats`][sportsbet.sources.FootballDataStats] and
[`FootballDataOdds`][sportsbet.sources.FootballDataOdds], the free soccer feed of football-data.co.uk.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt
import pandas as pd

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import FootballDataOdds, FootballDataStats

# %%
# Asking the source what exists
# -----------------------------
#
# A parameter grid cannot be written before it is known what exists, so the question goes to the source rather than to
# a dataloader.

stats = FootballDataStats()
params = stats.available_params()
len(params)

# %%
# The leagues it publishes:

sorted({param['league'] for param in params})

# %%
# It is free, and it is the only feed in the library that gives both statistics and odds for nothing. The odds are the
# closing prices offered before kick-off.

odds = FootballDataOdds()
odds.name, odds.kind, odds.sport

# %%
# Extracting the data
# -------------------
#
# Both sources read the same upstream files, so declaring the same items means they are downloaded once rather than
# twice.

dataloader = DataLoader(
    param_grid={'league': ['Spain'], 'division': [1], 'year': [2024]},
    stats=stats,
    odds=odds,
)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum')

# %%
# The input data:
X_train

# %%
# The available odds types are the providers the data carries:

dataloader.get_odds_types()

# %%
# A picture of it
# ---------------

coverage = pd.DataFrame(params).groupby('league').size().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(coverage.index, coverage.to_numpy())
ax.set_title('Seasons published by football-data.co.uk, per league')
ax.set_ylabel('seasons')
fig.autofmt_xdate(rotation=75)
