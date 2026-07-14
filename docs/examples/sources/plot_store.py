"""The store, and reconciling two feeds ====================================

This example illustrates [`LocalStore`][sportsbet.sources.LocalStore],
[`NotPreparedError`][sportsbet.sources.NotPreparedError], [`PreparationReport`][sportsbet.sources.PreparationReport],
[`resolve`][sportsbet.sources.resolve] and [`ReconciliationReport`][sportsbet.sources.ReconciliationReport].
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import tempfile

import matplotlib.pyplot as plt
import pandas as pd

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import (
    LocalStore,
    NotPreparedError,
    SampleSoccerOdds,
    SampleSoccerStats,
    UnmatchedError,
    resolve,
)

# %%
# The store is the only thing that fetches
# ----------------------------------------
#
# It keeps the raw downloads **forever**, since metered data cannot be obtained again without paying for it again,
# and everything derived from them is rebuilt at no cost. By default it lives in `~/.sportsbet`, or wherever
# `SPORTSBET_HOME` says.

store = LocalStore(tempfile.mkdtemp())
dataloader = DataLoader(
    param_grid={'league': ['England']},
    stats=SampleSoccerStats(),
    odds=SampleSoccerOdds(),
    store=store,
)

# %%
# Nothing is downloaded unless you ask
# ------------------------------------
#
# An extraction that was not given `download` refuses, and says what getting the data would take. It never fetches in
# order to tell you.

try:
    dataloader.extract_train_data(odds_type='market_average')
except NotPreparedError as error:
    refusal = str(error)
refusal

# %%
# Ask for it, and it is fetched once and kept.

X, Y, O = dataloader.extract_train_data(odds_type='market_average', download=True)
len(X)

# %%
# When two feeds name a club differently
# --------------------------------------
#
# This is the most dangerous thing in the library. One feed says `Man United`, another says `Manchester United`. If a
# name fails to match, that match simply has no odds — and **a missing odd does not look like an error**. It looks
# like a slightly smaller dataset, which produces a backtest that is clean, plausible and wrong.

identity = {'league': 'England', 'division': 1, 'year': 2025}
moment = {'event_status': 'preplay', 'event_time': pd.Timedelta(0)}
stats = pd.DataFrame(
    [
        {
            'date': pd.Timestamp('2025-08-16', tz='UTC'),
            **identity,
            **moment,
            'home_team': 'Man United',
            'away_team': 'Arsenal',
        },
    ],
)
odds = pd.DataFrame(
    [
        {
            'date': pd.Timestamp('2025-08-16', tz='UTC'),
            **identity,
            **moment,
            'home_team': 'Manchester United',
            'away_team': 'Arsenal',
            'provider': 'acme',
            'home_win': 1.8,
        },
    ],
)

paired, report = resolve(stats, odds)
paired[['home_team', 'away_team', 'home_win']]

# %%
# The odds carry the identity of the statistics, so `Manchester United` became `Man United`.

report.matched, report.unmatched_rate

# %%
# A club it cannot place is **raised**, not dropped. By default not one match may go without odds.

try:
    resolve(stats, odds.assign(home_team='Real Madrid'))
except UnmatchedError as error:
    complaint = str(error)
complaint

# %%
# It never applies a guess on its own: a wrong alias attaches one club's odds to another and says nothing about it. It
# tells you what it suspects, and you decide.

# %%
# A picture of it
# ---------------

fig, ax = plt.subplots()
ax.bar(['matched', 'without odds'], [report.matched, len(report.unmatched_stats)])
ax.set_title('Reconciling two feeds that name a club differently')
ax.set_ylabel('matches')
