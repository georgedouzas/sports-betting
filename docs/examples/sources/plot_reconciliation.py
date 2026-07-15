"""
Reconciling two feeds
=====================

This example illustrates [`resolve`][sportsbet.sources.resolve],
[`ReconciliationReport`][sportsbet.sources.ReconciliationReport] and
[`UnmatchedError`][sportsbet.sources.UnmatchedError]. They pair the odds of one feed to the matches of another when the
two name their clubs differently.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt
import pandas as pd

from sportsbet.sources import UnmatchedError, resolve

# %%
# The problem
# -----------
#
# This is the most dangerous thing in the library. One feed says `Man United`, another says `Manchester United`. If a
# name fails to match, that match simply has no odds, and a missing odd does not look like an error: it looks like a
# slightly smaller dataset, which produces a backtest that is clean, plausible and wrong. A dataloader reconciles the
# two feeds for you; this is the machinery underneath.

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
# The odds now carry the identity of the statistics, so `Manchester United` became `Man United`.

report.matched, report.unmatched_rate

# %%
# A club it cannot place
# ----------------------
#
# A name the pairing cannot place leaves its match without odds. By default not one match may go without odds, so it is
# raised rather than dropped.

try:
    resolve(stats, odds.assign(home_team='Real Madrid'))
except UnmatchedError as error:
    complaint = str(error)
complaint

# %%
# It never applies a guess on its own: a wrong alias attaches one club's odds to another and says nothing about it. It
# tells you what it suspects, and you decide, passing the corrections as `aliases`.

# %%
# What was reconciled
# -------------------

fig, ax = plt.subplots()
ax.bar(['matched', 'without odds'], [report.matched, len(report.unmatched_stats)])
ax.set_title('Reconciling two feeds that name a club differently')
ax.set_ylabel('matches')
fig.show()
