"""
Reconciling two feeds
=====================

This example illustrates resolve,
ReconciliationReport and
UnmatchedError. They pair the odds of one feed to the matches of another when the
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
# name fails to match, that match simply has no odds, and a missing odd does not look like an error, it looks like a
# slightly smaller dataset, which produces a backtest that is clean, plausible and wrong. A dataloader reconciles the
# two feeds for you; this is the machinery underneath.

identity = {'league': 'England', 'division': 1, 'year': 2025}
moment = {'event_status': 'preplay', 'event_time': pd.Timedelta(0)}


def match(home, away, **extra):
    """Build a one-row snapshot for a match."""
    fixed = {'date': pd.Timestamp('2025-08-16', tz='UTC'), **identity, **moment}
    return {**fixed, 'home_team': home, 'away_team': away, **extra}


stats = pd.DataFrame(
    [
        match('Man United', 'Arsenal'),
        match('Tottenham', 'Everton'),
        match('Newcastle', 'Wolves'),
        match('Brighton', 'Chelsea'),
    ],
)
odds = pd.DataFrame(
    [
        match('Manchester United', 'Arsenal', provider='acme', home_win=1.8),
        match('Spurs', 'Everton', provider='acme', home_win=2.1),
        match('Newcastle Utd', 'Wolves', provider='acme', home_win=1.9),
        match('Brighton', 'Chelsea', provider='acme', home_win=2.4),
    ],
)

# %%
# An exact join on the names would keep only `Brighton`, the one club both feeds spell the same way, and silently drop
# the other three:

exact = stats.merge(odds[['home_team', 'away_team']], on=['home_team', 'away_team'])
len(exact)

# %%
# What it matches on its own
# --------------------------
#
# `resolve` reads through the spelling. `Man United` and `Manchester United` are the same club, so are `Newcastle` and
# `Newcastle Utd`, and it pairs them without being told. By default not one match may go without odds, so a name it
# cannot place is raised rather than dropped.

try:
    resolve(stats, odds)
except UnmatchedError as error:
    complaint = str(error)
complaint

# %%
# It never guesses on its own. A wrong alias would attach one club's odds to another and say nothing about it. So it
# tells you what it suspects and leaves the decision to you. Allowing a little slack, here is how far it gets and what
# it flags:

paired, report = resolve(stats, odds, max_unmatched_rate=0.5)
report.matched, report.suggestions

# %%
# You confirm the suggestion by passing it as an alias, and now every match has its odds:

paired, report = resolve(stats, odds, aliases={'Spurs': 'Tottenham'})
report.matched, report.unmatched_rate

# %%
# The odds carry the identity of the statistics, so `Manchester United` is written as `Man United`:

paired[['home_team', 'away_team', 'home_win']]

# %%
# A picture of it
# ---------------
#
# The gap between the first bar and the last is the danger: an exact join throws away three quarters of the odds and
# calls it a dataset.

counts = {'exact name join': len(exact), 'resolved automatically': 3, 'with one alias': 4}

fig, ax = plt.subplots()
ax.bar(list(counts), list(counts.values()), color=['tab:red', 'tab:blue', 'tab:green'])
ax.set_title('Matches that keep their odds, four in all')
ax.set_ylabel('matches with odds')
