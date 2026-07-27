"""
Reconciling two feeds
=====================

This example shows resolve_odds. It pairs the odds of one feed to the matches of another when the two feeds name their
clubs differently.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import matplotlib.pyplot as plt
import pandas as pd

from sportsbet.sources import resolve_odds

# %%
# The problem
# -----------
#
# This is the most dangerous step in the library. One feed says `Man United`. Another says `Manchester United`. If a
# name fails to match, that match has no odds. A missing odd does not look like an error. It looks like a slightly
# smaller dataset. That gives you a backtest that is clean, plausible and wrong. A dataloader reconciles the two feeds
# for you. This is the machinery underneath.

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
# An exact join on the names keeps only `Brighton`, the one club both feeds spell the same way. It silently drops the
# other three:

exact = stats.merge(odds[['home_team', 'away_team']], on=['home_team', 'away_team'])
len(exact)

# %%
# What it matches on its own
# --------------------------
#
# `resolve_odds` reads past the spelling. `Man United` and `Manchester United` are the same club. So are `Newcastle`
# and `Newcastle Utd`. It pairs them without being told. It cannot place a name like `Spurs` for `Tottenham`, so it
# drops that match rather than guess. Attaching one club's odds to another would be worse than a missing row.

paired = resolve_odds(stats, odds)
sorted(paired['home_team'])

# %%
# That silent drop is the danger. Three of the four matches keep their odds and the fourth does not, and nothing says
# so. When you know which club a leftover name means, pass it as an alias. Now every match has its odds:

paired = resolve_odds(stats, odds, aliases={'Spurs': 'Tottenham'})
sorted(paired['home_team'])

# %%
# The odds take the identity of the statistics, so the result writes `Manchester United` as `Man United`:

paired[['home_team', 'away_team', 'home_win']]

# %%
# A picture of it
# ---------------
#
# The gap between the first bar and the last is the danger. An exact join throws away three quarters of the odds and
# calls the result a dataset.

counts = {'exact name join': len(exact), 'resolved automatically': 3, 'with one alias': 4}

fig, ax = plt.subplots()
ax.bar(list(counts), list(counts.values()), color=['tab:red', 'tab:blue', 'tab:green'])
ax.set_title('Matches that keep their odds, four in all')
ax.set_ylabel('matches with odds')
