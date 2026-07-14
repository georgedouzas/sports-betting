"""
A source of your own
====================

This example illustrates [`BaseStatsSource`][sportsbet.sources.BaseStatsSource],
[`BaseOddsSource`][sportsbet.sources.BaseOddsSource], [`RawItem`][sportsbet.sources.RawItem] and
[`RawPayload`][sportsbet.sources.RawPayload], by writing a feed for a league the library has never heard of.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import io

import matplotlib.pyplot as plt
import pandas as pd

from sportsbet.sources import BaseOddsSource, BaseStatsSource, RawItem, RawPayload, market_outcomes

IDENTITY = ['date', 'league', 'division', 'year', 'home_team', 'away_team']
MARKETS = ['home_win', 'draw', 'away_win']

# %%
# Four questions, and never a fetch
# ---------------------------------
#
# A source says what it needs read and how to turn it into snapshots. It **never fetches** — the store does that — which
# is what makes an extraction structurally incapable of downloading anything it was not told to.


class MyStats(BaseStatsSource):
    """Statistics from a feed of your own."""

    name = 'my_stats'
    sport = 'soccer'

    def index_items(self, selection=None):
        return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json', volatile=True)]

    def catalogue(self, payloads):
        return [{'league': 'Ruritania', 'division': 1, 'year': 2025}]

    def required_items(self, params, schedule=None):
        return [
            RawItem(
                source=self.name,
                key=f'Ruritania_1_{param["year"]}',
                url=f'https://example.com/{param["year"]}.csv',
            )
            for param in params
        ]

    def to_snapshots(self, payloads):
        games = pd.read_csv(io.BytesIO(payloads[0].content))
        games['date'] = pd.to_datetime(games['date'], utc=True)  # the kick-off, in UTC
        preplay = games[IDENTITY].assign(event_status='preplay', event_time=0, home_form=games['home_form'])
        postplay = games[IDENTITY].assign(event_status='postplay', event_time=0)
        outcomes = market_outcomes(games['home_goals'], games['away_goals'], MARKETS)
        return pd.concat([preplay, pd.concat([postplay, outcomes], axis=1)], ignore_index=True)


# %%
# It declares what it needs. The store is what goes and gets it.

source = MyStats()
source.required_items([{'year': 2025}])[0]

# %%
# Given the bytes that came back, it says what the snapshots are. Here we hand it a payload directly, which is exactly
# what the store would have handed it.

csv = b'date,league,division,year,home_team,away_team,home_form,home_goals,away_goals\n'
csv += b'2025-08-16,Ruritania,1,2025,A,B,0.5,2,1\n'
snapshots = source.to_snapshots([RawPayload(item=source.required_items([{'year': 2025}])[0], content=csv)])
snapshots

# %%
# The odds are a source too
# -------------------------
#
# The markets are its **columns** and the bookmaker is its `provider` column, so nothing has to be registered anywhere.
# Drop `draw` and you have a sport that cannot be drawn, and the bettor works the two-way market out on its own.


class MyOdds(BaseOddsSource):
    """Odds from a feed of your own."""

    name = 'my_odds'
    sport = 'soccer'

    def index_items(self, selection=None):
        return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json', volatile=True)]

    def catalogue(self, payloads):
        return [{'league': 'Ruritania', 'division': 1, 'year': 2025}]

    def required_items(self, params, schedule=None):
        return [
            RawItem(source=self.name, key=f'odds_{param["year"]}', url=f'https://example.com/odds/{param["year"]}.csv')
            for param in params
        ]

    def to_snapshots(self, payloads):
        odds = pd.read_csv(io.BytesIO(payloads[0].content))
        odds['date'] = pd.to_datetime(odds['date'], utc=True)
        return odds.assign(event_status='preplay', event_time=0)


# %%
csv = b'date,league,division,year,home_team,away_team,provider,home_win,draw,away_win\n'
csv += b'2025-08-16,Ruritania,1,2025,A,B,acme,1.8,3.4,4.2\n'
odds_source = MyOdds()
odds_source.to_snapshots([RawPayload(item=odds_source.required_items([{'year': 2025}])[0], content=csv)])

# %%
# Hand both to a dataloader and everything on the other pages applies to them:
#
# ```python
# from sportsbet.dataloaders import DataLoader
#
# dataloader = DataLoader(stats=MyStats(), odds=MyOdds())
# X, Y, O = dataloader.extract_train_data(odds_type='acme', download=True)
# ```
#
# Four rules that are not style:
#
# 1. **Never fetch.** If a source could fetch, an extraction could download by accident.
# 2. **`date` is the kick-off instant, in UTC.** Resolve your feed's time zone at your own boundary.
# 3. **A finished season is not `volatile`; a fixture is.** That is what makes a download incremental.
# 4. **Credentials go in `request_url`**, never in a `RawItem`. An item is written to the store; a key must not be.

# %%
# A picture of it
# ---------------

moments = snapshots.groupby('event_status').size()

fig, ax = plt.subplots()
ax.bar(moments.index, moments.to_numpy())
ax.set_title('Snapshots of my own feed, by moment')
ax.set_ylabel('snapshots')
