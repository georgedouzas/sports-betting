"""
A source of your own
====================

This example shows BaseStatsSource, BaseOddsSource, RawItem and RawPayload. It writes a feed for a league the library
does not know.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import io

import matplotlib.pyplot as plt
import pandas as pd

from sportsbet.sources import BaseOddsSource, BaseStatsSource, RawItem, RawPayload, derive_market_outcomes

IDENTITY = ['date', 'league', 'division', 'year', 'home_team', 'away_team']
MARKETS = ['home_win', 'draw', 'away_win']

# %%
# A source describes a feed
# -------------------------
#
# A source says what to read and how to turn it into snapshots. It never fetches. The dataloader reads the items the
# source declares. So a source stays a plain description of a feed. It is easy to write and to test.


class MyStats(BaseStatsSource):
    """Statistics from a feed of your own."""

    name = 'my_stats'
    sport = 'soccer'

    def list_index_items(self, selection=None):
        return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json')]

    def read_catalogue(self, payloads):
        return [{'league': 'Ruritania', 'division': 1, 'year': 2025}]

    def list_required_items(self, params, schedule=None):
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
        outcomes = derive_market_outcomes(games['home_goals'], games['away_goals'], MARKETS)
        return pd.concat([preplay, pd.concat([postplay, outcomes], axis=1)], ignore_index=True)


# %%
# The source declares what it needs. The dataloader reads it.

source = MyStats()
source.list_required_items([{'year': 2025}])[0]

# %%
# The source turns the returned bytes into snapshots. Here we hand it a payload directly. The dataloader would hand it
# the same payload.

csv = b'date,league,division,year,home_team,away_team,home_form,home_goals,away_goals\n'
csv += b'2025-08-16,Ruritania,1,2025,A,B,0.5,2,1\n'
snapshots = source.to_snapshots([RawPayload(item=source.list_required_items([{'year': 2025}])[0], content=csv)])
snapshots

# %%
# The odds are a source too
# -------------------------
#
# The markets are its columns. The bookmaker is its `provider` column. So nothing has to be registered anywhere. Drop
# `draw` and you have a sport that cannot be drawn. The bettor works out the two-way market on its own.


class MyOdds(BaseOddsSource):
    """Odds from a feed of your own."""

    name = 'my_odds'
    sport = 'soccer'

    def list_index_items(self, selection=None):
        return [RawItem(source=self.name, key='seasons', url='https://example.com/seasons.json')]

    def read_catalogue(self, payloads):
        return [{'league': 'Ruritania', 'division': 1, 'year': 2025}]

    def list_required_items(self, params, schedule=None):
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
odds_source.to_snapshots([RawPayload(item=odds_source.list_required_items([{'year': 2025}])[0], content=csv)])

# %%
# Give both to a dataloader. Then everything on the other pages applies to them:
#
# ```python
# from sportsbet.dataloaders import DataLoader
#
# dataloader = DataLoader(stats=MyStats(), odds=MyOdds())
# X, Y, O = dataloader.extract_train_data(odds_type='acme')
# ```
#
# Four rules matter here:
#
# 1. Never fetch. A source that could fetch might download during an extraction by accident.
# 2. `date` is the kick-off instant in UTC. Convert your feed's time zone at your own boundary.
# 3. Fixtures come from `list_fixtures_items`. It defaults to the training items. Override it only when the upcoming
#    matches live in a different file than the finished seasons.
# 4. Credentials go in `request_url`, never in a `RawItem`. The transform sees the item, and you save the item.

# %%
# A picture of it
# ---------------

moments = snapshots.groupby('event_status').size()

fig, ax = plt.subplots()
ax.bar(moments.index, moments.to_numpy())
ax.set_title('Snapshots of my own feed, by moment')
ax.set_ylabel('snapshots')
