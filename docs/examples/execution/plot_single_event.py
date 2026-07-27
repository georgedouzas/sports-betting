"""
Watching one event
===================

This example runs the single-event execution unit end to end, offline. It watches one upcoming match, applies a fitted
model at the match's betting moment, and places the model's bet, once. Nothing here reaches a real bookmaker, opens a
real browser or moves real money: the data is made up, the browser is a stand-in, and time is an injected clock.

The unit runs a fixed sequence. It explores the candidate URLs to find the page for the event, logs in, monitors the
event to its moment while logging it, and at the moment applies the fitted bettor and places the configured stake on the
model's selection. A run is a no-stakes dry run unless it is armed with `live=True`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import asyncio

import matplotlib.pyplot as plt
import pandas as pd

from sportsbet.dataloaders import BaseDataLoader
from sportsbet.evaluation import OddsComparisonBettor
from sportsbet.execution import (
    FixedSession,
    PageSnapshot,
    PlacementReceipt,
    PlacementStatus,
    execute_event,
)
from sportsbet.sources import derive_market_outcomes

EVENT = 'Arsenal vs Chelsea'
STAKE = 10.0
URL = 'https://book.demo/arsenal-chelsea'

# %%
# The event and its data
# ----------------------
#
# The unit gets the event's evolving data from a dataloader, the same component the rest of the library uses. A
# dataloader normally pulls from a source, but a test or an example can carry its own snapshots by implementing
# `_load_snapshots`. The snapshots below are a handful of finished matches to train on and one upcoming
# `Arsenal vs Chelsea` with a kick-off a week away and generous preplay odds, so the model finds value on it.

MARKETS = ['home_win', 'draw', 'away_win', 'over_2.5', 'under_2.5']
FEATURES = ['home_points_avg', 'away_points_avg']
IDENTITY = ['date', 'league', 'division', 'year', 'home_team', 'away_team']
BASE_ODDS = {'home_win': 1.80, 'draw': 3.40, 'away_win': 4.20, 'over_2.5': 1.90, 'under_2.5': 1.95}
GENEROUS_ODDS = {'home_win': 2.35, 'draw': 3.70, 'away_win': 4.60, 'over_2.5': 2.05, 'under_2.5': 2.05}
PROVIDERS = {'market_average': 0.98, 'market_maximum': 1.06}
PLAYED = [
    ('2024-08-16', 'Man United', 'Fulham', (2.1, 1.2), (1, 0)),
    ('2024-08-24', 'Newcastle', 'Tottenham', (1.5, 1.8), (1, 2)),
    ('2024-09-01', 'Brighton', 'Everton', (1.3, 0.9), (0, 0)),
    ('2024-09-14', 'Chelsea', 'West Ham', (1.9, 1.4), (3, 1)),
    ('2024-10-05', 'Liverpool', 'Crystal Palace', (2.4, 1.1), (2, 1)),
]


def build_snapshots():
    """Build long stats/odds snapshots: the finished matches to train on and the one upcoming fixture."""
    kick_off = (pd.Timestamp.now(tz='UTC') + pd.Timedelta('7D')).strftime('%Y-%m-%d')
    matches = [*PLAYED, (kick_off, 'Arsenal', 'Chelsea', (2.6, 1.0), None)]
    stats_records = []
    odds_records = []
    for date, home, away, form, score in matches:
        identity = {
            'date': date,
            'league': 'England',
            'division': 1,
            'year': 2025,
            'home_team': home,
            'away_team': away,
        }
        snapshots = [('preplay', 0, 0, 0)] if score is None else [('preplay', 0, 0, 0), ('postplay', 0, *score)]
        for status, minutes, home_goals, away_goals in snapshots:
            outcomes = (
                derive_market_outcomes(pd.Series([home_goals]), pd.Series([away_goals]), MARKETS).iloc[0].to_dict()
            )
            features = dict(zip(FEATURES, form, strict=True)) if status == 'preplay' else dict.fromkeys(FEATURES)
            stats_records.append(
                {
                    **identity,
                    'event_status': status,
                    'event_time': pd.Timedelta(minutes=minutes),
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    **outcomes,
                    **features,
                },
            )
            if status != 'preplay':
                continue
            prices = GENEROUS_ODDS if score is None else BASE_ODDS
            for provider, factor in PROVIDERS.items():
                odds_records.append(
                    {
                        **identity,
                        'event_status': status,
                        'event_time': pd.Timedelta(minutes=minutes),
                        'provider': provider,
                        **{market: round(price * factor, 2) for market, price in prices.items()},
                    },
                )
    stats = pd.DataFrame(stats_records)[
        ['event_status', 'event_time', *IDENTITY, 'home_goals', 'away_goals', *MARKETS, *FEATURES]
    ]
    odds = pd.DataFrame(odds_records)[['event_status', 'event_time', *IDENTITY, 'provider', *MARKETS]]
    return stats, odds


class DemoDataLoader(BaseDataLoader):
    """A dataloader that carries its own snapshots, for the example.

    A real one comes from a source configured for the event's league and season.
    """

    def __init__(self, stats, odds):
        """Keep the snapshots the example built."""
        super().__init__()
        self._stats = stats
        self._odds = odds

    def _load_snapshots(self):
        """Return the snapshots the example built."""
        return self._stats, self._odds


stats, odds = build_snapshots()
dataloader = DemoDataLoader(stats, odds)
X, Y, O = dataloader.extract_train_data(odds_type='market_maximum')
X[['home_points_avg', 'away_points_avg']].tail()

# %%
# A fitted bettor
# ---------------
#
# The bettor decides whether to bet and which selection to back, never the stake. An
# `OddsComparisonBettor` backs a market when the price on offer beats its consensus probability. Fitted on the training
# data, it finds value on the home win of the upcoming match, at the generous price the fixture carries.

bettor = OddsComparisonBettor(alpha=0.02, betting_markets=['home_win', 'draw', 'away_win']).fit(X, Y, O)

X_fix, _, O_fix = dataloader.extract_fixtures_data()
value_bets = pd.DataFrame(bettor.bet(X_fix, O_fix), columns=list(bettor.betting_markets_))
value_bets.insert(0, 'match', (X_fix['home_team'] + ' vs ' + X_fix['away_team']).to_numpy())
value_bets

# %%
# A stand-in browser session
# --------------------------
#
# For a bookmaker with no API the venue is a headless browser session driving the site. The unit only needs a small set
# of methods from it: to navigate, log in, pin and read the site's controls, type, click and stop. The stand-in below
# duck-types those methods so the example drives no real browser. Its pages carry the event name and a ref for the
# control to act on, which is all the unit reads.


def first_ref(snapshot):
    """Return the first actionable ref in a page snapshot."""
    line = next(line for line in snapshot.yaml.splitlines() if '[ref=' in line)
    return line.split('[ref=')[1].split(']')[0]


class StubSession:
    """A browser session held in the example's hand, driving no real browser."""

    key = 'demo'

    def __init__(self):
        """Start pinned to the stake and confirm controls, as `execution page fix` would leave a session."""
        self.fixed_ = FixedSession(
            match='',
            url='demo://',
            locators={'stake': 'textbox[name="Stake"]', 'confirm': 'button[name="Place bet"]'},
        )
        self.typed = []
        self.clicked = []

    async def navigate(self, url):
        """Return a page that names the event and carries a ref to act on."""
        return PageSnapshot(yaml=f'- link "{EVENT}" [ref=e1]', url=url)

    async def authenticate(self):
        """Accept the login, since a profile that already holds one lands logged in."""

    def fix(self, match, locators):
        """Pin the site's controls for the match."""
        self.fixed_ = FixedSession(match=match, url='demo://', locators=dict(locators))
        return self.fixed_

    async def resolve(self, name):
        """Return what a pinned control points at now, with a ref to act on."""
        return PageSnapshot(yaml=f'- textbox "{name}" [ref=e2]', url='demo://')

    async def type(self, ref, text):
        """Record a fill and return the page it produced."""
        self.typed.append((ref, text))
        return PageSnapshot(yaml='typed', url='demo://')

    async def click(self, ref):
        """Record a click and return the page it produced."""
        self.clicked.append(ref)
        return PageSnapshot(yaml='clicked', url='demo://')

    async def stop(self):
        """Close nothing, since a stand-in opens nothing."""


# %%
# The placer drives the pinned controls to put one bet on. The default placer does exactly this; the one here also
# records the stake it put on, so the example can chart it.


class RecordingPlacer:
    """A placer that drives the pinned controls and records the stake it put on."""

    def __init__(self):
        """Start having staked nothing."""
        self.staked = 0.0

    async def __call__(self, intent, session):
        """Enter the stake into the pinned control, click confirm, and record what was staked."""
        stake_control = await session.resolve('stake')
        await session.type(first_ref(stake_control), str(intent.stake))
        confirm_control = await session.resolve('confirm')
        await session.click(first_ref(confirm_control))
        self.staked += intent.stake
        return PlacementReceipt(
            identity=intent.identity,
            status=PlacementStatus.MATCHED_FULL,
            stake=intent.stake,
            price=intent.min_price,
            value_bet=intent.value_bet,
            placed_at=pd.Timestamp.now(tz='UTC').to_pydatetime(),
        )


# %%
# Time, so the moment is now
# --------------------------
#
# The unit waits until the event's betting moment and only then acts. Waiting is real by default and driven by an
# injected clock in tests and examples. A preplay model's moment is the kick-off, so a clock pinned to the kick-off puts
# the moment at now: the monitor returns at once and the bet is decided immediately. The wait does nothing.

kick_off = X_fix.index[0]


def clock():
    """Return the kick-off, so the betting moment is now."""
    return kick_off


async def wait(seconds):
    """Wait for no time at all."""
    return


# %%
# A dry run
# ---------
#
# A run is a no-stakes dry run unless it is armed. The unit explores the URL, logs in, monitors to the moment, applies
# the bettor and logs the exact bet it would place, but stakes nothing. The receipt records the decision, and the money
# staked is nothing.

session = StubSession()
dry_run = asyncio.run(
    execute_event(EVENT, bettor, dataloader, session, stake=STAKE, urls=[URL], live=False, clock=clock, wait=wait),
)
dry_run[['match', 'market', 'selection', 'status', 'stake']]

# %%
# An armed run places one bet
# ---------------------------
#
# The same run with `live=True` and a placer arms the unit. At the moment it drives the pinned controls once and places
# the configured stake on the model's selection. It places at most one bet, whatever the price does afterward.

placer = RecordingPlacer()
armed_run = asyncio.run(
    execute_event(
        EVENT,
        bettor,
        dataloader,
        StubSession(),
        stake=STAKE,
        urls=[URL],
        live=True,
        placer=placer,
        clock=clock,
        wait=wait,
    ),
)
armed_run[['match', 'market', 'selection', 'status', 'stake', 'price']]

# %%
# What was staked, in one view. The dry run stakes nothing and the armed run stakes the configured stake, once.

staked = {
    'dry run': float(dry_run['stake'].sum()),
    'armed run': float(armed_run['stake'].sum()),
}

fig, ax = plt.subplots()
ax.bar(staked.keys(), staked.values(), color=['#4c72b0', '#c44e52'])
ax.set_title('A dry run stakes nothing; an armed run stakes once')
ax.set_ylabel('money staked')
