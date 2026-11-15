# Execution

A bettor finds value bets. Execution places one of them, at a venue where you hold an account. You give it a fitted
bettor, one upcoming match and a stake. It waits for the match, and a few minutes before kickoff it applies the bettor
to fresh data and places the bet once. This spends real money, so read the risks first.

## Risks

* Driving a bookmaker's website breaks the terms of service of almost every bookmaker and puts the account and its
  balance at risk.
* It places one preplay bet on one match. Running many matches is your own work, and a bookmaker may block or close an
  account for automating that.
* It does not evade a website's automation controls. A site that blocks automation is reported and the run stops.

## Installation

```bash
pip install 'sports-betting[execution]'
python -m playwright install chromium
```

Playwright is needed only for a bookmaker's website. A venue with an API needs nothing more.

## What a run does

A run acts on exactly one match, named as `Home vs Away`, which must be among the dataloader's fixtures. It goes
through a fixed sequence and is not interactive once it starts.

1. It checks the venue once, by logging in, and releases it again.
2. It waits for the betting moment. On each poll it reads the match's data from the dataloader again and logs the
   match, the time left and the source's prices. It stops when the match leaves the fixtures, for example because it
   was postponed.
3. At the moment it reads the data once more and applies the bettor to it. When the bettor backs several markets, the
   first in its market order is taken.
4. It reads the venue's price for the selection, then records a dry run or places the bet once.

The bettor decides whether to bet and which side to back. `home_win` backs the home team, `away_win` the away team,
`draw` backs `Draw`, and an over or under market backs its line, as in `Over 2.5`. The bettor never decides the stake,
which is a parameter of the run.

## The betting moment

Most bookmakers suspend a preplay market at kickoff, so the bet goes on a lead time before it. The lead time defaults
to five minutes, and [`find_betting_moment`][sportsbet.execution.find_betting_moment] returns the moment for a kickoff.
A run supports preplay models only. A dataloader fitted with `target_event_status='inplay'` is refused before any venue
is contacted. A run that starts after its moment places nothing.

## The price

The lowest price the run accepts is the source's price the model saw value at. The venue's price is read before
anything is staked, and a venue that offers less is refused with a `refused_price` receipt. The run never falls back to
a made-up price, so some value bets are not placed. That is the intended trade.

## Dry run and armed

A run is a dry run unless you arm it. A dry run does everything except stake and confirm. It reads the venue's real
price for the selection, so it proves the whole path to the venue, and records the bet it would place. You arm a run
with `live=True`, or `--live` on the command line. Once armed it places at the moment with no second confirmation,
after logging the match, market, selection, stake, venue and minimum price.

## A run in full

The example below builds everything inline, so it runs offline. Two small sources carry a season of four games whose
last game is tomorrow, the venue is a minimal API venue, and an injected clock and wait stand in for the twelve minutes
before the moment, so the run finishes at once.

```python exec="1" source="block" result="text"
import asyncio
import tempfile
from pathlib import Path

import pandas as pd
from sportsbet.dataloaders import DataLoader
from sportsbet.evaluation import OddsComparisonBettor
from sportsbet.execution import BaseVenue, PlacementReceipt, PlacementStatus, execute_event
from sportsbet.sources import BaseOddsSource, BaseStatsSource, derive_market_outcomes

MARKETS = ['home_win', 'draw', 'away_win']
IDENTITY = {'league': 'Ruritania', 'division': 1, 'year': 2027}
TOMORROW = (pd.Timestamp.now(tz='UTC') + pd.Timedelta('1D')).floor('h').strftime('%Y-%m-%d %H:%M')
GAMES = [
    ('2026-08-22 14:00', 'Arsenal', 'Chelsea', 2, 1, (2.1, 3.4, 3.6), (2.2, 3.6, 3.8)),
    ('2026-08-29 14:00', 'Chelsea', 'Everton', 1, 1, (1.8, 3.6, 4.5), (1.9, 3.8, 4.8)),
    ('2026-09-05 14:00', 'Everton', 'Arsenal', 0, 2, (4.2, 3.6, 1.9), (4.4, 3.8, 2.0)),
    (TOMORROW, 'Arsenal', 'Everton', -1, -1, (1.7, 3.8, 5.0), (1.9, 4.0, 5.4)),
]


def identify(date, home, away):
    return {'date': pd.Timestamp(date, tz='UTC'), **IDENTITY, 'home_team': home, 'away_team': away}


class Games(BaseStatsSource):
    """A season written inline, its last game tomorrow."""

    name = 'games'

    def list_index_items(self, selection=None):
        return []

    def read_catalogue(self, payloads):
        return [IDENTITY]

    def list_required_items(self, params, schedule=None):
        return []

    def to_snapshots(self, payloads):
        rows = []
        for date, home, away, home_goals, away_goals, _, _ in GAMES:
            game = identify(date, home, away)
            rows.append({**game, 'event_status': 'preplay', 'event_time': 0})
            if home_goals >= 0:
                outcomes = derive_market_outcomes(pd.Series([home_goals]), pd.Series([away_goals]), MARKETS)
                result = {'home_goals': home_goals, 'away_goals': away_goals, **outcomes.iloc[0].to_dict()}
                rows.append({**game, 'event_status': 'postplay', 'event_time': 0, **result})
        return pd.DataFrame(rows)


class Prices(BaseOddsSource):
    """The average and the best price of each game."""

    name = 'prices'

    def list_index_items(self, selection=None):
        return []

    def read_catalogue(self, payloads):
        return [IDENTITY]

    def list_required_items(self, params, schedule=None):
        return []

    def to_snapshots(self, payloads):
        rows = []
        for date, home, away, _, _, average, best in GAMES:
            for provider, prices in (('market_average', average), ('market_maximum', best)):
                quote = {'event_status': 'preplay', 'event_time': 0, 'provider': provider}
                rows.append({**identify(date, home, away), **quote, **dict(zip(MARKETS, prices))})
        return pd.DataFrame(rows)


class Exchange(BaseVenue):
    """A venue with an API, reduced to what a run needs."""

    key = 'exchange'

    async def authenticate(self):
        return None

    async def read_price(self, identity):
        book = {('Arsenal vs Everton', 'home_win', 'Arsenal'): 1.95}
        return book.get((identity.match, identity.market, identity.selection))

    async def place(self, intent):
        return PlacementReceipt(intent.identity, PlacementStatus.REJECTED, detail='A dry run never places.')


dataloader = DataLoader(stats=Games(), odds=Prices())
X, Y, O = dataloader.extract_train_data(odds_type='market_maximum')
bettor = OddsComparisonBettor(alpha=0.03).fit(X, Y, O)

X_fix, _, _ = dataloader.extract_fixtures_data()
now = {'time': X_fix.index[0] - pd.Timedelta('17min')}


async def wait(seconds):
    now['time'] += pd.Timedelta(seconds=seconds)


record = asyncio.run(
    execute_event(
        'Arsenal vs Everton',
        bettor,
        dataloader,
        Exchange(),
        stake=10.0,
        ledger_path=Path(tempfile.mkdtemp()) / 'ledger.jsonl',
        clock=lambda: now['time'],
        wait=wait,
    ),
)
print(record.outcome.value)
print(record.steps[['step', 'detail']].to_string(index=False))
print(record.receipts[['selection', 'market', 'status', 'stake', 'price']].to_string(index=False))
```

[`execute_event`][sportsbet.execution.execute_event] takes the match, the fitted bettor, the dataloader and the venue,
and these settings:

* `stake` is the fixed amount to place.
* `ledger_path` is the ledger file, described below.
* `lead` is how long before kickoff the bet goes on, five minutes by default.
* `poll` is how often the match is read while the run waits, five minutes by default. Each poll downloads the fixtures
  again, so a shorter poll asks more of the source for no gain before kickoff.
* `live` arms the run.
* `clock` and `wait` replace the real time, as above, so a run can be tested with no waiting.

## The run record

A run returns an [`EventRunRecord`][sportsbet.execution.EventRunRecord]. It holds the match, the betting moment, how
the run ended as a [`RunOutcome`][sportsbet.execution.RunOutcome], every step with its time and detail, and the receipt
when there is one. The steps and the receipt are validated frames, and each step is also logged to the
`sportsbet.execution` logger as it happens, so the terminal and the record agree. The outcomes are:

* `placed`, `dry_run` and `no_value`, for a run that reached its decision.
* `refused`, when the venue offered less, did not offer the selection, or refused the bet.
* `already_placed`, when the ledger holds the bet.
* `moment_passed`, `venue_failed`, `event_unlisted` and `data_failed`, for a run that stopped before deciding.

## At most one bet

A run keeps a ledger file at the path you give, one JSON object per line. It records the bet as attempted before it
asks the venue to place it, and records the receipt after. A rerun reads the latest line for its bet:

* a placed bet is not placed again, and the run records `already_placed`.
* a bet attempted with no receipt, as after a crash, is not placed again. The run records `refused_unconfirmed` and
  tells you to check the venue for the bet.
* a dry run, a refused price or a rejection lets the bet go on.

Keep one ledger for all your runs. The ledger holds no secret.

## Writing an API venue

A venue is a [`BaseVenue`][sportsbet.execution.BaseVenue] subclass with a `key` and three coroutines, as `Exchange`
above shows.

* `authenticate` logs in, reading each secret from the variable it is named by.
* `read_price` returns the price on offer for a match, market and selection, or `None`, and stakes nothing.
* `place` reads the price first, refuses with `refused_price` when it is below the intent's `min_price`, and otherwise
  places the bet once. Its receipt carries only what the venue confirmed: the status, the price and the venue's bet
  identifier. A bet the venue does not confirm is `refused_unconfirmed`.

Listing markets, reading the balance, reading the status of bets and cancelling are optional. A venue that does not
override them says it does not support them. `close` releases what the venue holds open, and does nothing by default.

A secret is named, never passed. [`resolve`][sportsbet.execution.resolve] reads the variable a
[`CredentialRef`][sportsbet.execution.CredentialRef] names, and a missing variable names itself and stops.

```python
from sportsbet.execution import CredentialRef, resolve

token = resolve(CredentialRef('EXCHANGE_TOKEN'))
```

## Writing a browser venue

A bookmaker's website is a venue too. [`BaseBrowserVenue`][sportsbet.execution.BaseBrowserVenue] drives it through a
[`BrowserSession`][sportsbet.execution.BrowserSession], and you complete it for your bookmaker by writing seven hooks.
The base runs them in a fixed order: it opens the event, adds the selection to the slip, reads the slip, checks it holds
the intended selection, checks the price, enters the stake, confirms and reads the confirmation. A step that fails stops
the sequence before anything is staked. A hook reaches the page through `self.session.page_`, the playwright page.

The code below is reference code for a made-up bookmaker, since driving a real site cannot run here.

```python
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from sportsbet.execution import BaseBrowserVenue, BrowserSession, Confirmation, PlacementStatus


class Book(BaseBrowserVenue):
    async def check_login(self):
        return await self.session.page_.get_by_role('link', name='My account').count() > 0

    async def find_event(self, match):
        await self.session.navigate(f'{self.session.url}/football')
        link = self.session.page_.get_by_role('link', name=match)
        return await link.get_attribute('href') if await link.count() else None

    async def add_selection(self, identity):
        await self.session.page_.get_by_role('button', name=identity.selection).first.click()

    async def read_slip(self):
        slip = self.session.page_.get_by_role('form', name='Bet slip')
        if not await slip.is_visible():
            return None
        return await slip.locator('.selection').inner_text(), float(await slip.locator('.price').inner_text())

    async def enter_stake(self, stake):
        await self.session.page_.get_by_role('textbox', name='Stake').fill(str(stake))

    async def confirm_bet(self):
        await self.session.page_.get_by_role('button', name='Place bet').click()

    async def read_confirmation(self):
        receipt = self.session.page_.get_by_role('status')
        try:
            await receipt.wait_for(timeout=10000)
        except PlaywrightTimeoutError:
            return None
        return Confirmation(PlacementStatus.ACCEPTED, venue_bet_id=await receipt.get_attribute('data-bet-id'))


VENUE = Book('book', BrowserSession(key='book', url='https://book.example', user_data_dir='.book-profile'))
```

The session opens the browser from the profile in `user_data_dir`. Log in once with a visible browser, by building the
session with `headless=False` and the same profile, and every later run lands logged in. A run whose profile holds no
login stops and says so.

`match_selection` compares the slip's label with the intended selection without case. Override it for a site that
translates team names.

To find the locators, explore the site. `sportsbet execution page read` shows a page as an accessibility snapshot, with
a ref for everything you can act on, and `sportsbet execution page act` clicks, fills or chooses by ref and shows the
page it produced. Both accept a browser session or a browser venue. The `browser_*` tools of the MCP server do the same
for an agent.

```bash
sportsbet execution page read --venue venue.py:VENUE --url https://book.example/football
```

## The command line and the tools

The command runs the same run as the Python call.

```bash
# A dry run: read the venue's price and record the bet it would place, staking nothing.
sportsbet execution run --venue venue.py:VENUE -d loader.pkl -b model.pkl \
  --event "Arsenal vs Chelsea" --stake 10 --output out

# Armed: the same run, placing the stake at the moment.
sportsbet execution run --venue venue.py:VENUE -d loader.pkl -b model.pkl \
  --event "Arsenal vs Chelsea" --stake 10 --live --output out
```

It takes `--lead`, `--poll` and `--ledger` as the Python call does, prints the steps and the receipt, and writes them
to `out/sports-betting-data/run_steps.csv` and `out/sports-betting-data/receipts.csv`. The MCP server exposes the same
run as the `execution_run` tool, which returns the outcome, the moment, the steps and the receipt. The other execution
commands, `venue`, `markets`, `balance`, `status` and `cancel`, map one to one to the `execution_*` tools.
