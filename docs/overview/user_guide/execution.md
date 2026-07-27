# Execution

A bettor finds value bets and stops. Execution acts on one of them. The single-event unit takes a fitted bettor, one
upcoming match, and a fixed stake, watches that one event, and at the moment the model was fitted for places the model's
bet, once, at a bookmaker where you hold an account. Placing spends real money, so read the risks before anything else.

## Risks

* Driving a bookmaker's website breaches almost every bookmaker's terms of service and risks the account and its balance.
* The unit watches one event. Driving many units across many events at once is your job, and a bookmaker may block or
  close the account for that behaviour. That price is yours to pay.
* A bet goes on at the price on offer when it lands, down to the minimum price the value bet was computed at.
* Nothing here evades a venue's automation controls. A venue that blocks automation is reported and placement stops.

## Installation

```bash
pip install 'sports_betting[execution]'
python -m playwright install chromium
```

The browser drives the bookmaker's website to log in and to place. Monitoring does not use it: what the unit shows you
comes from the configured source, logged to the terminal.

## One event at a time, by design

The unit handles exactly one betting event. It is not a fleet manager. It binds one match, one fitted bettor, and one
browser session, and it places at most one bet over its whole run. Extending it to many events is left to you, and so is
the consequence: a bookmaker that recognises automation across many matches may close the account.

## The deterministic sequence

A run is a fixed, ordered sequence. Nothing in it is interactive once it starts.

1. **Explore the URLs.** You give the unit one or more candidate bookmaker URLs. It navigates each headless and keeps the
   one whose page carries the event, pinning that page's controls. If no URL carries the event, it stops and stakes
   nothing.
2. **Log in.** It ensures the browser session is authenticated before it monitors or places. A browser profile that
   already holds a login lands logged in; otherwise you log in during setup. If login is not in place, it stops and
   stakes nothing.
3. **Monitor and log.** It polls the configured source on a schedule and logs the event to the terminal: the event's
   identity, its status as it advances through preplay, inplay and postplay, the current price where it is available, and
   its decision at each step. The browser runs headless with no preview window, so the terminal log is the whole view.
4. **Place at the moment.** When the event reaches the moment the model was fitted for, it applies the bettor to the
   event's data as of that moment. If the model finds a value bet, it logs the selection, stake and price it is about to
   stake, then places the configured stake on that selection, once. If the model finds no value, it logs that and stakes
   nothing.

The bettor decides whether to bet and which selection to back. It never decides the stake.

## The betting moment

The dataloader fixes the moment the model bets at. A preplay model bets at the kick-off. A live model, fitted with
`target_event_status='inplay'` and a `target_event_time`, bets at the kick-off plus that time.
[`find_betting_moment`][sportsbet.execution.find_betting_moment] returns that moment for a kick-off. If the moment has
already passed when the unit starts, because the event is in-play past the fitted moment or already finished, it logs
that the moment is unreachable, places nothing, and returns an empty frame.

## Dry run and `--live`

A run is a no-stakes dry run unless you arm it. A dry run does everything, exploring, logging in, monitoring, deciding
and logging the exact bet it would place, but it never calls the placer, so no money moves and the receipts frame is
empty. Arming the run with `--live` is a single up-front opt-in. Once armed, the unit places automatically at the moment
with no second confirmation, and its pre-place log line is what makes the action legible before it happens.

```bash
# Pin the site's controls once, so the unit knows where the stake and confirm buttons are.
sportsbet execution page fix --venue venue.py:VENUE --url "$URL" \
  --match "Arsenal vs Chelsea" \
  --locator stake='textbox[name="Stake"]' --locator confirm='button[name="Place bet"]'

# A dry run: watch the event and log the bet it would place, staking nothing.
sportsbet execution run --venue venue.py:VENUE -d loader.pkl -b model.pkl \
  --event "Arsenal vs Chelsea" --stake 10 --url "$URL"

# Armed: the same run, placing the stake at the moment.
sportsbet execution run --venue venue.py:VENUE -d loader.pkl -b model.pkl \
  --event "Arsenal vs Chelsea" --stake 10 --url "$URL" --live
```

The stake is an execution parameter of the run, not the bettor's. The unit places exactly that amount when it bets.

## The Python API

The command line drives [`execute_event`][sportsbet.execution.execute_event]. Call it directly to run the unit from
Python.

```python
import asyncio
import pandas as pd
from sportsbet.dataloaders import load_dataloader
from sportsbet.evaluation import load_bettor
from sportsbet.execution import BrowserSession, execute_event

session = BrowserSession(key='novibet', url='https://www.novibet.gr/stoixima')
receipts = asyncio.run(
    execute_event(
        'Arsenal vs Chelsea',
        load_bettor('model.pkl'),
        load_dataloader('loader.pkl'),
        session,
        stake=10.0,
        urls=['https://www.novibet.gr/stoixima/arsenal-chelsea'],
        live=False,
        poll=pd.Timedelta('30s'),
    ),
)
```

The call returns a receipts frame, one row when a bet was placed and no rows otherwise. `event` names the one match, and
must be among the dataloader's fixtures. `bettor` is a model fitted and saved by `evaluation fit`. `dataloader` supplies
the event's evolving data and carries the fitted moment. `stake` is the fixed amount, `urls` the candidate pages, and
`live` arms the run. `poll` is the interval between source polls, and `clock` and `wait` are injection points for time,
so a test drives the whole watch with no real waiting.

### A custom placer

By default the unit places by entering the stake into the pinned `stake` control and clicking the pinned `confirm`
control of the matched site. A `placer` overrides that for a site the default cannot drive. It is a coroutine that takes
the [`PlacementIntent`][sportsbet.execution.PlacementIntent] and the session, drives the site for the one bet, and
returns its [`PlacementReceipt`][sportsbet.execution.PlacementReceipt]. A custom placer is Python, so it is a Python-API
capability; the command line and the MCP tool place through the default placer over the pinned controls.

```python
async def placer(intent, session):
    stake_control = await session.resolve('stake')
    # drive the site for `intent`, then confirm the wager
    return PlacementReceipt(identity=intent.identity, status=PlacementStatus.MATCHED_FULL, stake=intent.stake)


receipts = asyncio.run(execute_event(event, bettor, dataloader, session, stake=10.0, urls=urls, placer=placer, live=True))
```

The [gallery example][execution-gallery] runs the whole unit end to end, offline, with a stand-in session and an
injected clock.

## Credentials

A venue names the variable holding a secret and never takes the secret itself. A secret passed as an argument is a
secret written into a shell history, a transcript and a traceback.

```python
from sportsbet.execution import CredentialRef, resolve

secret = resolve(CredentialRef('VENUE_API_KEY'))
```

[`resolve`][sportsbet.execution.resolve] reads the variable where it is used. A missing variable names itself and stops.
A fitted bettor holds no credential, so a saved and reloaded bettor cannot spend money.

## Driving a website

The library supplies a browser and you supply the knowledge of the site.
[`BrowserSession`][sportsbet.execution.BrowserSession] takes the venue and how to reach it.

```python
from sportsbet.execution import BrowserSession

session = BrowserSession(
    key='novibet',
    url='https://www.novibet.gr/stoixima',
    notes="""
    Bet history is under My Account.
    The slip opens on the right after clicking a price. Confirm is two steps.
    """,
    credential_env=('NOVIBET_USERNAME', 'NOVIBET_PASSWORD'),
    user_data_dir='.novibet-profile',
    min_interval=1.0,
)
```

* `notes` is a free-text blob handed to an agent unread. It is where your knowledge of the site lives.
* `credential_env` names the variables holding the username and password.
* `user_data_dir` is where the browser keeps its profile, so a login survives a restart and the headless run lands
  already authenticated.
* `min_interval` is the seconds to leave between actions.
* `headless` hides the browser window. It is `True` by default, since the unit needs no window to watch.

### Explore, then fix, then run

The session drives the page through a small set of methods: `navigate(url)` goes to a page and returns it,
`read_snapshot()` reads it without acting, and `click(ref)`, `type(ref, text)` and `select(ref, value)` act on it. Each
returns the page it produced, as an accessibility snapshot: compact YAML with a ref for every element that can be acted
on.

```yaml
- form "Bet slip" [ref=e2]:
  - textbox "Stake" [ref=e3]: "10.00"
  - button "Place bet" [ref=e4]
```

Exploring a page is slow, since it reads whole snapshots and reasons over them. Placing is not. So explore once, pin what
you found, and let the unit act against the pinned session.

```python
await session.navigate('https://www.novibet.gr/stoixima')
await session.read_snapshot()
session.fix('Arsenal vs Chelsea', {'stake': 'textbox[name="Stake"]', 'confirm': 'button[name="Place bet"]'})
```

[`fix`][sportsbet.execution.BrowserSession.fix] pins roles and accessible names, which survive the page re-rendering. It
refuses a ref, which belongs to one state of the page, and it stores no price, since the price is read when the bet is
placed and checked against the minimum. The `execution page` commands, `read`, `act` and `fix`, do this exploring from
the command line, and the `browser_*` tools do it from the MCP server.

## The three surfaces

The single-event unit is reachable from the command line and from the MCP server, so an agent reaches it too.

```bash
sportsbet execution run --venue venue.py:VENUE --dataloader loader.pkl --bettor model.pkl \
  --event "Arsenal vs Chelsea" --stake 10 --url "$URL" --live
```

The MCP server exposes the same run as the `execution_run` tool, with the same parameters. The other execution
commands, `venue`, `markets`, `balance`, `status` and `cancel`, and the `page` group for exploring and pinning a site,
mirror one to one as the `execution_*` and `browser_*` tools.

[execution-gallery]: ../../generated/gallery/execution/plot_single_event.md
