# Execution

A bettor finds value bets. The single-event unit places one of them. It takes a fitted bettor, one match, and a stake.
It watches the match. At the moment the model was fitted for, it places the bet, once. You place the bet at a bookmaker
where you hold an account. This spends real money. Read the risks first.

## Risks

* You drive a bookmaker's website. This breaks the terms of service of almost every bookmaker. It puts the account and
  its balance at risk.
* The unit watches one event. Running many units across many events is your own work. A bookmaker may block or close an
  account for that.
* The unit places the bet at the price on offer when it lands. It will not place below the minimum price the value bet
  was computed at.
* The unit does not evade a website's automation controls. If a website blocks automation, the unit reports it and
  stops.

## Installation

```bash
pip install 'sports_betting[execution]'
python -m playwright install chromium
```

The browser drives the bookmaker's website to log in and to place the bet. Monitoring does not use the browser. The
unit reads the event from the configured source and logs it to the terminal.

## One event only

The unit handles exactly one betting event. It binds one match, one fitted bettor, and one browser session. It places at
most one bet over its whole run. You extend it to many events yourself. A bookmaker that detects automation across many
matches may close the account.

## What a run does

A run is a fixed, ordered sequence. Once it starts, it is not interactive.

1. Explore the URLs. You give the unit one or more candidate bookmaker URLs. It opens each one headless and keeps the
   page that carries the event. It pins that page's controls. If no URL carries the event, it stops and stakes nothing.
2. Log in. The unit checks that the browser session is authenticated before it monitors or places. A browser profile
   that already holds a login lands logged in. Otherwise you log in during setup. If login is not in place, the unit
   stops and stakes nothing.
3. Monitor and log. The unit polls the configured source on a schedule and logs the event to the terminal. It logs the
   event's identity, its status as it moves through preplay, inplay and postplay, the current price where it is
   available, and its decision at each step. The browser runs headless with no preview window, so the terminal log is
   the whole view.
4. Place at the moment. When the event reaches the moment the model was fitted for, the unit applies the bettor to the
   event's data as of that moment. If the model finds a value bet, the unit logs the selection, stake and price it is
   about to stake, then places the configured stake on that selection, once. If the model finds no value, the unit logs
   that and stakes nothing.

The bettor decides whether to bet and which selection to back. It never decides the stake.

## The betting moment

The dataloader fixes the moment the model bets at. A preplay model bets at the kick-off. A live model, fitted with
`target_event_status='inplay'` and a `target_event_time`, bets at the kick-off plus that time.
[`find_betting_moment`][sportsbet.execution.find_betting_moment] returns that moment for a kick-off.

The moment can already be past when the unit starts. This happens when the event is in-play past the fitted moment or
already finished. The unit then logs that the moment is unreachable, places nothing, and returns an empty frame.

## Dry run and live

A run is a no-stakes dry run unless you arm it. A dry run does everything else. It explores, logs in, monitors, decides,
and logs the exact bet it would place. It never calls the placer, so no money moves and the receipts frame is empty.

You arm a run with `--live`. This is a single up-front opt-in. Once armed, the unit places automatically at the moment
with no second confirmation. Its pre-place log line shows the action before it happens.

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

The stake is a parameter of the run, not of the bettor. The unit places exactly that amount when it bets.

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

The call returns a receipts frame. It holds one row when the unit placed a bet and no rows otherwise. The arguments are:

* `event` names the one match. It must be among the dataloader's fixtures.
* `bettor` is a model fitted and saved by `evaluation fit`.
* `dataloader` supplies the event's evolving data and carries the fitted moment.
* `stake` is the fixed amount to place.
* `urls` are the candidate pages.
* `live` arms the run.
* `poll` is the interval between source polls.
* `clock` and `wait` are injection points for time, so a test can drive the whole watch with no real waiting.

### A custom placer

By default the unit places the bet by entering the stake into the pinned `stake` control and clicking the pinned
`confirm` control of the matched site. A `placer` replaces that for a site the default cannot drive. A placer is a
coroutine. It takes the [`PlacementIntent`][sportsbet.execution.PlacementIntent] and the session, drives the site for
the one bet, and returns a [`PlacementReceipt`][sportsbet.execution.PlacementReceipt]. A custom placer is Python only.
The command line and the MCP tool place through the default placer over the pinned controls.

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

A venue names the variable that holds a secret. It never takes the secret itself. A secret passed as an argument ends up
in a shell history, a transcript and a traceback.

```python
from sportsbet.execution import CredentialRef, resolve

secret = resolve(CredentialRef('VENUE_API_KEY'))
```

[`resolve`][sportsbet.execution.resolve] reads the variable where it is used. A missing variable names itself and stops.
A fitted bettor holds no credential, so a saved and reloaded bettor cannot spend money.

## Driving a website

The library supplies the browser. You supply the knowledge of the site.
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

* `notes` is free text handed to an agent. Put your knowledge of the site here.
* `credential_env` names the variables that hold the username and password.
* `user_data_dir` is where the browser keeps its profile. A login survives a restart, so the headless run lands already
  authenticated.
* `min_interval` is the seconds to leave between actions.
* `headless` hides the browser window. It is `True` by default, since the unit needs no window to watch.

### Explore, fix, then run

The session drives the page through a small set of methods. `navigate(url)` goes to a page and returns it.
`read_snapshot()` reads the page without acting. `click(ref)`, `type(ref, text)` and `select(ref, value)` act on the
page. Each method returns the page it produced, as an accessibility snapshot. The snapshot is compact YAML with a ref
for every element you can act on.

```yaml
- form "Bet slip" [ref=e2]:
  - textbox "Stake" [ref=e3]: "10.00"
  - button "Place bet" [ref=e4]
```

Exploring a page is slow. The unit reads whole snapshots and reasons over them. Placing is fast. So you explore once,
pin what you found, and let the unit act against the pinned session.

```python
await session.navigate('https://www.novibet.gr/stoixima')
await session.read_snapshot()
session.fix('Arsenal vs Chelsea', {'stake': 'textbox[name="Stake"]', 'confirm': 'button[name="Place bet"]'})
```

[`fix`][sportsbet.execution.BrowserSession.fix] pins roles and accessible names. These survive the page re-rendering. It
refuses a ref, which belongs to one state of the page. It stores no price, since the unit reads the price when it places
the bet and checks it against the minimum. The `execution page` commands `read`, `act` and `fix` do this exploring from
the command line. The `browser_*` tools do it from the MCP server.

## The three surfaces

You reach the single-event unit from the command line and from the MCP server. An agent reaches it through the MCP
server too.

```bash
sportsbet execution run --venue venue.py:VENUE --dataloader loader.pkl --bettor model.pkl \
  --event "Arsenal vs Chelsea" --stake 10 --url "$URL" --live
```

The MCP server exposes the same run as the `execution_run` tool, with the same parameters. The other execution commands
`venue`, `markets`, `balance`, `status` and `cancel`, and the `page` group for exploring and pinning a site, map one to
one to the `execution_*` and `browser_*` tools.

[execution-gallery]: ../../generated/gallery/execution/plot_single_event.md
