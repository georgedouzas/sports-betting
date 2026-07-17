# Execution

Execution places the value bets a bettor found. Everything else in the library reads data and returns numbers. This
part spends real money, so read the risks first.

## Risks

- Driving a bookmaker's website breaches almost every bookmaker's terms of service and risks the account and its
  balance.
- A venue's own test key may still place real bets. There is no substitute for reading what it does before you use it.
- A bet goes on at the price on offer when it lands, down to the minimum you set.
- Nothing here evades a venue's automation controls. A venue that blocks automation is reported and placement stops.

## Install

```bash
pip install 'sports_betting[execution]'
python -m playwright install chromium
```

The browser is needed only to drive a website.

## Two venues

The library ships no bookmaker. It ships the venue contract and a browser, and you supply the venue.

A venue with an official API is a `BaseVenue` you implement, and the library places at it. A `BrowserSession` is a
bookmaker with no API, driven in a browser by an agent. Only the first keeps once-only placement and the stake limits,
since the second is driven by the agent rather than the library.

A venue is named the way a model is, by where it lives, as in `venue.py:VENUE`.

## A venue with an API

You implement the contract. A minimal one keeps its bets in memory, which is what the tests use. Yours calls the
bookmaker's API.

```python
import pandas as pd
from sportsbet.execution import BaseVenue, PlacementReceipt, PlacementStatus


class MyVenue(BaseVenue):
    key = 'my-venue'
    can_cancel = True

    async def authenticate(self):
        ...

    async def list_markets(self, matches):
        ...

    async def read_balance(self):
        ...

    async def place(self, intent):
        ...

    async def read_status(self, identities):
        ...

    async def cancel(self, identity):
        ...
```

`place` puts one bet on and returns a `PlacementReceipt`. It goes on once for an identity: asked again for the same
selection, it reports the bet already there rather than staking twice. The [gallery example][execution-gallery] writes
one in full and shows the safety model.

## Placing

`execute` takes a fitted bettor and its dataloader, picks the upcoming matches it can still bet on, and places them one
at a time.

```python
import asyncio
from sportsbet.execution import execute

receipts = asyncio.run(
    execute(venue, dataloader, bettor, stake=10.0, max_stake=10.0, max_exposure=1000.0),
)
```

That run stakes nothing, because `confirm_total` is missing. It logs the total it would stake, and you pass that total
back to place the bets.

```python
receipts = asyncio.run(
    execute(venue, dataloader, bettor, stake=10.0, max_stake=10.0, max_exposure=1000.0, confirm_total=250.0),
)
```

A wrong total refuses and states the real one. A model fitted for a live moment bets at the kickoff plus that time, so
set `window='2h'` to bound how long the run keeps placing.

[execution-gallery]: ../../generated/gallery/execution/plot_place_value_bets.md

## Credentials

A venue names the variable holding a secret and never takes the secret itself.

```bash
export VENUE_USERNAME=...
export VENUE_PASSWORD=...
```

A missing variable names itself and stops. A saved bettor holds no credential.

## Driving a website

For a bookmaker with no API, write down what you know for the agent.

```python
from sportsbet.execution import BrowserSession

venue = BrowserSession(
    key='novibet',
    url='https://www.novibet.gr/stoixima',
    notes='Bet history is under My Account. The slip opens on the right after clicking a price.',
    credential_env=('NOVIBET_USERNAME', 'NOVIBET_PASSWORD'),
    headless=False,
)
```

`notes` is handed to the agent unread. `headless=False` opens a window so you can watch the agent work. A page comes
back as an accessibility snapshot with a ref for every element to act on, and a disabled control is not clicked.

```yaml
- form "Bet slip" [ref=e2]:
  - textbox "Stake" [ref=e3]: "10.00"
  - button "Place bet" [ref=e4]
  - button "Confirm bet" [disabled] [ref=e5]
```
