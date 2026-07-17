# Execution

Execution places the value bets a bettor found. Everything else in the library reads data and returns numbers. This
part spends real money, so read the risks first.

## Risks

- Driving a bookmaker's website breaches almost every bookmaker's terms of service and risks the account and its
  balance.
- No exchange offers a placement sandbox. Betfair's delayed application key places real bets on the live exchange.
- A bet goes on at the price on offer when it lands, down to the minimum you set.
- Nothing here evades a venue's automation controls. A venue that blocks automation is reported and placement stops.

## Install

```bash
pip install 'sports_betting[execution]'
python -m playwright install chromium
```

The browser is needed only to drive a website.

## Two venues

A `BetfairVenue` has an official API and the library places at it. A `BrowserSession` is a bookmaker with no API, driven
in a browser by an agent. Only the first keeps once-only placement and the stake limits, since the second is driven by
the agent rather than the library.

A venue is named the way a model is, either ready made or by where your own lives.

```python
from sportsbet.execution import BetfairVenue

venue = BetfairVenue()
```

## Placing

`execute` takes a fitted bettor and its dataloader, picks the upcoming matches it can still bet on, and places one at a
time. Nothing stakes until you pass back the quoted total.

```python
from sportsbet.execution import execute

execute(venue, dataloader, bettor, stake=10.0, max_stake=10.0, max_exposure=100.0, confirm_total=250.0)
```

Without `confirm_total` it prints the plan and stakes nothing. A wrong total refuses and prints the real one. Matches
are handled one at a time, in random order, up to `window` if you set one for live betting.

## Credentials

A venue names the variable holding a secret and never takes the secret itself.

```bash
export BETFAIR_APP_KEY=...
export BETFAIR_USERNAME=...
export BETFAIR_PASSWORD=...
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
