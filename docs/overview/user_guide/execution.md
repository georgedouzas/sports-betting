# Execution

Everything else in this library reads data and produces numbers. This part spends money.

## Read this first

Driving a bookmaker's website breaches essentially every bookmaker's terms of service. Being caught means the
account is closed and the balance in it is gone. That is a matter between you and the bookmaker, and the library states
it rather than deciding it for you.

There is no sandbox anywhere. No exchange offers one. Betfair's delayed application key is widely described as a
test environment and it is not: it places real bets on the live exchange, with delayed prices. Pointing anything at it
to try things out spends real money.

A bet goes on at the price that is there when it lands, which can be worse than the price the model was backtested
against, down to the minimum you allowed.

Nothing here evades a venue's automation controls. There is no stealth browsing, no fingerprint spoofing, no
captcha solving and no proxy rotation. A venue that blocks automation is reported, and that is the end of it.

## Install it

```bash
pip install 'sports-betting[execution]'
python -m playwright install chromium
```

The browser is only needed to drive a website. A venue with an API needs nothing beyond the first line.

## What refuses, and where

There are two ways to reach a venue, and they keep different promises. The page says which so that you can tell them
apart before money is involved.

| | A venue with an API | A bookmaker's website |
| --- | --- | --- |
| Who places the bet | the library | your agent |
| One stake per bet, whatever happens | the library keeps this | your agent's job |
| The stake and exposure ceilings | the library keeps these | your agent's job |
| Refusing without a confirmation | the library keeps this | your agent's job |
| Reference implementation | `BetfairVenue` | `BrowserSession` |

The difference comes from what each path knows. The library calls a venue's API itself, so it is the thing that can
count the exposure and recognise a bet that already went on. A website is driven by your agent, and recognising a bet
that already went on there means reading that site's own bet history, which means knowing how that site is built. The
library holds no knowledge of any site, so on that path those promises are yours to keep.

## Placing at a venue with an API

Fit a model and extract the fixtures the way you already do, then quote what would be staked.

```bash
sportsbet execution quote --venue betfair --dataloader dataloader.pkl --bettor model.pkl \
                          --stake 10 --output quote.json
```

The quote lists every bet, its stake, its price, and the total exposure of the batch. Nothing has moved.

```bash
sportsbet execution place --venue betfair --quote quote.json \
                          --confirm-stake 50.0 --confirm-exposure 50.0 \
                          --max-stake 10 --max-exposure 100
```

The bets go on only when both figures match the quote exactly. Anything else stakes nothing and states the real
figures. There is no dry run flag, because a dry run is what happens when a confirmation is missing, so there is no
default to set wrongly and no flag to forget.

The ceilings are checked before each bet, and the bets go on one at a time. A batch that reaches a ceiling partway
leaves the bets that already went on alone and refuses the rest, naming the ceiling it reached.

## Placing once and only once

A bet is the same bet when it names the same venue, the same match, the same market and the same selection. Its
reference is computed from those four rather than remembered, so a run that crashed and started again with nothing
carried over computes the same reference and finds its own bet at the venue.

That is why nothing is written down. The venue is the record, and a second copy of it is a thing that goes out of step
with the first.

At Betfair the reference goes on as both `customerRef` and `customerOrderRef`. A resubmission within sixty seconds is
de-duplicated by the venue, and one after that is recognised by reading the order back.

## Credentials

A credential is named, never passed.

```bash
export BETFAIR_APP_KEY=...
export BETFAIR_USERNAME=...
export BETFAIR_PASSWORD=...
```

The venue names the variables and reads them where they are used. A secret passed as an argument is a secret written
into a shell history, a transcript and a traceback. A missing variable names itself and stops.

A fitted bettor holds no credential and reaches nothing, so a saved model cannot spend money.

## Driving a bookmaker's website

For a bookmaker with no API, the library supplies a browser and the knowledge of the site is yours. Write down what
you know for the agent.

```python
# venue.py
from sportsbet.execution import BrowserSession

VENUE = BrowserSession(
    key='example',
    url='https://example.invalid/betting',
    notes="""
    Bet history is under My Account.
    The slip opens on the right after clicking a price. Confirm is two steps.
    Check the history for this match before placing, so a retry does not stake twice.
    """,
    credential_env=('EXAMPLE_USERNAME', 'EXAMPLE_PASSWORD'),
    min_interval=1.0,
)
```

The notes are stored as written and handed back as written. The library never reads them, so they are for the agent.
The first run is exploration: the agent opens the site, works out the layout, and tells you what it found. Paste what
lasts into the notes, and the knowledge accumulates in your file rather than in this library.

A page comes back as an accessibility snapshot, which is compact enough to reason over and carries a ref for
everything that can be acted on.

```yaml
- form "Bet slip" [ref=e2]:
  - textbox "Stake" [ref=e3]: "10.00"
  - button "Place bet" [ref=e4]
  - button "Confirm bet" [disabled] [ref=e5]
```

The refs are what `click`, `type` and `select` take. An element the site has disabled or hidden is not clicked, and the
tool says so rather than reporting a click that did not happen. That matters most on the control that confirms a wager.

Exploring is slow and placing is not, so pin what exploring found.

```
sportsbet execution page fix --venue venue.py:VENUE --url https://example.invalid/match \
                            --match 'Arsenal vs Chelsea' --locator 'stake=textbox[name="Stake"]'
```

Pinning stores roles and accessible names, which survive the page re-rendering. A ref belongs to one state of the page
and is refused. A price is read when the bet is placed and checked against your minimum, so pinning one is refused too.

## Driving it with an agent

The tools are how an agent reaches all of this. The agent lives outside the library and makes the decisions, and the
library holds no model, no key and no loop.

```bash
pip install 'sports-betting[mcp,execution]'
```

An agent that calls `execution_place` without the quoted figures stakes nothing and is told what the figures are. The
rule is in the code rather than in the description of the tool, because the caller most likely to skim a description is
an agent.

## Where the exchanges are available

Betfair is the reference implementation because it is the only exchange of the four surveyed that carries a reference
of the caller's on a bet, which is what lets the venue be the record. Smarkets and Matchbook have no such field, and
Betdaq's is a number it does not police and cannot filter on.

Availability is a separate question from the code. All four of those exchanges are closed to some countries, either by
the regulator or by the exchange itself, so check whether the one you want accepts customers where you live before
counting on it. Where an exchange will not have you, the website path is what is left, with the risk at the top of this
page.
