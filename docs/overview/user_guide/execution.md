# Execution

A bettor finds value bets and stops. Execution places them. It takes the value bets a fitted bettor produces, turns
them into intents, and places them at a venue where you hold an account. Placing spends real money, so read the risks
before anything else.

## Risks

* Driving a bookmaker's website breaches almost every bookmaker's terms of service and risks the account and its balance.
* A venue's own test key may still place real bets. Read what it does before you point anything at it.
* A bet goes on at the price on offer when it lands, down to the minimum price you set.
* Nothing here evades a venue's automation controls. A venue that blocks automation is reported and placement stops.

## Installation

```bash
pip install 'sports_betting[execution]'
python -m playwright install chromium
```

The browser is needed only to drive a website. A venue with an API needs the first line alone.

## Two venues

The library ships no bookmaker. It ships the venue contract and a browser, and you supply the venue.

* A venue with an official API is a [`BaseVenue`][sportsbet.execution.BaseVenue] you implement. The library calls it, so
  the library keeps the guarantees: it places once and only once, holds to the stake and exposure limits, and refuses
  until you confirm the quote.
* A bookmaker with no API is a [`BrowserSession`][sportsbet.execution.BrowserSession] driven in a browser by an agent.
  The agent places, so placing once and only once and the limits are the agent's to keep.

Both are named the way a model is, by where they live, as in `venue.py:VENUE`.

## The venue contract

A `BaseVenue` implements six methods. Every one is a coroutine, so you `await` it, or run it with `asyncio.run`.

* `authenticate()` logs in, reading each secret from the variable the venue names. Returns nothing.
* `list_markets(matches)` returns a frame with a `price` column for each `match`, `market` and `selection` on offer.
* `read_balance()` returns `(balance, exposure)`, the funds available and the amount already at stake.
* `place(intent)` puts one bet on and returns a [`PlacementReceipt`][sportsbet.execution.PlacementReceipt]. It goes on
  once for an identity: asked again for the same selection, it reports the bet already there rather than staking twice.
* `read_status(identities)` returns what the venue holds for the given identities.
* `cancel(identity)` cancels a bet where the venue allows it, or raises `CancellationUnsupportedError`.

The class also carries two attributes: `key`, the name of the venue, and `can_cancel`, whether it cancels.

### Writing an API venue

Subclass [`BaseVenue`][sportsbet.execution.BaseVenue] and implement the six methods. The venue below keeps its bets in
memory, which is what the tests use. Yours calls the bookmaker's API in the same places.

```python
import pandas as pd
from sportsbet.execution import BaseVenue, PlacementReceipt, PlacementStatus


class DemoVenue(BaseVenue):
    """An in-memory venue. A real one calls a bookmaker's API in the same places."""

    key = 'demo'
    can_cancel = True

    def __init__(self, prices):
        self.prices = prices          # {(match, market, selection): price}
        self.orders = {}              # what has been placed, keyed by identity.ref

    async def authenticate(self):
        # A real venue reads its key here, e.g. resolve(CredentialRef('VENUE_API_KEY')).
        ...

    async def list_markets(self, matches):
        records = [
            {'match': match, 'market': market, 'selection': selection, 'price': price}
            for (match, market, selection), price in self.prices.items()
            if match in matches
        ]
        return pd.DataFrame.from_records(records, columns=['match', 'market', 'selection', 'price'])

    async def read_balance(self):
        return 10000.0, sum(order['stake'] for order in self.orders.values())

    async def place(self, intent):
        ref = intent.identity.ref
        if ref in self.orders:
            order = self.orders[ref]
            return PlacementReceipt(
                identity=intent.identity,
                status=PlacementStatus.ALREADY_PLACED,
                stake=order['stake'],
                price=order['price'],
                value_bet=intent.value_bet,
                detail='This selection already has a bet.',
            )
        price = self.prices[(intent.identity.match, intent.identity.market, intent.identity.selection)]
        self.orders[ref] = {'stake': intent.stake, 'price': price}
        return PlacementReceipt(
            identity=intent.identity,
            status=PlacementStatus.MATCHED_FULL,
            stake=intent.stake,
            price=price,
            value_bet=intent.value_bet,
        )

    async def read_status(self, identities):
        held = [{'ref': i.ref, **self.orders[i.ref]} for i in identities if i.ref in self.orders]
        return pd.DataFrame.from_records(held)

    async def cancel(self, identity):
        self.orders.pop(identity.ref, None)
        return PlacementReceipt(identity=identity, status=PlacementStatus.REJECTED, detail='Cancelled.')
```

The [gallery example][execution-gallery] runs this venue end to end.

## From value bets to intents

A bettor returns value bets. [`value_bet_intents`][sportsbet.execution.value_bet_intents] turns them into the intents a
venue places.

```python
from sportsbet.execution import value_bet_intents

intents = value_bet_intents('demo', bettor, X_fix, O_fix, stake=10.0)
```

Each item is a [`PlacementIntent`][sportsbet.execution.PlacementIntent] with four fields.

* `identity`, the [`BetIdentity`][sportsbet.execution.BetIdentity] of the bet.
* `stake`, what to put on it.
* `min_price`, the lowest acceptable price. It defaults to the price the value bet was computed at, since below that
  price the bet is no longer a value bet.
* `value_bet`, a label tracing the bet back to the match and market it came from.

The identity is a venue, a match, a market and a selection, and nothing else. Its `ref` is derived from those four, so
two intents that name the same selection are the same bet, whichever run produced them.

```python
intent = intents[0]
assert intent.identity.venue == 'demo'
assert intent.stake == 10.0
assert len(intent.identity.ref) == 32
```

## Quoting

[`quote`][sportsbet.execution.quote] returns what would be staked before anything is. It reads the venue balance, so it
knows the exposure already open.

```python
import asyncio
from sportsbet.execution import ExposureLimits, quote

limits = ExposureLimits(max_stake_per_bet=10.0, max_total_exposure=1000.0)
quoted = asyncio.run(quote(venue, intents, limits))
assert quoted.total_stake == sum(intent.stake for intent in intents)
```

A [`PlacementQuote`][sportsbet.execution.PlacementQuote] carries the `intents`, the `total_stake` of the batch, the
`total_exposure` including what is already open, and the time it was quoted. [`ExposureLimits`][sportsbet.execution.ExposureLimits]
carries `max_stake_per_bet`, `max_total_exposure` and `killed`, the kill switch.

## Placing

[`place`][sportsbet.execution.place] takes the quote back and places it. It stakes nothing unless you pass the quoted
figures back exactly.

```python
from sportsbet.execution import place

receipts = asyncio.run(place(venue, quoted, limits))
```

That run is a dry run. It stakes nothing, not because a flag was set, but because the confirmation is missing. There is
no `dry_run` flag to forget. To place the bets, pass the quoted figures back.

```python
receipts = asyncio.run(
    place(venue, quoted, limits, confirm_stake=quoted.total_stake, confirm_exposure=quoted.total_exposure),
)
```

`place` returns a frame of receipts, one row per intent, with the columns `ref`, `venue`, `match`, `market`,
`selection`, `status`, `stake`, `price`, `venue_bet_id`, `value_bet`, `placed_at` and `detail`. The `status` is one of:

* `matched_full`, `matched_partial`, `accepted`, the bet went on.
* `already_placed`, the venue already held the bet, so nothing was staked again.
* `dry_run`, no confirmation was passed, so nothing was staked.
* `refused_unconfirmed`, the confirmed figures did not match the quote. `detail` states the real ones.
* `refused_limit`, the bet would breach `max_stake_per_bet` or `max_total_exposure`. `detail` names the limit.
* `refused_price`, the venue price was below `min_price`.
* `refused_killed`, the kill switch was on.
* `blocked`, the venue blocked automation, so placement stopped.
* `rejected`, the venue declined the bet.

The bets go on one at a time, with the exposure counted before each. A batch that reaches a limit partway leaves the
bets already placed alone and refuses the rest.

## Placing once and only once

The identity `ref` is derived, not stored, so a retry, a reconnection or a fresh run recomputes the same `ref` and finds
the bet already at the venue.

```python
first = asyncio.run(place(venue, quoted, limits, confirm_stake=quoted.total_stake, confirm_exposure=quoted.total_exposure))
again = asyncio.run(place(venue, quoted, limits, confirm_stake=quoted.total_stake, confirm_exposure=quoted.total_exposure))
assert (first['status'] == 'matched_full').all()
assert (again['status'] == 'already_placed').all()
```

Nothing is written to disk. The venue is the record, so there is no second copy to drift out of step with it.

## Running the whole thing

[`execute`][sportsbet.execution.execute] runs the four steps in order.

1. It authenticates at the venue. If that fails, it stops and places nothing.
2. It selects the upcoming matches the bettor bets on and can still reach, sized by `stake`.
3. It orders them by their moment.
4. It places them in turn, waiting until each moment, and stakes nothing until `confirm_total` matches the total.

```python
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

`stake` is a number for the same stake on every bet, or a mapping keyed by `(match, market, selection)` for a stake per
event, which is where sizing computed offline goes. A mapping stakes only the events it holds.

```python
sizing = {('Arsenal vs Chelsea', 'home_win', 'Arsenal'): 25.0, ('Everton vs Fulham', 'draw', 'Everton'): 15.0}
receipts = asyncio.run(execute(venue, dataloader, bettor, stake=sizing, confirm_total=40.0))
```

The dataloader fixes the moment the model bets at. A prematch model bets at the kickoff. A live model, fitted with
`target_event_status='inplay'` and a `target_event_time`, bets at the kickoff plus that time.
[`betting_moment`][sportsbet.execution.betting_moment] returns that moment for a kickoff, and
[`feasible`][sportsbet.execution.feasible] returns which upcoming matches the bet can still go on for. Set `window` to
bound how long the run keeps placing, which matters for a live model, since watching one match at a time means a busy
slot of simultaneous kickoffs cannot all be reached.

```python
receipts = asyncio.run(
    execute(venue, dataloader, bettor, stake=10.0, confirm_total=250.0, window=pd.Timedelta('2h')),
)
```

The run logs each selection and placement through the `sportsbet.execution` logger, so the command line shows it as it
goes.

### Running a browser venue

`execute` runs a browser venue the same way, through the same four steps, with one difference: the library cannot click
a bet slip, so it hands each event to a `placer` you supply. The placer drives the site for one bet and returns its
receipt. Authentication opens the browser at the site, and a profile that already holds a login lands logged in.

```python
async def placer(intent, session):
    await session.navigate(session.url)
    # find the market, fill the stake, work the confirm flow for `intent`, using session.snapshot/click/type
    return PlacementReceipt(identity=intent.identity, status=PlacementStatus.MATCHED_FULL, stake=intent.stake)


receipts = asyncio.run(execute(browser_venue, dataloader, bettor, stake=10.0, placer=placer, confirm_total=250.0))
```

The placer holds the site knowledge, so placing once and only once is its job too: it reads the site's bet history for
the match before placing, since the library cannot know how that site records a bet.

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

For a bookmaker with no API, the library supplies a browser and you supply the knowledge of the site.
[`BrowserSession`][sportsbet.execution.BrowserSession] takes the venue and how to reach it.

```python
from sportsbet.execution import BrowserSession

venue = BrowserSession(
    key='novibet',
    url='https://www.novibet.gr/stoixima',
    notes="""
    Bet history is under My Account.
    The slip opens on the right after clicking a price. Confirm is two steps.
    Check the history for this match before placing, so a retry does not stake twice.
    """,
    credential_env=('NOVIBET_USERNAME', 'NOVIBET_PASSWORD'),
    min_interval=1.0,
    headless=False,
)
```

* `notes` is a free-text blob handed to the agent unread. It is where your knowledge of the site lives.
* `credential_env` names the variables holding the username and password.
* `user_data_dir` is where the browser keeps its profile, so a login survives a restart.
* `min_interval` is the seconds to leave between actions.
* `headless` hides the browser window. `headless=False` opens it, so you can watch the agent work.

### The primitives

The session drives the page through a small set of methods. Each act returns the page it produced, so the agent sees
what it did without asking again.

* `navigate(url)` goes to a page and returns it.
* `snapshot(selector=None, depth=None)` returns the page, or a part of it, without acting.
* `click(ref)` clicks an element.
* `type(ref, text)` fills an element.
* `select(ref, value)` chooses an option.

A page comes back as an accessibility snapshot: compact YAML with a ref for every element that can be acted on.

```yaml
- form "Bet slip" [ref=e2]:
  - textbox "Stake" [ref=e3]: "10.00"
  - button "Place bet" [ref=e4]
  - button "Confirm bet" [disabled] [ref=e5]
```

The refs are what `click`, `type` and `select` take. An element the site has disabled or hidden is not clicked, and the
method fails rather than reporting a click that did not happen. That matters most on the control that confirms a wager.

### Explore, then fix, then bet

Exploring a page is slow, since it reads whole snapshots and reasons over them. Placing a bet is not. So explore once,
pin what you found, and act against the pinned session.

```python
await venue.navigate('https://www.novibet.gr/stoixima')
await venue.snapshot()                                   # read the layout
fixed = venue.fix('Arsenal vs Chelsea', {'stake': 'textbox[name="Stake"]', 'confirm': 'button[name="Place bet"]'})
await venue.resolve('stake')                             # read a pinned locator now
```

[`fix`][sportsbet.execution.BrowserSession.fix] pins roles and accessible names, which survive the page re-rendering. It
refuses a ref, which belongs to one state of the page, and it stores no price, since the price is read when the bet is
placed and checked against the minimum. `resolve(name)` reads a pinned locator as it is now.

The session holds one browser for its life, so a login lasts across calls. `start()` opens it, and `stop()` closes it,
though `navigate` opens it for you.

## The three surfaces

Everything above is reachable from the command line and from the MCP server, so an agent reaches it too.

```bash
sportsbet execution run --venue venue.py:VENUE --dataloader dataloader.pkl --bettor model.pkl \
  --stake 10 --confirm-total 250 --max-stake 10 --max-exposure 1000
```

`execution quote`, `execution place`, `execution status`, `execution balance`, `execution markets` and
`execution cancel` cover the pieces, and `execution page read`, `execution page act` and `execution page fix` drive a
website. The MCP server exposes the same set as `execution_run`, `execution_quote`, `execution_place` and the
`browser_*` tools, so an agent with no shell reaches them all.

[execution-gallery]: ../../generated/gallery/execution/plot_place_value_bets.md
