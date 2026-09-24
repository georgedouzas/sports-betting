# Phase 1 Data Model: Bet execution

**Feature**: [spec.md](./spec.md) | **Research**: [research.md](./research.md) | **Date**: 2026-07-16

## BetIdentity

What makes two bets the same bet (FR-014). Derived from the venue, the match, the market and the
selection, and from nothing else. Not from the run, the model, the timestamp, or the batch.

| Field | Type | Notes |
| --- | --- | --- |
| `venue` | `str` | The venue key, such as `betfair` |
| `match` | `str` | The match identity as the dataloader already models it |
| `market` | `str` | The betting market, such as `home_win` |
| `selection` | `str` | The backed outcome |

**Derived**: `ref` is `blake2s(f'{venue}|{match}|{market}|{selection}', digest_size=16).hexdigest()`,
giving exactly 32 hex characters.

That width is not a coincidence and the constraint is Betfair's. Both reference fields cap at 32
characters over the charset `A-Za-z0-9 : - . _ + * ; ~`, and hex is a subset of it. The digest is
deterministic, so the same four inputs produce the same reference on any machine, in any run, with
no state kept anywhere. That is what lets FR-015 hold: the identity is recomputed rather than
remembered, so there is no local store to drift out of step with the venue.

### The two references collapse into one value

Research D3 established that Betfair needs both `customerRef` (deduplicates, 60 second window,
unreadable afterwards) and `customerOrderRef` (durable and filterable, uniqueness unpoliced), and
that neither alone satisfies FR-014.

Sequential placement (research D7) resolves this more cleanly than expected. `customerRef` is
scoped per request and `customerOrderRef` per instruction, so once a request carries exactly one
instruction the two scopes coincide and both fields take the same derived value:

```text
placeOrders(
    customerRef = identity.ref,                        # dedupes a retry inside 60s
    instructions = [PlaceInstruction(
        customerOrderRef = identity.ref,               # durable, filterable after 60s
        ...)],
)
```

The once-only guarantee then has no hole in it and needs no stored state:

| Failure | Recovery |
| --- | --- |
| Retry within 60s | Betfair deduplicates on `customerRef`. Zero extra stakes. |
| Retry after 60s, or a crash, or a fresh run | `listCurrentOrders(customerOrderRefs=[ref])` finds the bet. Report already placed (FR-014a). |
| Venue carries no reference at all (site-driven) | Recognise the bet from the venue's own bet history by the same four fields (FR-015). |

Emitting a colliding `order_ref` stays our defect to prevent rather than the venue's to catch,
since Betfair validates no uniqueness on it. The derivation makes a collision mean the bets are
genuinely the same bet, which is the intent.

## PlacementIntent

What the caller means to do. Immutable.

| Field | Type | Notes |
| --- | --- | --- |
| `identity` | `BetIdentity` | |
| `stake` | `float` | An input, never a model output. Bankroll strategy is out of scope. |
| `min_price` | `float` | Defaults to the price the value bet was computed at (FR-013), below which it stops being a value bet. The caller may set another. |
| `value_bet` | `str` | Back-reference to the row that caused it, for FR-016. |

## PlacementQuote

What the system promises before it does anything. This is the object FR-009 and FR-010 turn on:
the caller must echo `total_stake` and `total_exposure` back exactly, or nothing stakes.

| Field | Type | Notes |
| --- | --- | --- |
| `intents` | `list[PlacementIntent]` | Every bet, its stake and its price, itemised. |
| `total_stake` | `float` | The sum being risked in this batch. |
| `total_exposure` | `float` | Batch stake plus exposure already open at the venue. |
| `quoted_at` | `datetime` | |

A quote is a promise, not a reservation. Prices move, and FR-013's `min_price` is what protects
the caller between quoting and landing.

## PlacementReceipt

What actually happened (FR-016). Crosses a public boundary as a DataFrame, so it carries an
explicit `pandera` schema per Constitution Principle II.

| Field | Type | Notes |
| --- | --- | --- |
| `identity` | `BetIdentity` | |
| `status` | `PlacementStatus` | Below. |
| `stake` | `float` | What was actually staked. Zero on every refusal. |
| `price` | `float \| None` | What was actually obtained, which may be worse than the backtest price down to `min_price`. |
| `venue_bet_id` | `str \| None` | The venue's own identifier, when it gives one. |
| `value_bet` | `str` | Carried through from the intent. |
| `placed_at` | `datetime \| None` | |
| `detail` | `str` | Names the limit, the price, or the reason. Never contains a credential (FR-018). |

### PlacementStatus

| Value | Meaning |
| --- | --- |
| `DRY_RUN` | The default. Quoted, staked nothing (FR-009, SC-002). |
| `ACCEPTED` | The venue took it. |
| `MATCHED_FULL` / `MATCHED_PARTIAL` | Exchange fills. |
| `ALREADY_PLACED` | Found at the venue under this identity. Reported, not failed (FR-014a). |
| `REFUSED_LIMIT` | Would breach the stake or exposure ceiling. `detail` names which (FR-011). |
| `REFUSED_PRICE` | Below `min_price` (FR-013). |
| `REFUSED_UNCONFIRMED` | The echoed quote did not match (FR-009). |
| `REFUSED_KILLED` | The kill switch is set (FR-012). |
| `BLOCKED` | The venue blocked automation. Reported, and placement stops (FR-023). |
| `REJECTED` | The venue declined it. |

`BLOCKED` is a first-class outcome rather than an error, because FR-023's contract is that a
blocked venue is reported honestly. A receipt saying `BLOCKED` is the tool telling the truth about
a bet that did not happen.

## ExposureLimits

| Field | Type | Notes |
| --- | --- | --- |
| `max_stake_per_bet` | `float` | |
| `max_total_exposure` | `float` | Checked against a running total before each stake, which is why placement is sequential (research D7). |
| `killed` | `bool` | FR-012. Checked before every stake, so it stops a batch mid-flight. |

## CredentialRef

The name of the thing, never the thing (FR-017).

| Field | Type | Notes |
| --- | --- | --- |
| `var` | `str` | The environment variable or secret entry holding the secret. |

Resolution reads the variable at use and never returns it to a caller, never logs it, never
pickles it, and never puts it in `detail`. A missing variable names what was expected and stops
(FR-019). This mirrors the existing `odds_key_env` pattern exactly.

## VenueConfig

Not a separate type. It is the venue's constructor parameters, stored unmodified per Constitution
Principle I, and it is where every site-specific fact lives.

| Field | Type | Notes |
| --- | --- | --- |
| `key` | `str` | The user's name for the venue. |
| `url` | `str` | Which bookmaker. This is how a venue is chosen. |
| `notes` | `str \| None` | One free-text blob for the agent. URLs, layout, quirks, whatever the user learned. Stored and returned verbatim, never interpreted, so it is data rather than the model choice FR-022 forbids. |
| `credential_env` | `tuple[str, str] \| None` | Variable names (FR-017). |
| `min_interval` | `float` | Seconds between actions (research D7). |

**`notes` is one blob rather than structured URL fields, and that is a decision.** An earlier
draft had `login_url`, `betslip_url` and `history_url` as parameters. Once the library stopped
navigating to any of them, nothing parsed them, and a field nothing parses is decoration that
imposes a shape not every site has. The only reader is the agent, and the agent reads prose.

The library holds no table of sites, no selector pack, and no defaults naming a bookmaker.
Swapping Novibet for Stoiximan is an edit to the user's config. This is where FR-007's line
actually falls: shipped site knowledge rots on the venue's next deploy and we would own it, while
configured site knowledge is the user's, exactly as `param_grid` is.

The intended workflow is that the agent explores on the first run, reports what it found, and the
user pastes the durable parts into `notes`. Site knowledge accumulates in the user's config.

## FixedSession

The output of exploration and the input to placing. Navigation state, not placement state, so it
does not conflict with FR-015's ban on a local record of what was placed.

| Field | Type | Notes |
| --- | --- | --- |
| `match` | `str` | Which match this session is pinned to. |
| `url` | `str` | Where it was pinned. |
| `locators` | `dict[str, str]` | Role and accessible name pairs the agent discovered, such as `{'stake': 'textbox[name="Stake"]'}`. |

Two phases with opposite costs. Exploration is an LLM reasoning over full snapshots: slow,
token-heavy, and acceptable because it happens once. Placing runs against a moving price and
cannot stop to re-derive a layout. Pinning is what separates them, and it matters for a batch of
fifty pregame bets as much as for a live market.

**Locators, never refs.** A snapshot ref such as `[ref=e5]` is valid for one page state, and an
odds widget re-renders constantly, so a pinned ref is a stale ref. Roles and accessible names
survive a re-render.

**Never a price.** The price is read at the moment of placing and checked against `min_price`
(FR-013). Pinning it would defeat the only protection the caller has between quoting and landing.

A `FixedSession` also outlives the agent's own context, which is what degrades over a long
session.

## PageSnapshot

The site-driven read primitive (FR-007, research D5).

| Field | Type | Notes |
| --- | --- | --- |
| `yaml` | `str` | `locator.aria_snapshot(mode='ai')` output. |
| `url` | `str` | |

Refs inside the YAML are what `click`, `type` and `select` take as targets. The snapshot is scoped
by locator or depth rather than taken over `body` each call, since it is the token cost of every
agent turn.

## Entity relationships

```text
ValueBet (existing bettor output)
    │ one per row the caller chooses to act on
    ▼
PlacementIntent ──derives── BetIdentity ──hashes── ref (32 hex)
    │                                                  │
    │ batched, itemised                                │ sent as BOTH customerRef
    ▼                                                  │ and customerOrderRef
PlacementQuote ──caller echoes total_stake─────────────┤
    │            + total_exposure exactly (FR-009)     │
    ▼                                                  ▼
sequential loop, exposure checked before each ──► Venue ──► PlacementReceipt
                                                    │
                                              the record (FR-015)
                                              no local copy exists
```
