# Contract: The venue

**Feature**: [spec.md](../spec.md) | **Data model**: [data-model.md](../data-model.md)

Two shapes, not one, and planning found that the spec assumed one.

`BaseVenue` (FR-005) is implemented by a venue with an official API, where the library does the
placing and can therefore enforce every rail. Betfair implements it.

`BrowserSession` (FR-007) is not a venue and does not implement `BaseVenue`. It exposes primitives
and the agent does the placing. The reasoning is under its own heading below, and it changes what
the site-driven path can promise.

## `BaseVenue`

```python
class BaseVenue(abc.ABC):
    """A place where a user holds an account and can back a selection."""

    key: str
    can_cancel: bool

    @abc.abstractmethod
    async def authenticate(self, credential: CredentialRef) -> None:
        """Authenticate, reading the secret from the named variable."""

    @abc.abstractmethod
    async def list_markets(self, matches: list[str]) -> pd.DataFrame:
        """Return the markets on offer for the given matches, with current prices."""

    @abc.abstractmethod
    async def read_balance(self) -> tuple[float, float]:
        """Return the balance and the exposure currently open."""

    @abc.abstractmethod
    async def place(self, intent: PlacementIntent) -> PlacementReceipt:
        """Place one bet. Idempotent on `intent.identity.ref`."""

    @abc.abstractmethod
    async def read_status(self, identities: list[BetIdentity]) -> pd.DataFrame:
        """Return what the venue holds for these identities."""

    @abc.abstractmethod
    async def cancel(self, identity: BetIdentity) -> PlacementReceipt:
        """Cancel a bet. Raises `CancellationUnsupported` where the venue cannot."""
```

## Obligations on every implementation

**`place` is idempotent on `identity.ref`, and this is the load-bearing one.** Calling it twice
with the same identity results in one stake. An implementation that cannot guarantee that must
return `ALREADY_PLACED` from a read rather than risk a second stake. Betfair achieves this by
sending `ref` as both `customerRef` and `customerOrderRef` (data-model). A venue that cannot make
this guarantee does not implement this contract, which is why `BrowserSession` does not.

**`place` stakes nothing on its own authority.** It is reached only through the module's `place`
entry point, which holds the quote confirmation, the limits and the kill switch. A venue adapter
is a driver, not a decision maker.

**`cancel` refuses honestly** (FR-008). `can_cancel` is declared, and a venue that cannot cancel
raises rather than returning a receipt that implies it did something.

**A blocked venue returns `BLOCKED` and stops** (FR-023). It does not retry differently, vary its
timing, or change how it presents itself.

**No method takes a secret.** `authenticate` takes the *name* of the variable (FR-017). No method
puts a credential in a return value, an exception, or a log line (FR-018).

## `BetfairVenue`

`key = 'betfair'`, `can_cancel = True`. Built on the existing `aiohttp`, so it adds no dependency.

Parameterized on the same principle, since the endpoints differ by jurisdiction and the
credential material is per user.

```python
class BetfairVenue(BaseVenue):
    def __init__(
        self,
        *,
        app_key_env: str = 'BETFAIR_APP_KEY',
        username_env: str = 'BETFAIR_USERNAME',
        password_env: str = 'BETFAIR_PASSWORD',
        cert_path: str | Path | None = None,
        cert_key_path: str | Path | None = None,
        api_url: str = 'https://api.betfair.com/exchange/betting/json-rpc/v1',
        login_url: str = 'https://identitysso-cert.betfair.com/api/certlogin',
        min_interval: float = 0.2,
    ) -> None:
```

The `*_env` parameters carry variable names, never secrets (FR-017). Their defaults are names, not
values, so nothing sensitive has a default.

| Contract method | Betfair call |
| --- | --- |
| `authenticate` | Certificate login at `identitysso-cert.betfair.com/api/certlogin`, then `X-Application` plus `X-Authentication` on every call |
| `list_markets` | `listMarketCatalogue` plus `listMarketBook` |
| `read_balance` | `getAccountFunds` |
| `place` | `placeOrders`, one instruction per request, `customerRef` and `customerOrderRef` both set to `identity.ref` |
| `read_status` | `listCurrentOrders(customerOrderRefs=[...])` |
| `cancel` | `cancelOrders` |

Known limits to respect: 1000 transactions/sec overall, 5/sec per market ID. Sequential placement
sits far below both. The maximum number of `customerOrderRefs` per `listCurrentOrders` call is
undocumented (the stated 250 applies to `betIds` and `marketIds`), so batch reconciliation chunks
conservatively rather than assuming it generalises.

Two facts a contributor will otherwise get wrong, both from research:

- **The delayed application key is not a sandbox.** It places real bets on the live exchange. There
  is no venue anywhere to integration-test against. Tests use recorded payloads (FR-026).
- **`customerRef` is not readable back.** Betfair's own interface definition and its documentation
  contradict each other on this. The adapter treats it as unreadable and reconciles only on
  `customerOrderRef`.

## `BrowserSession`

**Not a `BaseVenue`, and this is the correction that matters.**

Planning found FR-005 and FR-007 in direct contradiction. FR-005 requires every venue to implement
`place`. FR-007 requires the site-driven path to expose generic primitives while the agent supplies
the site knowledge. Ask what `BrowserVenue.place(intent)` would have to do: find the market for
this match, click the right price, locate the stake field, work the confirm flow. Every step is
site knowledge. So it is either per-site code, which FR-007 forbids, or an LLM inside the package
interpreting `notes`, which FR-022 forbids. The method cannot exist.

The site-driven path is therefore a session exposing primitives, and **the agent is what places**.

```python
class BrowserSession:
    def __init__(
        self,
        key: str,
        url: str,
        *,
        notes: str | None = None,
        credential_env: tuple[str, str] | None = None,
        user_data_dir: str | Path | None = None,
        min_interval: float = 1.0,
    ) -> None:
```

Constructor parameters are stored unmodified and unvalidated, per Constitution Principle I.

**`notes` is one free-text blob, deliberately, and not a set of structured URL fields.** The
earlier draft had `login_url`, `betslip_url` and `history_url` as separate parameters. Once the
library stopped navigating to any of them, no code parsed them, and a field nothing parses is
decoration that implies a shape not every site has. The only reader is the agent, and the agent
reads prose.

```python
VENUE = BrowserSession(
    key='novibet',
    url='https://www.novibet.gr/stoixima',
    notes="""
    Bet history is under My Account > Bet History.
    The slip opens on the right after clicking a price; confirm is two steps.
    Check the history for this match before placing, so a retry does not stake twice.
    """,
    credential_env=('NOVIBET_USERNAME', 'NOVIBET_PASSWORD'),
)
```

The package stores `notes` and returns it verbatim through `execution_venue_info` and never reads
it, so it is data rather than the model choice FR-022 forbids. This is the line FR-007 actually
draws. Shipping selectors for Novibet would be a per-site adapter we own and that breaks on their
next deploy. A user writing prose into a constructor is configuration they own, exactly as
`param_grid` is.

The intended workflow is that the agent explores on the first run, reports the layout, and the
user pastes what is durable into `notes`. The knowledge accumulates in the user's config rather
than in this repository.

### Primitives

```python
async def navigate(self, url: str) -> PageSnapshot: ...
async def snapshot(self, selector: str | None = None) -> PageSnapshot: ...
async def click(self, ref: str) -> PageSnapshot: ...
async def type(self, ref: str, text: str) -> PageSnapshot: ...
async def select(self, ref: str, value: str) -> PageSnapshot: ...
```

Each act returns the resulting snapshot, so the agent sees what its action did without a second
call.

**Actionability is a safety property, not a convenience.** Playwright refuses to click hidden or
disabled elements, and this is exactly why it was chosen over injected JavaScript, which reported
success on a disabled Place-bet button under test (research D4). A confirm button the site has
disabled means the site is not ready to take the bet, and the tool must fail rather than pretend.

Session: one `BrowserContext` per process via `launch_persistent_context(user_data_dir=...)`, held
across tool calls so a login survives them (research D6). One instance per `user_data_dir`, and
the API is not thread-safe, so `start()` and `stop()` are managed explicitly.

Pacing: one action at a time, with a configurable minimum interval in seconds. Concurrency is not
reachable, since the process holds one context.

### Explore, fix, then bet

The session has two phases, and they have opposite costs. Exploration is an LLM reasoning over
full snapshots: seconds a turn, expensive in tokens, and fine, because it happens once. Placing
runs against a moving price and must not stop to reason about layout. A single loop cannot be both.

```python
async def fix(self, match: str, locators: dict[str, str]) -> FixedSession: ...
```

Phase one, the agent explores: authenticate, navigate, snapshot, work out where the market, the
stake field and the confirm control are. Phase two, it calls `fix` with what it found, and the
session is pinned. Placement then resolves the pinned locators per action without re-deriving the
layout. This matters for a batch of fifty pregame bets as much as for a live market, since nothing
should work out where the stake box is fifty times.

**A fix stores locators, never refs.** Snapshot refs such as `[ref=e5]` are valid for one page
state, and an in-play widget re-renders constantly, so a pinned ref is a stale ref. What survives
a re-render is the role and the accessible name the agent discovered, which is what `fix` takes.

**A fix never pins a price.** The price is read at the moment of placing and checked against
`min_price` (FR-013). Pinning it would defeat the one protection the caller has between quoting
and landing.

A `FixedSession` is navigation state, not placement state, so it does not conflict with FR-015's
ban on a local record of what was placed. It also outlives the agent's own context, which is the
thing that degrades over a long session.

### What this path does not guarantee, stated plainly

Two guarantees hold on the API path and do not hold here. The documentation says so, ahead of any
instruction to use it, rather than letting a user infer that the rails are the same.

**Once-only is the agent's, not the library's.** Detecting an already-placed bet means reading the
venue's bet history, which means knowing how that site renders history, which is the knowledge
this path exists to not have. A generic text match of the identity fields against a history
snapshot was considered and rejected: a false negative produces a double stake, which is precisely
the failure FR-014 exists to prevent, so a crude match is worse than an honest gap. FR-014's hard
guarantee holds where the venue carries the reference, which is Betfair.

**The stake and exposure ceilings are advisory here.** When the library places, it enforces them
because it is the caller. When the agent clicks, the library supplies `click` and `type` and cannot
know which field was the stake. The operative control on this path is that the user sees the
agent's tool calls and approves them.

Cancellation is likewise the agent's, by driving the site.

```python
async def navigate(self, url: str) -> PageSnapshot: ...
async def snapshot(self, selector: str | None = None) -> PageSnapshot: ...
async def click(self, ref: str) -> PageSnapshot: ...
async def type(self, ref: str, text: str) -> PageSnapshot: ...
async def select(self, ref: str, value: str) -> PageSnapshot: ...
```

Each act returns the resulting snapshot, so the agent sees what its action did without a second
call.

**Actionability is a safety property, not a convenience.** Playwright refuses to click hidden or
disabled elements, and this is exactly why it was chosen over injected JavaScript, which reported
success on a disabled Place-bet button under test (research D4). A confirm button the site has
disabled means the site is not ready to take the bet, and the tool must fail rather than pretend.

Session: one `BrowserContext` per process via `launch_persistent_context(user_data_dir=...)`, held
across tool calls so a login survives them (research D6). One instance per `user_data_dir`, and
the API is not thread-safe, so `start()` and `stop()` are managed explicitly.

Pacing: one action at a time, with a configurable minimum interval in seconds. Concurrency is not
reachable, since the process holds one context.

## Module entry point

```python
async def quote(venue, intents, limits) -> PlacementQuote: ...

async def place(
    venue: BaseVenue,
    quote: PlacementQuote,
    limits: ExposureLimits,
    confirm_stake: float | None = None,
    confirm_exposure: float | None = None,
) -> pd.DataFrame:
    """Place a quoted batch. Stakes nothing unless both confirmations match the quote exactly."""
```

The refusal order, before any venue call:

1. `limits.killed` is set, every receipt is `REFUSED_KILLED` (FR-012)
2. `confirm_stake` or `confirm_exposure` is missing or mismatched, every receipt is
   `REFUSED_UNCONFIRMED` and the message states the real figures (FR-009)
3. Otherwise, sequentially per intent: kill switch, then limits, then `min_price`, then place

A caller that passes no confirmation gets a full quote and zero stakes. That is `DRY_RUN`, and it
is what happens by default (SC-002).
