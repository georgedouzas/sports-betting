# Design

## Context

A fitted bettor produces value bets through `bettor.bet(X_fix, O_fix)` and nothing acts on them. This change adds a
`sportsbet.execution` module, behind an optional extra, that consumes those value bets and places them at a venue.
Bettors stay pure estimators, so a reloaded pickle cannot spend money.

Two venue families answer to one goal. A venue with an official betting API implements `BaseVenue`, and the library
places and enforces every rail. A bookmaker with no API is reached through `BrowserSession`, which exposes generic
navigate, read and act primitives, and the agent supplies the site knowledge and does the placing. Every capability is
reachable from the Python API, the CLI and the MCP server.

After implementation the maintainer reversed one decision. `BetfairVenue` was removed, and a venue with an API is user
provided as `venue.py:VENUE`, like a model. The Betfair analysis below is kept as the rationale for the shape of the
venue contract, not as a shipped adapter.

Technical context from the plan: Python `>=3.11, <3.14`, full annotations. `aiohttp`, `pandas`, `pandera` and `click`
were already core. Storage is none, by design. The one exception is a Playwright `user_data_dir` that holds the
bookmaker login, which is navigation state and not placement state. Tests use `pytest` with branch coverage,
`pytest-randomly`, `--doctest-modules` and the socket guard in `tests/conftest.py`, which permits loopback.
Network-touching classes carry non-executable examples. The CI matrix is Linux, macOS and Windows.

## Goals / Non-Goals

**Goals:**

- Close the loop from a value bet to a placed bet, with refusal as the default.
- Place one stake per selection per venue, whatever retries, crashes or fresh runs occur.
- Keep the venue as the only record of what was placed.
- Keep secrets out of every argument, log, error, file and transcript.
- Reach every execution capability from the Python API, the CLI and the MCP server.

**Non-Goals:**

- Shipping an adapter for any specific bookmaker or exchange, or per-site browser code.
- An agent loop, a model call, a model key or a model choice inside the package.
- Any means of evading a venue's automation controls.
- Opening accounts, depositing or withdrawing.
- Bankroll strategy. The stake is an input.
- A model registry or an MLOps lifecycle.
- In-play betting. Bettors are fitted on pregame features and odds, so an in-play stake would trace back to a value bet
  that stopped being true at kickoff. In-play needs an in-play model, which is a modelling feature.
- Any change to `src/sportsbet/dataloaders` or `src/sportsbet/evaluation`.

## Decisions

### Clarifications from the spec session of 2026-07-16

- Clarified: for a bookmaker with no API, where does the site-specific knowledge live? -> The library exposes generic
  browser and page capabilities and the agent supplies the site knowledge. No per-site adapters ship.
- Clarified: who may authorise real money, and how? -> Any caller including an agent, but only by passing back the
  exact stake and total exposure just quoted, as feature 005 does with `confirm_cost`. Human-set limits still cap it.
- Clarified: where does the durable record of what was placed live? -> The venue. The system attaches its own
  reference to each bet and reads it back to dedupe and reconcile. No local or hidden store.
- Clarified: what makes two bets the same bet for deduplication? -> The venue, the match, the market and the selection.
  One stake per selection per venue, independent of which run produced it.
- Clarified: how does the caller express the worst acceptable price? -> An absolute minimum price per bet, defaulting
  to the price the value bet was computed at.

### D1: The reference venue is Betfair

Betfair Exchange shapes the reference venue contract. It is the only exchange of the four surveyed whose place-order
call carries a caller-supplied reference that the venue stores, returns and filters on server side. FR-015 makes the
venue the record of what was placed, so the requirement is unimplementable at a venue that cannot carry the reference.
Betfair also permits automated betting in its general terms (18.11 "Bots", with 12.1.4 carving API bets out of
"suspicious betting"), publishes a machine-readable interface definition, and has a maintained Python client
(`betfairlightweight` 2.23.2, released 2026-03-16).

Alternatives considered:

- Smarkets. Rejected. The order body is `market_id, contract_id, price, quantity, side, label,
  minimum_accepted_quantity, type`. `label` is a strategy tag with no deduplication and no server-side filter. No caller
  reference exists. Its Python SDK was archived in 2019.
- Matchbook. Rejected. Offer fields are exactly `runner-id, side, odds, stake, keep-in-play`. No caller reference field
  exists. The community Python client died in 2018 and the official SDK is Java.
- Betdaq. Rejected. `PunterReferenceNumber` looks like a caller reference and is not one. It is an `xs:long`, so it
  cannot hold a UUID. The spec states it "does not need to be unique", and no endpoint filters on it.
  `ListOrdersChangedSince` and `GetOrderDetails` take only `sequenceNumber` and `handle`. It is a correlation ID.

A contract designed against any of the three would promise what the venue cannot honour, and the bug would surface as a
double stake rather than as a type error.

Reversed after implementation: `BetfairVenue` was removed. FR-006 now forbids shipping any bookmaker. The contract is
proved against a fake.

### D2: No venue offers a placement sandbox

The sanctioned path is proved against fakes, recorded responses and contract tests. No test touches a venue. This
falsified the spec's assumption that the reference exchange offers a sandbox. Betfair's delayed application key is
widely described as a sandbox and is not one. Betfair's documentation states it "operates on the live (production)
Betfair Exchange and not a testbed/sandbox environment", and its capability matrix lists Bet Placement (Live Exchange)
as available. FR-026 already forbade touching a real venue. The reason is now written down so nobody relaxes it.

Alternative considered: Betdaq's `RC598 MarketIsForPlayMoney` return code implies play-money markets, but nothing
documents how an external API customer reaches one, and Betdaq is rejected on D1 grounds anyway.

### D3: Idempotency needs two references

The venue contract carries two references. `dedupe_ref` maps to Betfair's `customerRef`. `order_ref` maps to
`customerOrderRef`. Both are sent on every placement.

| | `customerRef` | `customerOrderRef` |
| --- | --- | --- |
| Scope | per request | per instruction |
| Limit | 32 chars, charset `A-Za-z0-9 : - . _ + * ; ~` | 32 chars |
| Deduplicates | yes, within a 60 second window | no, "no validation will be done on uniqueness" |
| Readable back | no | yes, via the stream and `listCurrentOrders` |
| Filterable | no | yes, `listCurrentOrders(customerOrderRefs=[...])` |

Neither field alone satisfies FR-014. `customerRef` enforces once-only but expires after 60 seconds and never appears
in the response. `customerOrderRef` answers "did this already happen" durably but enforces nothing. A single
`client_ref` abstraction would leave a 60 second hole in the guarantee, found only in production under a retry.

The retry contract is explicit and pessimistic. Within 60 seconds, `customerRef` covers resubmission. Past it,
recovery is a `listCurrentOrders` lookup filtered on `order_ref`. A colliding `order_ref` is our defect to prevent.

Unresolved and treated pessimistically: Betfair's `SportsAPING.xml` defines `PlaceExecutionReport.customerRef` as
"Echo of the customerRef if passed", while the current documentation says the field does not persist into the
response. The adapter treats `customerRef` as unreadable and reconciles only on `order_ref`. The maximum number of
`customerOrderRefs` per `listCurrentOrders` call is undocumented. The stated cap of 250 applies to `betIds` and
`marketIds`, so batch reconciliation chunks conservatively.

### D4: The browser layer is Playwright, used directly

Playwright drives the site-driven path. crawl4ai is rejected, for three reasons in descending weight.

First, Playwright refuses to click what cannot be clicked. crawl4ai's only interaction primitive is `js_code`, which
injects arbitrary JavaScript. Against a mock bet slip with a disabled confirm button, Playwright reported
`Locator.click: Timeout 1200ms exceeded` and the raw JavaScript route reported success on the disabled Place-bet
button. On a control that moves real money, a false success is the worst failure, and it is FR-016's receipt lying.

Second, crawl4ai installs `patchright`, `playwright-stealth` and `fake-useragent` unconditionally. Its evasion switches
are off by default (`enable_stealth=False`, `magic=False`, `simulate_user=False`, `override_navigator=False`), but the
undetected-browser fork ships to every user. `magic=True` is an evasion bundle whose docstring advertises overlay
handling. Its documentation states it "may enable stealth mode and undetected browser by default" in future.

Third, weight and supply chain. Playwright is three packages against 55. crawl4ai pulls `openai`, `tiktoken`,
`tokenizers`, `huggingface-hub`, `scipy`, `networkx`, `shapely`, `rtree` and `trimesh`, and pins `unclecode-litellm`,
whose PyPI summary reads "Pre-compromise fork of litellm". Playwright is Apache-2.0, supports 3.11 through 3.14, and its
headless shell download measured 93.5 MiB. Its maintainer stated in playwright-python#527 that the project gives no
advice on hiding automation from servers.

Alternatives considered:

- Patchright. Disqualified. Evasion is the entire product.
- browser-use. Rejected. Its `Agent` always instantiates an LLM, falling back to `ChatBrowserUse()`, which is the
  in-package agent loop FR-022 forbids. It hard-pins 35 dependencies with `==`, including `anthropic`, `openai`,
  `google-genai`, `groq` and `ollama`, which do not resolve against this project.
- Steel. Rejected as unverified and disproportionate. The self-hosted server lists stealth plugins and fingerprint
  management as a core feature, whether the Docker image applies stealth by default was not established, and it
  requires a Docker server.

### D5: The page is read as an AI-mode ARIA snapshot

The read primitive returns `locator.aria_snapshot(mode='ai')`. Markdown lets the agent read a page but discards element
identity, so the agent cannot act on what it read. AI-mode ARIA snapshots carry stable refs, for example
`button "Place bet" [ref=e5]`, so one artifact closes the navigate, read and act loop. `get_by_role` and
`get_by_label` pair with the roles and names it emits.

Two corrections to widely cited material: `page.accessibility.snapshot()` was removed in Playwright 1.57, and plain
`aria_snapshot()` without `mode='ai'` omits the refs.

Alternative considered: markdown alongside snapshots through `markdownify` (MIT) or `trafilatura` (Apache-2.0) stays
open for cheaper odds reading. `html2text` is excluded permanently because it is GPL-3.0-or-later.

### D6: One browser context, held by the process

The MCP server holds one live `BrowserContext` for its lifetime, created with
`launch_persistent_context(user_data_dir=...)`. Tool calls act on that context, so a login outlives any single call and
survives restarts. Playwright allows one browser instance per `user_data_dir`, and its API is not thread-safe, so
`start()` and `stop()` are managed explicitly instead of the `with sync_playwright()` idiom. One context per process
makes concurrent watchers unreachable.

### D7: Placement is sequential

A batch places one bet at a time and checks the running exposure against the limit before each stake. A configurable
minimum interval in seconds separates actions. FR-011 needs a running total that concurrent stakes make unreadable
without a lock, and FR-014 allows one stake per identity, which concurrent retries break. Not adopted: behavioural
mimicry whose unit is human-likeness, such as randomised delays, mouse curves or typing cadence. FR-023 covers this.

### D8: Geography decides which adapter a user can reach

Both adapter families were planned to ship. All four surveyed exchanges are closed to Greece, verified from the
maintainer's connection. Smarkets, Matchbook and Betdaq resolve to `62.74.31.13`, the Hellenic Gaming Commission block
server, and all three appear on the EEEP blacklist (52nd edition, 12.06.2026). `betfair.com` resolves normally and is
absent from the blacklist, yet Betfair blocks Greece itself, redirecting to `/gr` with a page naming `Region: GR`. It
withdrew from Greece in 2012 over licensing. User Story 1 is the testable contract other users transact against, and
User Story 3 is the path the maintainer can use. A VPN is not a workaround. It breaches Betfair T&C 4.1.6, the stated
penalty is confiscation of the balance, and geographic controls are automation controls under FR-023.

Unverified and recorded as such: the block page also mentions unusual network traffic, and no test was run from a
non-Greek IP.

### D9: The no-evasion boundary

FR-023 stands. No stealth browsing, fingerprint spoofing, captcha solving, proxy rotation or geographic circumvention.
A blocked venue is reported and placement stops. The boundary separates automating an account the user holds from
defeating a control the venue built to deny that access. Evasion is also untestable under FR-026 and degrades with
every countermeasure. Proxy rotation achieves nothing on an authenticated session and makes the account look shared or
compromised. Practical consequence: bet365 is likely to block an honest automated browser. Novibet and Stoiximan are
the plausible site-driven targets. The system claims no site works.

### D10: FR-005 and FR-007 conflict, and the contract resolves it

FR-005 requires every venue to implement `place`. FR-007 requires the site-driven path to be generic primitives with
the agent supplying site knowledge. A `BrowserVenue.place(intent)` would have to find the market, click the price,
locate the stake field and work the confirm flow, and every step is site knowledge. That is per-site code, which FR-007
forbids, or an LLM in the package reading `notes`, which FR-022 forbids. `BrowserSession` is therefore not a
`BaseVenue` and offers no `place`, `read_status` or `cancel`. The agent places.

Two guarantees hold on the API path and not on the site path. Once-only needs the venue's bet history read back, which
needs site knowledge. A generic text match of identity fields against a history snapshot was considered and rejected,
because a false negative is the double stake FR-014 exists to prevent. The stake and exposure ceilings bind the caller
that places, which on the site path is the agent, and the operative control is that the user sees and approves the
agent's tool calls. Cancellation on the site path is also the agent's. The documentation states this ahead of any
instruction to use the path.

### D11: Explore, fix, then bet

Exploration is an LLM reasoning over full snapshots, slow and token-heavy, and it happens once. Placing runs against a
moving price and cannot re-derive a layout per bet. `fix(match, locators)` pins the role and accessible name pairs the
agent discovered into a `FixedSession`. It stores locators, never refs, because a ref such as `[ref=e5]` is valid for
one page state and an odds widget re-renders constantly. It never stores a price, because the price is read at placing
and checked against `min_price`. A `FixedSession` is navigation state, so it does not conflict with FR-015, and it
outlives the agent's own context.

### D12: The data model

- `BetIdentity`: `venue`, `match`, `market`, `selection`. The derived `ref` is
  `blake2s(f'{venue}|{match}|{market}|{selection}', digest_size=16).hexdigest()`, exactly 32 hex characters, inside
  Betfair's 32 character limit and charset. The digest is deterministic, so a fresh process recomputes the same
  reference with no stored state. With one instruction per request, the request and instruction scopes coincide and
  both Betfair references take the same value. A retry within 60 seconds is deduplicated on `customerRef`. A retry
  after 60 seconds, a crash or a fresh run finds the bet through `listCurrentOrders(customerOrderRefs=[ref])` and
  reports already placed. A venue that carries no reference recognises the bet from its own history by the four fields.
- `PlacementIntent`: `identity`, `stake` (an input, never a model output), `min_price` (defaults to the price the value
  bet was computed at), `value_bet` (a back reference to the row that caused it). Immutable.
- `PlacementQuote`: `intents`, `total_stake`, `total_exposure` (batch stake plus exposure already open at the venue),
  `quoted_at`. A quote is a promise, not a reservation.
- `PlacementReceipt`: `identity`, `status`, `stake` (zero on every refusal), `price` (what was obtained, possibly worse
  than the backtest price down to `min_price`), `venue_bet_id`, `value_bet`, `placed_at`, `detail` (names the limit,
  the price or the reason, never a credential). It crosses a public boundary as a DataFrame with a `pandera` schema.
- `PlacementStatus`: `DRY_RUN`, `ACCEPTED`, `MATCHED_FULL`, `MATCHED_PARTIAL`, `ALREADY_PLACED`, `REFUSED_LIMIT`,
  `REFUSED_PRICE`, `REFUSED_UNCONFIRMED`, `REFUSED_KILLED`, `BLOCKED`, `REJECTED`. `BLOCKED` is an outcome rather than
  an error, because a blocked venue is reported honestly.
- `ExposureLimits`: `max_stake_per_bet`, `max_total_exposure`, `killed`. The kill switch is checked before every stake.
- `CredentialRef`: `var`, the name of the variable. Resolution reads it at use and never returns, logs, pickles or
  reports it. This mirrors the existing `odds_key_env` pattern.
- Venue configuration is the venue's constructor parameters, stored unmodified: `key`, `url`, `notes`,
  `credential_env` and `min_interval`. `notes` is one free-text blob for the agent, stored and returned verbatim and
  never interpreted. An earlier draft had `login_url`, `betslip_url` and `history_url`. Once the library stopped
  navigating to them, nothing parsed them, so they became one blob. The library holds no table of sites, no selector
  pack and no default naming a bookmaker.
- `FixedSession`: `match`, `url`, `locators`.
- `PageSnapshot`: `yaml` (the AI-mode ARIA snapshot) and `url`. Snapshots are scoped by locator or depth, since a
  snapshot is the token cost of every agent turn.

### D13: The venue contract and the placing entry point

`BaseVenue` declares `key`, `can_cancel`, and the coroutines `authenticate(credential)`, `list_markets(matches)`,
`read_balance()` returning balance and open exposure, `place(intent)`, `read_status(identities)` and
`cancel(identity)`. Obligations on every implementation: `place` is idempotent on `identity.ref`, and a venue that
cannot guarantee that does not implement the contract. `place` stakes nothing on its own authority and is reached only
through the module's `place` entry point. `cancel` raises `CancellationUnsupported` where the venue cannot cancel. A
blocked venue returns `BLOCKED` and stops without varying its timing or presentation. No method takes a secret.

The removed `BetfairVenue` took `*_env` variable names with name defaults, optional certificate paths, the API and login
URLs, and `min_interval=0.2`. It mapped `authenticate` to the certificate login, `list_markets` to
`listMarketCatalogue` plus `listMarketBook`, `read_balance` to `getAccountFunds`, `place` to `placeOrders`,
`read_status` to `listCurrentOrders` and `cancel` to `cancelOrders`. Betfair's limits are 1000 transactions per second
overall and 5 per second per market. It was built on the existing `aiohttp` rather than `betfairlightweight`, because
five endpoints and a certificate login did not justify a client library.

`BrowserSession(key, url, *, notes=None, credential_env=None, user_data_dir=None, min_interval=1.0)` exposes
`navigate(url)`, `snapshot(selector=None)`, `click(ref)`, `type(ref, text)` and `select(ref, value)`, each returning
the resulting `PageSnapshot`, plus `fix(match, locators)`.

`quote(venue, intents, limits)` returns a `PlacementQuote`.
`place(venue, quote, limits, confirm_stake=None, confirm_exposure=None)` returns the receipts frame. The refusal order
before any venue call is: the kill switch, so every receipt is `REFUSED_KILLED`. Then the confirmation, so a missing or
mismatched figure makes every receipt `REFUSED_UNCONFIRMED` and the message states the real figures. Then, per intent
and in order, the kill switch, the ceilings, `min_price`, and the venue call. A caller that passes no confirmation gets
the full quote and zero stakes, which is `DRY_RUN`.

### D14: The surfaces

| Capability | Python API | CLI | MCP tool |
| --- | --- | --- | --- |
| Read the venue config | `venue.key`, `venue.notes` | `execution venue` | `execution_venue_info` |
| Authenticate | `venue.authenticate(cred)` | implicit in every command | `execution_authenticate` |
| Read markets | `venue.list_markets(matches)` | `execution markets` | `execution_list_markets` |
| Read balance | `venue.read_balance()` | `execution balance` | `execution_read_balance` |
| Quote a batch | `quote(venue, intents, limits)` | `execution quote` | `execution_quote` |
| Place a batch | `place(venue, quote, limits, ...)` | `execution place` | `execution_place` |
| Read status | `venue.read_status(ids)` | `execution status` | `execution_read_status` |
| Cancel | `venue.cancel(id)` | `execution cancel` | `execution_cancel` |
| Read a page | `session.navigate/snapshot` | `execution page read` | `browser_navigate`, `browser_snapshot` |
| Act on a page | `session.click/type/select` | `execution page act` | `browser_click/type/select` |
| Pin a session | `session.fix(match, locators)` | `execution page fix` | `browser_fix` |

A venue is named the way a model is named. A dataloader is a sport and a short closed list of names, so it fits in
arguments. A venue carries a URL, notes and certificate paths, so `build_venue` sits beside `build_bettor` and resolves
`venue.py:VENUE` through the existing `_load_object`. It imports `sportsbet.execution` lazily and names the extra to
install when it is missing. The plan also resolved a ready-made `--venue betfair`, which went with `BetfairVenue`.

CLI commands take every parameter with `--`, with no positional arguments. `quote` writes the itemised batch and its
totals to `--output`. `place` reads it back with `--quote` and takes `--confirm-stake`, `--confirm-exposure`,
`--max-stake` and `--max-exposure`. There is no `--dry-run` flag. Dry run is the absence of confirmation, so no default
can be misconfigured into spending. `place` without confirmations prints the quote, stakes nothing and exits non-zero.
The credential is named by the venue and never passed on the command line, because a CLI option is a shell history
entry.

MCP tools mirror the CLI. `execution_place` enforces the confirmation in code, following the `prepare` and
`confirm_cost` pattern of feature 005. `execution_venue_info(venue)` returns `key`, `url`, `notes` and `can_cancel`, and
it is how the user's site knowledge reaches the agent. DataFrames cross as records. `execution` and `mcp` are separate
extras and the tools need both. Missing either produces a message naming the extra.

### D15: Structure

`sportsbet/execution/` is a peer of `cli/` and `mcp/` and imports only the public API, which keeps the zero diff on the
core structural. The CLI group is a new `cli/_execution.py` rather than an addition to `_betting.py`, because execution
is a separate group in the Python API. Test files are flat, matching `tests/test_mcp.py`.

### Constitution check

Checked against constitution v1.1.0 before and after design. Principle I passes through FR-001, FR-002, FR-020 and
FR-022. Principle II passes with the obligation of a `pandera` schema on the receipt frame. Principle III passes
through fakes, recorded payloads and a loopback page. Principle IV passes with a watch item: the bandit skips `B404`,
`B603` and `B607` were removed when the GUI went, Playwright launches its driver internally, and any new bandit finding
is reported rather than skipped. Principle V passes with the obligation of an execution page that leads with the risks.
The tooling rule that a credentialled or side-effecting capability lives behind an optional extra is met by FR-004.

## Risks / Trade-offs

- Driving a bookmaker's website breaches essentially every bookmaker's terms of service and risks account closure and
  loss of the balance. The documentation states this ahead of any instruction to use it.
- Once-only and the stake and exposure ceilings are not enforced by the library on the site-driven path. The agent
  holds them, and the user approves the agent's tool calls.
- No sandbox exists. A contributor who believes the delayed-key folklore spends real money. Tests use fakes only.
- Betfair's documentation contradicts its interface definition on whether `customerRef` is readable back. The design
  treats it as unreadable.
- Site markup changes break the agent's knowledge, not the library, because the library ships none.
- A venue that blocks automation ends the session. The library does not work around it.
- Sequential placement bounds throughput on purpose.

## Validation scenarios

From the quickstart. Every scenario runs without a bookmaker account, against a fake venue, recorded payloads or a
loopback page. Prerequisites: `pdm install -dG maintenance -dG tests -G mcp -G execution` and, for browser scenarios
only, `pdm run python -m playwright install --only-shell chromium`.

1. The default refuses (SC-002, FR-009). `sportsbet execution place --venue tests/fake_venue.py:VENUE --quote
   quote.json` prints the itemised quote, `DRY_RUN` on every receipt, zero stakes and a non-zero exit. Wrong figures
   give `REFUSED_UNCONFIRMED` and the real figures.
2. Nothing is hidden before money moves (SC-008). Every placed row traces to a quoted row.
3. Limits refuse and name themselves (SC-009). The kill switch mid-batch leaves the placed prefix intact.
4. Once and only once under faults (SC-003, FR-014). A timeout after accept and a crash before receipt both end with
   one stake and `ALREADY_PLACED` on the second attempt, with no placement state on disk. This covers the API path
   only. On the site path the test asserts that `BrowserSession` exposes no `place`, `read_status` or `cancel`.
5. Credentials stay named (SC-004, FR-017). A sentinel secret appears in no output, log, exception, file or pickle. An
   unset variable is named and the run stops.
6. The browser refuses what it cannot click. A disabled confirm button raises on `click`, and re-enabling it shows the
   new state.
7. A fixed session survives a re-render, `fix` rejects a raw ref, and no price is stored.
8. A blocked venue reports `BLOCKED` and stops with no retry under different timing, headers or presentation.
9. Parity across the three surfaces through `tests/cli/test_parity.py` (SC-006).
10. Without the extra there is nothing to place with (SC-010).
11. The suite touches no venue (SC-007, FR-026).

The gate is `pdm run formatting`, `pdm run checks` and `pdm run tests`, all green on every phase, and
`git diff --stat -- src/sportsbet/dataloaders src/sportsbet/evaluation` is empty.

## Open Questions

- In-play was recorded as open in the plan rather than decided. The session design serves live markets and pregame
  batches equally, but a pregame model cannot know its edge evaporated at kickoff. The plan recommended keeping in-play
  out of scope for this change. An agent deciding in-play with no backtest behind it was buildable but not adopted.

## Quality checks

The specification quality checklist passed every item. Its review notes record:

- No venue, SDK, browser driver or library was named in the spec at authoring time. Research settled Betfair and
  Playwright.
- The Assumptions section names Betfair after planning, judged to pass because planning falsified two assumptions.
- `sportsbet.execution`, the bettor and the three surfaces are domain vocabulary for this library, not technology
  choices.
- Five clarifications resolved the site-driven scope (FR-007), who authorises money (FR-009), where the record lives
  (FR-015), what makes two bets the same (FR-014) and the worst acceptable price (FR-013).
- Falsified: the reference exchange offers a sandbox. Falsified: the sanctioned path is the one the maintainer uses.
- The FR-005 and FR-007 contradiction was flagged for a reviewer before tasks were written. The task list adopted the
  resolution in D10. The note stated that rejecting it would have required a change to FR-007.
- The money-safety requirements FR-009 to FR-016 and the boundary requirements FR-023 and FR-024 are the ones most
  worth a reviewer's attention.
