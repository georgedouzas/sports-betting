# Phase 0 Research: Bet execution

**Feature**: [spec.md](./spec.md) | **Date**: 2026-07-16

Nine decisions. Two of them falsified an assumption the spec was resting on, and both
are recorded here in full because the spec has since been corrected against them.

## D1: The reference venue is Betfair

**Decision**: Betfair Exchange implements the reference venue adapter.

**Rationale**: It is the only exchange of the four surveyed whose place-order call carries a
caller-supplied reference that the venue stores, returns, and filters on server side.
FR-015 makes the venue the record of what was placed, and decides "already placed" by reading
the identity back from the venue. That requirement is unimplementable at a venue that cannot
carry the reference. Betfair also permits automated betting explicitly in its general terms
(18.11 "Bots", with 12.1.4 carving API bets out of "suspicious betting"), publishes a
machine-readable interface definition, and has a maintained Python client
(`betfairlightweight` 2.23.2, last released 2026-03-16).

**Alternatives considered**:

- **Smarkets**: rejected. The order body is `market_id, contract_id, price, quantity, side, label,
  minimum_accepted_quantity, type`. `label` is a strategy tag with no deduplication and no
  server-side filter. No caller reference exists. Its Python SDK was archived in 2019.
- **Matchbook**: rejected. Offer fields are exactly `runner-id, side, odds, stake, keep-in-play`.
  No caller reference field at all. The community Python client died in 2018 and the official
  SDK is Java.
- **Betdaq**: rejected. `PunterReferenceNumber` looks like a caller reference and is not one.
  It is an `xs:long`, so it cannot hold a UUID, the spec states it "does not need to be unique",
  and no endpoint filters on it. `ListOrdersChangedSince` and `GetOrderDetails` take only
  `sequenceNumber` and `handle`. It is a correlation ID.

Designing the venue contract against any of the three would have baked in a promise the venue
cannot honour, and the bug would surface as a double stake rather than as a type error.

## D2: There is no placement sandbox, at any of the four

**Decision**: The sanctioned path is proved against fakes, recorded responses and contract
tests. No test touches a venue, because no venue offers a facility where that would be safe.

**Rationale**: This falsified the spec's assumption that the reference exchange "offers a sandbox
or equivalent test facility". None of the four does. The specific trap worth recording: Betfair's
delayed application key is widely described as a sandbox and is not one. Betfair's own docs state
it "operates on the live (production) Betfair Exchange and not a testbed/sandbox environment",
and its capability matrix lists Bet Placement (Live Exchange) as available. A contributor who
believes the folklore and points the suite at a delayed key spends real money. FR-026 already
forbade touching a real venue, so the requirement does not change. What changes is that this is
now the only available strategy rather than the conservative choice, and the reason is written
down so nobody relaxes it later.

**Alternatives considered**: Betdaq's `RC598 MarketIsForPlayMoney` return code implies play-money
markets exist somewhere in its model, but nothing documents how an external API customer reaches
one, and Betdaq is rejected on D1 grounds anyway.

## D3: Idempotency needs two references, not one

**Decision**: The venue contract carries two distinct references. `dedupe_ref` maps to Betfair's
`customerRef`. `order_ref` maps to `customerOrderRef`. Both are sent on every placement.

**Rationale**: The obvious interface is a single `client_ref`, and it is wrong. Betfair splits the
job across two fields with disjoint guarantees:

| | `customerRef` | `customerOrderRef` |
| --- | --- | --- |
| Scope | per request | per instruction |
| Limit | 32 chars, charset `A-Za-z0-9 : - . _ + * ; ~` | 32 chars |
| Deduplicates | yes, within a 60 second window | no, "no validation will be done on uniqueness" |
| Readable back | no | yes, via the stream and `listCurrentOrders` |
| Filterable | no | yes, `listCurrentOrders(customerOrderRefs=[...])` |

Neither field alone satisfies FR-014. `customerRef` enforces once-only but evaporates after 60
seconds and never appears in the response, so it cannot answer "did this already happen".
`customerOrderRef` answers that question durably but enforces nothing, so emitting a duplicate is
the caller's bug rather than a caught error. A single-reference abstraction would have collapsed
these into one field and left a 60 second hole in the guarantee, discoverable only in production
under a retry. Betfair is the only venue that reveals the distinction, which is a further reason
to let it shape the contract.

The retry contract is therefore explicit and pessimistic: within 60 seconds, `customerRef` covers
resubmission. Past it, recovery is a `listCurrentOrders` lookup filtered on `order_ref`, which is
a genuine server-side query only at Betfair. Generating a colliding `order_ref` is our defect to
prevent, not the venue's to catch.

**Unresolved and treated pessimistically**: Betfair's `SportsAPING.xml` defines
`PlaceExecutionReport.customerRef` as "Echo of the customerRef if passed", while the current
documentation says the field does not persist into the response. The two cannot be reconciled
from the outside, so the adapter treats `customerRef` as not readable back and depends on
`order_ref` for every reconciliation. Also undocumented: the maximum number of `customerOrderRefs`
accepted per `listCurrentOrders` call. The stated cap of 250 applies to `betIds` and `marketIds`.
Batch reconciliation must not assume it generalises.

## D4: The browser layer is Playwright, used directly

**Decision**: Playwright drives the site-driven path. crawl4ai is rejected.

**Rationale**: Three reasons, in descending order of weight.

**It refuses to click things that cannot be clicked.** crawl4ai's only interaction primitive is
`js_code`, arbitrary JavaScript injected and evaluated. There are no structured click, type or
fill methods. Tested against a mock bet slip with a disabled confirm button, the two models
diverge exactly where it matters:

```text
Playwright REFUSED disabled submit: Locator.click: Timeout 1200ms exceeded
raw JS route:  JS clicked a DISABLED confirm button silently
```

Playwright's actionability checks refuse hidden and disabled elements. The raw-JS route reported
success on a Place-bet button the site had deliberately disabled. On a control that moves real
money, a false success is the worst possible failure, and it is FR-016's receipt lying.

**Its stealth stack is a hard dependency, not an extra.** crawl4ai installs `patchright`,
`playwright-stealth` and `fake-useragent` unconditionally. Patchright self-describes as "a patched
and undetected version of the Playwright Testing and Automation Framework"; defeating detection is
its entire purpose. Every crawl4ai evasion switch is default-off today (`enable_stealth=False`,
`magic=False`, `simulate_user=False`, `override_navigator=False`, all verified in
`async_configs.py`), so it is technically usable within FR-023. Three things cut against relying
on that: the undetected-browser fork ships to every user regardless of the flags; `magic=True` is
a mislabelled evasion bundle whose docstring advertises overlay handling while the source
generates random user agents and injects a navigator overrider, so a contributor could enable it
having read only the docstring; and crawl4ai's own documentation states it "may enable stealth
mode and undetected browser by default" in future. A dependency that plans to turn evasion on is a
standing liability for a spec that forbids it.

**Weight and supply chain.** Three packages against 55. crawl4ai unconditionally pulls `openai`,
`tiktoken`, `tokenizers`, `huggingface-hub`, `scipy`, `networkx`, `shapely`, `rtree` and
`trimesh`, a 3D mesh library. It also pins `unclecode-litellm`, a personal fork whose PyPI summary
reads "Pre-compromise fork of litellm". For a module that handles bookmaker credentials, that is a
supply-chain surface to decline. Playwright is Apache-2.0, supports 3.11 through 3.14, and its
headless shell download measured 93.5 MiB.

Playwright's project stance matches the spec. Maintainer `pavelfeldman`, closing
playwright-python#527: "Playwright is a testing library and while you can use it for scraping, we
don't give advice on how hide the automation bit from the servers."

**Alternatives considered**:

- **Patchright**: disqualified outright. Evasion is the entire product; there is nothing to
  disable.
- **browser-use**: cleaner than its marketing implies, since stealth and proxy rotation are cloud
  upsell text with no evasion engine in the package. Rejected anyway on two counts: its `Agent`
  always instantiates an LLM, falling back to `ChatBrowserUse()` when passed none, which is the
  in-package agent loop FR-022 forbids; and it hard-pins 35 dependencies with `==`, including
  `anthropic`, `openai`, `google-genai`, `groq` and `ollama`, which will not resolve against this
  project.
- **Steel**: rejected as unverified and disproportionate. Its SDK flags (`solve_captcha`,
  `stealth_config`, `use_proxy`) are documented default-false, but the self-hosted server lists
  "Anti-Detection: stealth plugins and fingerprint management" as an unconditional core feature,
  and whether the Docker image applies stealth by default when `stealth_config` is omitted could
  not be established without reading launch internals. It also requires running a Docker server.

## D5: The page is read as an AI-mode ARIA snapshot, not as markdown

**Decision**: The site-driven read primitive returns `locator.aria_snapshot(mode='ai')`.

**Rationale**: FR-007 needs the page "in a form the agent can reason about", and the agent must
then act on what it read. Markdown satisfies the first half and defeats the second: it discards
element identity, so it can be read but not clicked. AI-mode ARIA snapshots carry stable refs:

```yaml
- form "Bet slip" [ref=e1]:
  - textbox "Stake" [ref=e2]: "10.00"
  - checkbox "Accept odds changes" [checked] [active] [ref=e4]
  - button "Place bet" [ref=e5]
```

That is compact YAML an LLM reads cheaply, refs to act against, and post-action state reflected
back. It closes the navigate, read and act loop in one artifact, and it inverts crawl4ai's single
advantage. `get_by_role` and `get_by_label` pair directly with the roles and names it emits.

**Two corrections to widely-cited material**: `page.accessibility.snapshot()` was removed in
Playwright 1.57, so code using it is broken. Plain `aria_snapshot()` without `mode='ai'` omits the
refs and is for test assertions.

**Alternatives considered**: markdown alongside snapshots, via `markdownify` (MIT) or `trafilatura`
(Apache-2.0), remains open if reading odds ever wants a cheaper representation. `html2text` is
excluded permanently: it is GPL-3.0-or-later and would contaminate this project's MIT licence.

## D6: One browser context, held by the process

**Decision**: The MCP server holds one live `BrowserContext` for its lifetime, created with
`launch_persistent_context(user_data_dir=...)`. Tool calls act on that context.

**Rationale**: An agent calls navigate, read and act as discrete tools, so the authenticated
bookmaker session has to outlive any single call. The live context is what spans them;
`user_data_dir` adds durability across restarts so a login survives. Two constraints shape the
implementation: Playwright allows one browser instance per `user_data_dir`, and "Playwright's API
is not thread-safe", so the documented `with sync_playwright()` idiom cannot be used as shown and
`start()`/`stop()` are managed explicitly.

This also makes D7 structural. One context per process means concurrent watchers are not
reachable, rather than merely discouraged.

## D7: Placement is sequential

**Decision**: A batch places one bet at a time, checking running exposure against the limit before
each stake. A configurable minimum interval separates actions.

**Rationale**: This is a correctness requirement before it is anything else. FR-011 enforces a
maximum total exposure, which needs a running total that concurrent in-flight stakes make
unreadable without a lock. FR-014 allows one stake per identity, and concurrent retries are the
canonical way to produce two. Sequential placement makes both tractable. A rate limit between
actions is ordinary good behaviour toward a service, expressed in seconds.

**Explicitly not adopted**: behavioural mimicry whose unit is human-likeness rather than time,
meaning randomised delays fitted to human timing distributions, mouse movement curves, and typing
cadence emulation. FR-023 covers this. One session acting once at a time at a sane pace is what a
person at a keyboard produces anyway, so the honest implementation and the realistic one coincide.

## D8: Geography decides which adapter a user can reach

**Decision**: Both adapter families ship. The Betfair adapter serves users within its footprint
and fixes the shape of the venue contract. The site-driven path serves users the exchanges cannot
reach, including the maintainer.

**Rationale**: All four surveyed exchanges are closed to Greece, verified from the maintainer's
connection rather than inferred. Smarkets, Matchbook and Betdaq resolve to `62.74.31.13`, the
Hellenic Gaming Commission's block server, and all three appear on the official EEEP blacklist
(52nd edition, 12.06.2026). Betfair is the subtle case and the reason the naive check misleads:
`betfair.com` resolves normally and is genuinely absent from the blacklist, yet Betfair blocks
Greece itself at the application layer, redirecting to `/gr` and returning a Betfair-branded page
naming `Region: GR`. It is unblacklisted precisely because it withdrew from Greece in 2012 over
licensing rather than operate unlicensed. Greece is not in the named Prohibited Territory list of
T&C 4.1.6, which closes with "and any other country with a comparable legal situation".

This inverts the spec's priority ordering for the maintainer specifically, without changing the
spec: User Story 1 is the testable, correct contract that other users can transact against, and
User Story 3 is the path the maintainer can use. Both are in scope per the maintainer's decision.

A VPN is not treated as a workaround. It breaches T&C 4.1.6 directly, the stated penalty is
confiscation of the balance, and geographic controls are automation controls under FR-023.

**Unverified and recorded as such**: Betfair's block page also mentions "traffic from your network
was detected as being unusual", so a single request from one Greek IP does not formally exclude
bot detection as the cause. The branded page naming the region, the `/gr` redirect, the documented
2012 exit and the absent Greek licence all point one way, but no test was run from a non-Greek IP.

## D9: The no-evasion boundary, and what it costs

**Decision**: FR-023 stands as written. No stealth browsing, fingerprint spoofing, captcha
solving, proxy rotation, or geographic circumvention. When a venue blocks automation, the system
reports it and stops.

**Rationale**: The boundary separates automating an account the user holds, which is a matter
between the user and the venue, from defeating a control the venue built to deny that access,
which is not. It also survives on engineering grounds alone: evasion is untestable under FR-026,
degrades whenever the venue ships a countermeasure, and converts a closed account into a forfeited
balance with a record of deliberate circumvention.

Proxy rotation deserves its own note, because it is the one that looks harmless. It accomplishes
nothing here: the session is authenticated, so the venue knows the account from the cookie before
the IP is relevant. What rotation adds is an account that appears to be accessed from many places
at once, which is the signature venues watch for as sharing, resale or compromise. It works
against the user's own goal.

**Practical consequence, stated plainly**: bet365 is likely to block an honest automated browser.
Novibet and Stoiximan are the plausible site-driven targets. Per FR-007 the system claims no site
works, and per FR-023 a blocked venue is reported rather than circumvented.
