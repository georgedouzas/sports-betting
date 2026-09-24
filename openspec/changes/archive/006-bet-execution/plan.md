# Implementation Plan: Bet execution

> REVERSED after implementation: the maintainer rejected shipping any bookmaker in the library. `BetfairVenue`
> was removed. A venue with an API is now user-provided (`venue.py:VENUE`), like a model. The Betfair analysis
> below is kept as the historical rationale for the venue contract's shape, not as a shipped adapter.

**Branch**: `006-bet-execution` | **Date**: 2026-07-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/006-bet-execution/spec.md`

## Summary

Close the loop between finding a value bet and placing it. A new `sportsbet.execution` module,
behind an optional extra, consumes the value bets a bettor produces and places them at a venue.
Bettors are untouched and stay pure estimators, so a reloaded pickle still cannot spend money.

Two venue families implement one contract. Betfair is the reference adapter, chosen because it is
the only surveyed exchange that can carry a caller reference and therefore the only one where
FR-015 (the venue is the record) is implementable. Playwright supplies generic navigate, read and
act primitives for venues with no API, with the agent bringing the site knowledge. All of it is
reachable from the Python API, the CLI and the MCP server.

Refusal is the default throughout: nothing stakes money until the caller echoes back the exact
quote, sequentially, under a stake ceiling and an exposure ceiling.

## Technical Context

**Language/Version**: Python `>=3.11, <3.14`, targeting `py311`. Full annotations.

**Primary Dependencies**: No new *required* runtime dependency. `aiohttp`, `pandas`, `pandera` and
`click` are already core. The `execution` extra adds `playwright` alone. The Betfair adapter is
built on the existing `aiohttp` rather than on `betfairlightweight`: the needed surface is five
endpoints (`placeOrders`, `cancelOrders`, `listCurrentOrders`, `listMarketCatalogue`,
`getAccountFunds`) plus a certificate login, which does not justify a client library whose surface
is an order of magnitude larger. Revisit only if the certificate login against
`identitysso-cert.betfair.com` proves awkward under `ssl.SSLContext.load_cert_chain`.

**Storage**: None, deliberately. FR-015 makes the venue the record and forbids a local or hidden
store of placements, since a second copy is a thing that drifts. Browser session state is the one
exception and it is not placement state: a Playwright `user_data_dir` holds the bookmaker login.

**Testing**: `pytest` with branch coverage, `pytest-randomly`, `--doctest-modules`, and the
existing socket guard in `tests/conftest.py`. No venue sandbox exists anywhere (research D2), so
the sanctioned path is proved with fakes, recorded payloads and contract tests, and the
site-driven path against a locally served page over loopback, which the guard already permits.
Network-touching classes carry non-executable examples, following the established pattern.

**Target Platform**: Linux, macOS and Windows, matching the current CI matrix. Playwright's wheel
supports 3.11 through 3.14; the browser binary is a separate explicit `playwright install` step,
so installing the extra does not pull 93.5 MiB.

**Project Type**: Single library with three surfaces.

**Performance Goals**: None that matter. Placement is sequential by design (research D7), so
throughput is not a target and is in fact bounded on purpose. Betfair's limits are far above
anything reachable here: 1000 transactions/sec overall, 5/sec per market.

**Constraints**: Additive only, `dataloaders/**` and `evaluation/**` show a zero diff (FR-025).
Line length 120, `skip-string-normalization`, Google docstrings, no explanatory inline comments,
flat test files, no private imports from private modules in tests.

**Scale/Scope**: One venue contract, one API adapter, one browser layer, three surfaces. Roughly
five new source modules and three new test files.

## Constitution Check

*GATE: checked against constitution v1.1.0 before Phase 0 and re-checked after Phase 1.*

| Principle | Status | Basis |
| --- | --- | --- |
| **I. scikit-learn-Compatible API** | PASS | FR-001 and FR-002 keep placement off the bettor entirely, so estimators keep the contract and a pickle stays inert. FR-020 puts every capability on all three surfaces. FR-022 and research D4 keep any agent loop out: `browser-use` was rejected partly because its `Agent` instantiates an LLM by default, which would have imported the thing the constitution forbids. |
| **II. Type Safety & Schema Validation** | PASS with an obligation | Full annotations, mypy clean. The receipt table crosses a public boundary, so it gets an explicit `pandera` schema like every other public frame. Phase 1 owes that schema. |
| **III. Test Coverage & Doctest Discipline** | PASS | FR-026 forbids touching a venue and research D2 removes the sandbox that might have tempted an exception. Fakes, recorded payloads and a loopback page instead. Examples on network-touching classes stay non-executable, as elsewhere. |
| **IV. Automated Quality Gates** | PASS with a watch item | The bandit skips `B404`/`B603`/`B607` were deliberately removed when the GUI went, and the security gate got stricter as a result. Playwright launches its driver internally rather than through our `subprocess`, so nothing should re-trip them. If bandit flags something once Playwright lands, that gets reported rather than skipped. |
| **V. Documentation as a First-Class Artifact** | PASS with an obligation | The user guide gains an execution page that leads with the money and terms-of-service risk per FR-024, ahead of any instruction to use it. Every public name keeps a runnable example. |

**Tooling rule**: "A capability that needs a credential or performs a real-world side effect MUST
live behind an optional extra." Satisfied by FR-004 and the `execution` extra. Note the
combination: MCP execution tools need both `mcp` and `execution`, and must degrade with a clear
message when only one is installed rather than raising an `ImportError`.

No violations. Complexity Tracking stays empty.

## Project Structure

### Documentation (this feature)

```text
specs/006-bet-execution/
├── plan.md              # This file
├── spec.md              # Corrected against research D2 and D3
├── research.md          # Phase 0 output, nine decisions
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2, not created by /speckit-plan
```

### Source Code (repository root)

```text
src/sportsbet/
├── execution/                # New. Imports only the public API.
│   ├── __init__.py           # Public names
│   ├── _base.py              # Venue contract, PlacementIntent, PlacementReceipt, Exposure
│   ├── _betfair.py           # Reference adapter over aiohttp, implements BaseVenue
│   ├── _browser.py           # BrowserSession: primitives, explore/fix. NOT a BaseVenue.
│   └── _credentials.py       # Reads a named variable, never carries a secret
├── cli/
│   └── _execution.py         # New `sportsbet execution` group, mirroring the Python API
├── mcp/
│   └── _server.py            # Extended with execution tools
├── dataloaders/              # ZERO DIFF (FR-025)
└── evaluation/               # ZERO DIFF (FR-025)

tests/
├── test_execution.py         # Contract, refusal, limits, idempotency under fault injection
├── test_betfair.py           # Recorded payloads, the two-reference round trip
└── test_browser.py           # Loopback page, actionability on a disabled confirm
```

**Structure Decision**: `sportsbet/execution/` is a peer of `cli/` and `mcp/`, importing only the
public API, exactly as the MCP server does. That keeps FR-025's zero diff structural rather than
a matter of discipline: execution consumes what the core already exposes, so it has no reason to
reach inside it. The CLI group is a new file rather than an addition to `_betting.py`, because
`_betting.py` is the evaluation group and execution is a separate group in the Python API.

Test files are flat, matching `tests/test_mcp.py`.

## Notable design consequences carried from research

Three findings shape the interface rather than merely the implementation, so they are recorded
here where tasks will read them:

1. **The venue contract carries two references, not one** (research D3). `dedupe_ref` covers a 60
   second retry window and is unreadable afterwards. `order_ref` persists and is filterable but
   the venue enforces no uniqueness on it. A single `client_ref` would leave FR-014's once-only
   guarantee with a 60 second hole. Both fields are sent on every placement, and recovery past 60
   seconds is a `listCurrentOrders` lookup filtered on `order_ref`.

2. **Placement is sequential and exposure is checked before each stake** (research D7). This is
   what makes FR-011 and FR-014 implementable at all, not a concession to appearing human.

3. **The read primitive returns AI-mode ARIA snapshots, not markdown** (research D5). A bet slip
   must be acted on, and markdown discards element identity. Snapshot refs are what `click` and
   `type` take as targets.

4. **FR-005 and FR-007 contradict each other, and the contract resolves it** (contracts/venue).
   FR-005 requires every venue to implement `place`. FR-007 requires the site-driven path to be
   generic primitives with the agent supplying site knowledge. `BrowserVenue.place(intent)` would
   have to find the market, click the price, locate the stake field and work the confirm flow,
   every step of which is site knowledge, so it is either per-site code (FR-007 forbids) or an LLM
   in the package reading `notes` (FR-022 forbids). `BrowserSession` is therefore not a `BaseVenue`
   and offers no `place`. The agent places.

   Two guarantees consequently hold on the API path and not on the site path, and the docs say so
   rather than letting a user infer parity. Once-only needs the venue's record read back, which
   needs site knowledge. The stake and exposure ceilings bind the caller that places, which on the
   site path is the agent. A generic text match of identity fields against a history snapshot was
   considered and rejected, since a false negative there is the double stake FR-014 exists to
   prevent.

5. **A session is explored, then fixed, then bet against.** Exploration is an LLM over full
   snapshots, slow and token-heavy, and happens once. Placement runs against a moving price and
   cannot re-derive a layout per bet. `fix` pins roles and accessible names, never refs, which go
   stale on the re-render an odds widget does constantly, and never a price, which is read at
   placement and checked against `min_price`.

## Open scope question: in-play

Recorded rather than decided, since it changes the spec.

The session design serves live markets and pregame batches equally. What does not carry over is
the justification for the stake. Bettors are fitted on pregame features and pregame odds, and
`bettor.bet(X_fix, O_fix)` produces value against the pregame price. A model that has not seen the
score cannot know its edge evaporated at kickoff, so an in-play stake wired to a pregame value bet
would trace back, through FR-016's receipt, to a claim that stopped being true when the match
started.

**Recommendation: in-play stays out of scope for 006.** Execution places what the models actually
produce. In-play needs an in-play model, which is a modelling feature rather than an execution
one. The alternative, an agent deciding in-play with nothing behind it, is buildable, but the
documentation would have to say the stake has no backtest justifying it, which is a strange thing
for this library to ship.

## Complexity Tracking

> Fill ONLY if Constitution Check has violations that must be justified.

No violations. Table intentionally empty.
