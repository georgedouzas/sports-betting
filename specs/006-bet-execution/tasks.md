---

description: "Task list for bet execution"
---

# Tasks: Bet execution

**Input**: Design documents from `specs/006-bet-execution/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md),
[data-model.md](./data-model.md), [contracts/](./contracts/)

**Tests**: Requested and mandatory. This feature spends real money, so every refusal path carries
its own test. FR-026 makes an untested refusal a defect.

**Organization**: Grouped by user story. Each story carries its own surface tasks, because FR-020
and SC-006 mean a story without its CLI and MCP is not delivered.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: US1, US2, US3
- Exact file paths in every description

---

## Non-negotiables

These bind every task below. A task that cannot be done within them is a task that stops and
reports rather than one that bends them.

- **THE CORE MUST NOT CHANGE.** `git diff --stat -- src/sportsbet/dataloaders src/sportsbet/evaluation`
  MUST be empty (FR-025). Execution imports the public API and has no reason to reach inside. If
  the core must change, STOP AND REPORT why.
- **NO TEST MAY CONTACT A REAL VENUE OR PLACE A REAL BET, EVER** (FR-026). There is no sandbox
  anywhere. Betfair's delayed application key is widely believed to be one and is not: it places
  REAL bets on the live exchange (research D2). Fakes, recorded payloads and a loopback page only.
  A test that could stake money is a bug.
- **No new REQUIRED runtime dependency.** `execution` is an optional extra holding `playwright`
  alone. The Betfair adapter uses the `aiohttp` already present. The extra MUST be locked with
  `pdm lock -G execution` or CI fails exactly as the `mcp` group did.
- **NO EVASION** (FR-023). No stealth, fingerprint spoofing, captcha solving, proxy rotation,
  geographic circumvention, or human-timing mimicry. Pacing is a knob in SECONDS. A blocked venue
  returns `BLOCKED` and stops.
- **CREDENTIALS** are named variables, never a function, CLI or tool argument, never logged, never
  pickled (FR-017, FR-018, FR-019).
- **Style**: full annotations; mypy, ruff, bandit, interrogate clean; line length 120;
  skip-string-normalization; short Google-style docstrings; no explanatory inline comments; flat
  test files; never import a private name from a private module in a test; no executable doctests
  on network-touching classes.
- **Every phase ends with the full gate green**: `pdm run formatting`, `pdm run checks`,
  `pdm run tests`.

**Bandit watch item**: the `B404`, `B603` and `B607` skips were removed when the GUI went, and the
gate is stricter now. Playwright launches its driver internally rather than through our
`subprocess`, so nothing should re-trip them. If something does, REPORT it rather than re-adding
the skip.

---

## Phase 1: Setup

**Purpose**: The extra, the dependency, the skeleton. Mechanical.

- [X] T001 Add the `execution` optional-dependency group holding `playwright` alone to `pyproject.toml`, and lock it with `pdm lock -G execution`. The Betfair adapter uses the existing `aiohttp`, so the group stays at one package. Verify `pdm install -dG maintenance -dG tests -G mcp -G execution` resolves, since an unlocked group is the exact failure the `mcp` group hit in CI.
- [X] T002 Create the `src/sportsbet/execution/` package skeleton (`__init__.py`, `_base.py`, `_betfair.py`, `_browser.py`, `_credentials.py`) as a peer of `cli/` and `mcp/`, importing only the public API. Add `sportsbet.execution` to `docs/generate_api.py`.
- [X] T003 [P] Add the test doubles to `tests/conftest.py`: a `FakeVenue` implementing `BaseVenue` in memory with injectable faults (timeout after accept, crash before receipt, partial match, suspended market, insufficient funds), and a loopback page server fixture. The socket guard already permits loopback, so no guard change is needed. State in the fixture docstring that these exist because no venue sandbox does.

**Checkpoint**: The extra installs, the package imports, and nothing can reach a venue.

---

## Phase 2: Foundational (Blocking Prerequisites)

**⚠️ CRITICAL**: No user story work can begin until this phase is complete. This is where the
money-safety primitives live.

- [X] T004 Implement `BetIdentity` in `src/sportsbet/execution/_base.py` with `venue`, `match`, `market`, `selection`, and a derived `ref` property computing `blake2s(f'{venue}|{match}|{market}|{selection}'.encode(), digest_size=16).hexdigest()`. Exactly 32 hex characters, which is Betfair's field limit, and hex is inside its charset `A-Za-z0-9 : - . _ + * ; ~`. Identity derives from those four and NOTHING else: not the run, the model, the timestamp, or the batch (FR-014).
- [X] T005 [P] Test `BetIdentity` in `tests/test_execution.py`: the ref is exactly 32 chars and hex; the same four fields give the same ref across processes; any field changing changes the ref; two intents from different runs sharing the four ARE the same identity. This determinism is what lets FR-015 keep no state, so it is a contract test rather than a unit test.
- [X] T006 [P] Implement `PlacementIntent` (identity, stake, min_price, value_bet) in `src/sportsbet/execution/_base.py`. `min_price` defaults to the price the value bet was computed at, since below it the bet stops being a value bet (FR-013). `stake` is an input, never a model output.
- [X] T007 [P] Implement `PlacementStatus` in `src/sportsbet/execution/_base.py` with `DRY_RUN`, `ACCEPTED`, `MATCHED_FULL`, `MATCHED_PARTIAL`, `ALREADY_PLACED`, `REFUSED_LIMIT`, `REFUSED_PRICE`, `REFUSED_UNCONFIRMED`, `REFUSED_KILLED`, `BLOCKED`, `REJECTED`. `BLOCKED` is a first-class outcome rather than an exception, because FR-023's contract is that a blocked venue is reported honestly.
- [X] T008 Implement `PlacementQuote` (intents, total_stake, total_exposure, quoted_at) in `src/sportsbet/execution/_base.py`. A quote is a promise, not a reservation.
- [X] T009 Implement `PlacementReceipt` and its `pandera` schema in `src/sportsbet/execution/_base.py`. It crosses a public boundary as a DataFrame, so Constitution Principle II requires the schema. `detail` names the limit, the price or the reason and MUST NOT contain a credential (FR-018).
- [X] T010 [P] Implement `ExposureLimits` (max_stake_per_bet, max_total_exposure, killed) in `src/sportsbet/execution/_base.py`.
- [X] T011 Implement `CredentialRef` and resolution in `src/sportsbet/execution/_credentials.py`. It carries the NAME of the variable and reads the secret at use. It never returns the secret to a caller, never logs it, never pickles it, and never puts it in `detail`. A missing variable names what was expected and stops (FR-019). Mirror the existing `odds_key_env` pattern.
- [X] T012 [P] Test credential handling in `tests/test_execution.py`: a missing variable names the variable it wanted and stops; the resolved secret appears in no return value, exception, or `repr`. Uses a sentinel value.
- [X] T013 Implement the `BaseVenue` ABC in `src/sportsbet/execution/_base.py` per [contracts/venue.md](./contracts/venue.md): `authenticate`, `list_markets`, `read_balance`, `place`, `read_status`, `cancel`, plus `key` and `can_cancel`. Docstring `place` as idempotent on `identity.ref`. A venue that cannot make that guarantee does not implement this contract.

**Checkpoint**: The identity derives, the entities validate, credentials stay named. Stories can start.

---

## Phase 3: User Story 1 - Place the value bets at a venue with an official betting API (Priority: P1) 🎯 MVP

**Goal**: Value bets go from a fitted bettor to a venue, with refusal as the default and exactly
one stake per intended bet.

**Independent Test**: Against `FakeVenue`, run in default mode and assert zero stakes; run with the
exact quote echoed back and assert each intended bet placed once; inject faults and assert the
count never exceeds one.

**This phase carries the risk.** Every refusal below is a separate task and a separate test,
because a refusal that is only described is a refusal that does not exist.

### Tests for User Story 1

> Write these FIRST and ensure they FAIL before implementing.

- [X] T014 [P] [US1] Test the default refuses in `tests/test_execution.py`: `place` with no confirmation returns `DRY_RUN` on every receipt, stakes zero, and the fake venue records zero calls. This is SC-002, and it must hold for 100% of runs that do not opt in.
- [X] T015 [P] [US1] Test a mismatched confirmation in `tests/test_execution.py`: `confirm_stake` or `confirm_exposure` differing from the quote returns `REFUSED_UNCONFIRMED`, stakes zero, and the message states the REAL figures. Cover each of the two figures wrong alone, and both wrong.
- [X] T016 [P] [US1] Test the ceilings in `tests/test_execution.py`: a stake above `max_stake_per_bet` and a batch above `max_total_exposure` each return `REFUSED_LIMIT` with `detail` NAMING which ceiling was hit (SC-009). Include the exposure ceiling being reached partway through a batch, so the placed prefix stands and the remainder refuses.
- [X] T017 [P] [US1] Test the kill switch in `tests/test_execution.py`: engaged before a batch, every receipt is `REFUSED_KILLED`; engaged mid-batch, the already-placed prefix stands and the remainder refuses (FR-012).
- [X] T018 [P] [US1] Test the price floor in `tests/test_execution.py`: a venue price below `min_price` returns `REFUSED_PRICE` and stakes zero. Include the default, where `min_price` is the price the value bet was computed at (FR-013).
- [X] T019 [P] [US1] Test once-only under faults in `tests/test_execution.py`. The load-bearing test (SC-003). Inject a timeout after the fake venue accepted but before the response landed, then retry. Inject a crash between accept and receipt, then re-run FROM SCRATCH with no state carried over. Both end with exactly ONE stake per identity and the second attempt reporting `ALREADY_PLACED` (FR-014a). Assert that no file, cache or directory holds placement state, since FR-015 forbids a local record that could drift.
- [X] T020 [P] [US1] Test the Betfair two-reference round trip in `tests/test_betfair.py` against recorded payloads: `placeOrders` carries `identity.ref` as BOTH `customerRef` and `customerOrderRef`, with exactly ONE instruction per request. Recovery calls `listCurrentOrders(customerOrderRefs=[ref])`. Assert nothing reads `customerRef` back, since Betfair's docs and its interface definition contradict each other on whether it persists and the adapter treats it as unreadable (research D3).
- [X] T021 [P] [US1] Test sequential placement in `tests/test_execution.py`: the fake venue records calls strictly in order, one at a time, with the running exposure checked before each. Assert the configured minimum interval is honoured. Pacing is a knob in seconds and carries no timing mimicry (FR-023).

### Implementation for User Story 1

- [X] T022 [US1] Implement `quote(venue, intents, limits)` in `src/sportsbet/execution/__init__.py`: itemise every bet, its stake and its price, and total the batch stake and the exposure including what is already open at the venue (FR-010, SC-008).
- [X] T023 [US1] Implement `place(venue, quote, limits, confirm_stake=None, confirm_exposure=None)` in `src/sportsbet/execution/__init__.py` with the refusal order from [contracts/venue.md](./contracts/venue.md): kill switch, then confirmation, then per intent sequentially kill switch, ceilings, `min_price`, place. Dry run is the ABSENCE of confirmation, not a flag, so no default can be misconfigured into spending. Enforced IN CODE, never in a docstring.
- [X] T024 [US1] Implement the sequential loop with the running exposure total in `src/sportsbet/execution/__init__.py`. One bet at a time is what makes FR-011's ceiling readable and FR-014's once-only tractable, since concurrent in-flight stakes make both unenforceable without a lock (research D7).
- [X] T025 [US1] Implement `BetfairVenue.authenticate` in `src/sportsbet/execution/_betfair.py`: certificate login at `identitysso-cert.betfair.com/api/certlogin` over `aiohttp` with `ssl.SSLContext.load_cert_chain`, then `X-Application` and `X-Authentication` on every call. Constructor takes `*_env` NAMES with name defaults, so nothing sensitive has a default.
- [X] T026 [US1] Implement `BetfairVenue.place` in `src/sportsbet/execution/_betfair.py`: `placeOrders` with `customerRef` and `customerOrderRef` both set to `identity.ref`, ONE instruction per request. On a duplicate inside 60 seconds Betfair deduplicates on `customerRef`; past it, recover with `listCurrentOrders(customerOrderRefs=[ref])` and return `ALREADY_PLACED` (FR-014a). Emitting a colliding ref is our defect to prevent, since Betfair validates no uniqueness on it.
- [X] T027 [P] [US1] Implement `BetfairVenue.list_markets` in `src/sportsbet/execution/_betfair.py` via `listMarketCatalogue` and `listMarketBook`. Respect 5 requests/sec per market ID.
- [X] T027a [US1] Add `build_venue(venue)` beside `build_bettor` in `src/sportsbet/_selection.py`, resolving `betfair` to `BetfairVenue()` and `venue.py:VENUE` through the existing `_load_object`, exactly as `--model` resolves `models.py:BETTOR`. A venue is the model case rather than the dataloader case: its URLs, notes and certificate paths are not a closed list of names, so no argument list describes one. Import `sportsbet.execution` LAZILY inside the function, since the extra is optional and `_selection.py` is imported on every run, and name the extra to install when it is missing (SC-010).
- [X] T028 [US1] Add the `execution quote` and `execution place` commands to a new `src/sportsbet/cli/_execution.py`, taking `--venue` and resolving it with `build_venue`. Every parameter passed with `--`, no positional arguments. No `--dry-run` flag: dry run is the absence of `--confirm-stake` and `--confirm-exposure`. `place` without them prints the full quote, stakes nothing, and exits non-zero.
- [X] T029 [US1] Add `execution_venue_info`, `execution_authenticate`, `execution_quote` and `execution_place` tools to `src/sportsbet/mcp/_server.py`, each taking a `venue` reference resolved by `build_venue`, exactly as the existing tools take a `model`. `execution_place` enforces the confirmation IN CODE and states the real figures when it refuses, following the `prepare`/`confirm_cost` pattern from feature 005. DataFrames cross as records.
- [X] T030 [P] [US1] Test the MCP refusal in `tests/test_mcp.py`: `execution_place` without confirmations refuses; with wrong figures refuses and names the real ones; with exact figures places against the fake. An agent is the caller most likely to skim, which is why FR-009 binds every surface.
- [X] T031 [US1] Extend `tests/cli/test_parity.py` to cover the execution group, asserting every US1 capability is reachable from the Python API, the CLI and the MCP server (SC-006).

**Checkpoint**: DONE. The loop from model to money closes, and refuses by default. MVP delivered, gate green
on 3.11, 3.12 and 3.13, core diff empty.

---

## Phase 4: User Story 2 - Follow the money after placement (Priority: P2)

**Goal**: What became of the bets, traced back to the value bet that caused each.

**Independent Test**: Against `FakeVenue` holding known bets, read balance, open bets and statuses,
and assert each placement traces back to the match, the market and the selection from the venue
alone.

### Tests for User Story 2

- [X] T032 [P] [US2] Test status reporting in `tests/test_execution.py`: bets are reported open, matched, rejected or settled, including a partial match.
- [X] T033 [P] [US2] Test traceability in `tests/test_execution.py`: every receipt identifies the venue, the market, the selection, the stake, the price obtained and the value bet (FR-016). Assert the trace works FROM THE VENUE ALONE, with no local state, which is SC-005 and the reason identity excludes the run.
- [X] T034 [P] [US2] Test cancellation honesty in `tests/test_execution.py`: a venue with `can_cancel=False` raises `CancellationUnsupported` rather than returning a receipt implying it cancelled (FR-008).

### Implementation for User Story 2

- [X] T035 [P] [US2] Implement `BetfairVenue.read_status` in `src/sportsbet/execution/_betfair.py` via `listCurrentOrders(customerOrderRefs=[...])`. Chunk conservatively: the documented 250 cap applies to `betIds` and `marketIds`, and the limit for `customerOrderRefs` is undocumented, so do not assume it generalises (research D3).
- [X] T036 [P] [US2] Implement `BetfairVenue.read_balance` in `src/sportsbet/execution/_betfair.py` via `getAccountFunds`, returning balance and open exposure.
- [X] T037 [P] [US2] Implement `BetfairVenue.cancel` in `src/sportsbet/execution/_betfair.py` via `cancelOrders`, with `can_cancel = True`.
- [X] T038 [US2] Add `execution status`, `execution balance` and `execution cancel` to `src/sportsbet/cli/_execution.py`.
- [X] T039 [US2] Add `execution_read_status`, `execution_read_balance` and `execution_cancel` tools to `src/sportsbet/mcp/_server.py`.
- [X] T040 [US2] Extend `tests/cli/test_parity.py` for the US2 capabilities.

**Checkpoint**: DONE. Placement stops being write-only.

---

## Phase 5: User Story 3 - Place at a bookmaker that publishes no API, driven by an agent (Priority: P3)

**Goal**: Generic browser primitives an outside agent drives, with the agent supplying the site
knowledge.

**Independent Test**: Against a mock bookmaker page over loopback, an agent navigates, reads
markets, pins a session, fills a slip and confirms.

**Read this before starting.** `BrowserSession` is NOT a `BaseVenue` and has NO `place`,
`read_status` or `cancel`. Planning found FR-005 and FR-007 in contradiction: a `place` on a
website has to find the market, click the price, fill the stake and work the confirm flow, all of
which is site knowledge, making it either a per-site adapter (FR-007 forbids) or a model call in
the package (FR-022 forbids). The agent places. Once-only and the ceilings are consequently the
agent's on this path, and the docs say so.

### Tests for User Story 3

- [X] T041 [P] [US3] Test that the guarantee is unreachable in `tests/test_browser.py`: `BrowserSession` is not a `BaseVenue` and exposes no `place`, `read_status` or `cancel`. A caller must not be able to reach a guarantee that does not exist. This test IS the FR-005/FR-007 resolution, so it stays permanently.
- [X] T042 [P] [US3] Test actionability in `tests/test_browser.py` against a mock bet slip served over loopback: `click` on a DISABLED confirm control fails rather than reporting success, and the same for a hidden one. This is a permanent regression test and the property that decided Playwright over injected JavaScript, which silently "clicked" a disabled Place-bet button under test (research D4). A false success on a confirm button is a receipt that lies about money.
- [X] T043 [P] [US3] Test the pinned session in `tests/test_browser.py`: serve a page whose odds widget re-renders on a timer, changing every snapshot ref. Explore, `fix` the stake and confirm locators, act after a re-render, and assert the pinned locators still resolve. Assert `fix` rejects a raw ref and stores no price.
- [X] T044 [P] [US3] Test the block path in `tests/test_browser.py`: a page returning a block response yields `BLOCKED` and stops. Assert no retry with different timing, headers, or presentation (FR-023).
- [X] T045 [P] [US3] Test `notes` in `tests/test_browser.py`: stored and returned verbatim through `execution_venue_info`, never parsed. Assert the package holds no site table, no selector pack, and no default naming a bookmaker.

### Implementation for User Story 3

- [X] T046 [US3] Implement `BrowserSession.__init__` in `src/sportsbet/execution/_browser.py` taking `key`, `url`, `notes`, `credential_env`, `user_data_dir` and `min_interval`. Parameters stored unmodified and unvalidated per Constitution Principle I. `notes` is ONE free-text blob rather than structured URL fields, because nothing in the library parses it and the only reader is the agent.
- [X] T047 [US3] Implement session lifecycle in `src/sportsbet/execution/_browser.py`: one `BrowserContext` per process via `launch_persistent_context(user_data_dir=...)`, held across tool calls so a login survives them. One instance per `user_data_dir`. Playwright's API is not thread-safe, so manage `start()`/`stop()` explicitly rather than using the documented `with sync_playwright()` idiom (research D6).
- [X] T048 [US3] Implement `navigate` and `snapshot` in `src/sportsbet/execution/_browser.py` returning `PageSnapshot` from `locator.aria_snapshot(mode='ai')`. Scope by locator or depth rather than snapshotting `body`, since it is the token cost of every agent turn. Note `page.accessibility.snapshot()` was removed in Playwright 1.57 and plain `aria_snapshot()` without `mode='ai'` omits the refs.
- [X] T049 [US3] Implement `click`, `type` and `select` in `src/sportsbet/execution/_browser.py`, each taking a ref and returning the resulting snapshot so the agent sees what its action did without a second call. Pace with `min_interval`.
- [X] T050 [US3] Implement `fix(match, locators)` and `FixedSession` in `src/sportsbet/execution/_browser.py`. Store roles and accessible names, NEVER refs, which go stale on the re-render an odds widget does constantly. Store NO price: the price is read at placement and checked against `min_price`, and pinning it would defeat the only protection between quoting and landing. A `FixedSession` is navigation state, not placement state, so it does not conflict with FR-015.
- [X] T051 [US3] Add `execution page read`, `execution page act` and `execution page fix` to `src/sportsbet/cli/_execution.py`.
- [X] T052 [US3] Add `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_select` and `browser_fix` tools to `src/sportsbet/mcp/_server.py`. They carry no site knowledge: the agent supplies which site, which control, which market.
- [X] T053 [US3] Extend `tests/cli/test_parity.py` for the US3 capabilities.

**Checkpoint**: DONE. An agent can drive a site, and the library promises only what it can keep.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T054 Add `docs/overview/user_guide/execution.md` leading with the money and terms-of-service risks, ahead of any instruction to use it (FR-024). State plainly that driving a bookmaker's website breaches essentially every bookmaker's terms and risks account closure and loss of the balance. State that once-only and the ceilings hold on the API path and are the agent's on the site path. No bold, no dashes, no semicolons; say what things do rather than what they do not.
- [ ] T055 [P] Document in `docs/overview/user_guide/execution.md` that no exchange offers a placement sandbox, and that Betfair's delayed application key places REAL bets on the live exchange. Users will otherwise believe the folklore and test with real money.
- [ ] T056 [P] Register the execution page in the `properdocs.yml` nav.
- [ ] T057 [P] Add the CHANGELOG entry for `sportsbet.execution` and the `execution` extra, with a runnable snippet.
- [ ] T058 [P] Re-run the public-API docs audit: every public name has a runnable example. Network-touching classes keep non-executable examples.
- [ ] T059 Run every scenario in [quickstart.md](./quickstart.md) and confirm each success criterion.
- [ ] T060 Verify `git diff --stat -- src/sportsbet/dataloaders src/sportsbet/evaluation` is EMPTY (FR-025, SC-007). This is an additive feature.
- [ ] T061 Verify SC-010 in a clean environment: `pip install sports-betting` pulls no `playwright`, and `sportsbet execution` reports the extra to install rather than raising `ImportError`. Install `[mcp]` alone and confirm the execution tools name `[execution]`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: depends on Setup. BLOCKS every story. The identity derivation (T004)
  blocks essentially everything, since once-only rests on it.
- **US1 (Phase 3)**: depends on Foundational. The MVP.
- **US2 (Phase 4)**: depends on Foundational. Reads what US1 placed, so it is meaningful after US1
  though independently testable against `FakeVenue`.
- **US3 (Phase 5)**: depends on Foundational only. Genuinely independent of US1 and US2, since it
  shares no venue code with them.
- **Polish (Phase 6)**: depends on the stories being delivered.

### Within US1

T014 to T021 (tests) fail first. Then T022 to T024 (the entry point), then T025 to T027 (the
adapter), then T028 to T031 (surfaces).

### Parallel Opportunities

- T005, T006, T007, T010, T012 in Phase 2 touch different concerns and can run together.
- All of T014 to T021 are independent tests and can be written in parallel.
- T035, T036 and T037 are separate Betfair endpoints.
- All of T041 to T045 are independent.
- Most of Phase 6 is marked [P].
- With more than one person, US3 can run alongside US1 from the Foundational checkpoint, since
  `_browser.py` and `_betfair.py` share nothing but `_base.py`.

---

## Implementation Strategy

### MVP: Phase 1, 2 and 3

Setup, Foundational, then US1. Stop and validate: the default refuses, the ceilings bind, and
faults never produce a second stake. That is the loop from model to money, closed and safe.

### Then

US2 makes placement readable rather than write-only. US3 adds the path the maintainer can actually
use, given that all four surveyed exchanges are closed to Greece (research D8). US3 is last by
spec priority but is the only path usable from the maintainer's own jurisdiction, so it is worth
delivering rather than deferring indefinitely.

### Not in this feature

In-play. The session design serves live markets, but bettors are fitted on pregame odds and cannot
see a score, so an in-play stake would trace back through FR-016 to a value bet that stopped being
true at kickoff. In-play needs an in-play model, which is a modelling feature. Recorded in
[plan.md](./plan.md).
