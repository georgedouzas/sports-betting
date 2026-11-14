# Tasks

## 1. Setup

- [x] 1.1 Add the `execution` optional-dependency group holding `playwright` alone to `pyproject.toml`, and lock it with
  `pdm lock -G execution`. The Betfair adapter uses the existing `aiohttp`, so the group stays at one package. Verify
  `pdm install -dG maintenance -dG tests -G mcp -G execution` resolves, since an unlocked group is the exact failure the
  `mcp` group hit in CI. (T001)
- [x] 1.2 Create the `src/sportsbet/execution/` package skeleton (`__init__.py`, `_base.py`, `_betfair.py`,
  `_browser.py`, `_credentials.py`) as a peer of `cli/` and `mcp/`, importing only the public API. Add
  `sportsbet.execution` to `docs/generate_api.py`. (T002)
- [x] 1.3 Add the test doubles to `tests/conftest.py`: a `FakeVenue` implementing `BaseVenue` in memory with injectable
  faults (timeout after accept, crash before receipt, partial match, suspended market, insufficient funds), and a
  loopback page server fixture. The socket guard already permits loopback, so no guard change is needed. State in the
  fixture docstring that these exist because no venue sandbox does. (T003)

## 2. Foundational money-safety primitives

- [x] 2.1 Implement `BetIdentity` in `src/sportsbet/execution/_base.py` with `venue`, `match`, `market`, `selection`,
  and a derived `ref` property computing `blake2s(f'{venue}|{match}|{market}|{selection}'.encode(),
  digest_size=16).hexdigest()`. Exactly 32 hex characters, which is Betfair's field limit, and hex is inside its charset
  `A-Za-z0-9 : - . _ + * ; ~`. Identity derives from those four and NOTHING else: not the run, the model, the
  timestamp, or the batch (FR-014). (T004)
- [x] 2.2 Test `BetIdentity` in `tests/test_execution.py`: the ref is exactly 32 chars and hex, and the same four fields
  give the same ref across processes, and any field changing changes the ref, and two intents from different runs
  sharing the four ARE the same identity. This determinism is what lets FR-015 keep no state, so it is a contract test
  rather than a unit test. (T005)
- [x] 2.3 Implement `PlacementIntent` (identity, stake, min_price, value_bet) in `src/sportsbet/execution/_base.py`.
  `min_price` defaults to the price the value bet was computed at, since below it the bet stops being a value bet
  (FR-013). `stake` is an input, never a model output. (T006)
- [x] 2.4 Implement `PlacementStatus` in `src/sportsbet/execution/_base.py` with `DRY_RUN`, `ACCEPTED`, `MATCHED_FULL`,
  `MATCHED_PARTIAL`, `ALREADY_PLACED`, `REFUSED_LIMIT`, `REFUSED_PRICE`, `REFUSED_UNCONFIRMED`, `REFUSED_KILLED`,
  `BLOCKED`, `REJECTED`. `BLOCKED` is a first-class outcome rather than an exception, because FR-023's contract is that
  a blocked venue is reported honestly. (T007)
- [x] 2.5 Implement `PlacementQuote` (intents, total_stake, total_exposure, quoted_at) in
  `src/sportsbet/execution/_base.py`. A quote is a promise, not a reservation. (T008)
- [x] 2.6 Implement `PlacementReceipt` and its `pandera` schema in `src/sportsbet/execution/_base.py`. It crosses a
  public boundary as a DataFrame, so Constitution Principle II requires the schema. `detail` names the limit, the price
  or the reason and MUST NOT contain a credential (FR-018). (T009)
- [x] 2.7 Implement `ExposureLimits` (max_stake_per_bet, max_total_exposure, killed) in
  `src/sportsbet/execution/_base.py`. (T010)
- [x] 2.8 Implement `CredentialRef` and resolution in `src/sportsbet/execution/_credentials.py`. It carries the NAME of
  the variable and reads the secret at use. It never returns the secret to a caller, never logs it, never pickles it,
  and never puts it in `detail`. A missing variable names what was expected and stops (FR-019). Mirror the existing
  `odds_key_env` pattern. (T011)
- [x] 2.9 Test credential handling in `tests/test_execution.py`: a missing variable names the variable it wanted and
  stops, and the resolved secret appears in no return value, exception, or `repr`. Uses a sentinel value. (T012)
- [x] 2.10 Implement the `BaseVenue` ABC in `src/sportsbet/execution/_base.py` per
  the venue contract in design.md: `authenticate`, `list_markets`, `read_balance`, `place`, `read_status`,
  `cancel`, plus `key` and `can_cancel`. Docstring `place` as idempotent on `identity.ref`. A venue that cannot make
  that guarantee does not implement this contract. (T013)

## 3. User Story 1: place the value bets at a venue with an official betting API

- [x] 3.1 Test the default refuses in `tests/test_execution.py`: `place` with no confirmation returns `DRY_RUN` on every
  receipt, stakes zero, and the fake venue records zero calls. This is SC-002, and it must hold for 100% of runs that do
  not opt in. (T014)
- [x] 3.2 Test a mismatched confirmation in `tests/test_execution.py`: `confirm_stake` or `confirm_exposure` differing
  from the quote returns `REFUSED_UNCONFIRMED`, stakes zero, and the message states the REAL figures. Cover each of the
  two figures wrong alone, and both wrong. (T015)
- [x] 3.3 Test the ceilings in `tests/test_execution.py`: a stake above `max_stake_per_bet` and a batch above
  `max_total_exposure` each return `REFUSED_LIMIT` with `detail` NAMING which ceiling was hit (SC-009). Include the
  exposure ceiling being reached partway through a batch, so the placed prefix stands and the remainder refuses. (T016)
- [x] 3.4 Test the kill switch in `tests/test_execution.py`: engaged before a batch, every receipt is `REFUSED_KILLED`,
  and engaged mid-batch, the already-placed prefix stands and the remainder refuses (FR-012). (T017)
- [x] 3.5 Test the price floor in `tests/test_execution.py`: a venue price below `min_price` returns `REFUSED_PRICE` and
  stakes zero. Include the default, where `min_price` is the price the value bet was computed at (FR-013). (T018)
- [x] 3.6 Test once-only under faults in `tests/test_execution.py`. The load-bearing test (SC-003). Inject a timeout
  after the fake venue accepted but before the response landed, then retry. Inject a crash between accept and receipt,
  then re-run FROM SCRATCH with no state carried over. Both end with exactly ONE stake per identity and the second
  attempt reporting `ALREADY_PLACED` (FR-014a). Assert that no file, cache or directory holds placement state, since
  FR-015 forbids a local record that could drift. (T019)
- [x] 3.7 Test the Betfair two-reference round trip in `tests/test_betfair.py` against recorded payloads: `placeOrders`
  carries `identity.ref` as BOTH `customerRef` and `customerOrderRef`, with exactly ONE instruction per request.
  Recovery calls `listCurrentOrders(customerOrderRefs=[ref])`. Assert nothing reads `customerRef` back, since Betfair's
  docs and its interface definition contradict each other on whether it persists and the adapter treats it as unreadable
  (research D3). (T020)
- [x] 3.8 Test sequential placement in `tests/test_execution.py`: the fake venue records calls strictly in order, one at
  a time, with the running exposure checked before each. Assert the configured minimum interval is honoured. Pacing is a
  knob in seconds and carries no timing mimicry (FR-023). (T021)
- [x] 3.9 Implement `quote(venue, intents, limits)` in `src/sportsbet/execution/__init__.py`: itemise every bet, its
  stake and its price, and total the batch stake and the exposure including what is already open at the venue (FR-010,
  SC-008). (T022)
- [x] 3.10 Implement `place(venue, quote, limits, confirm_stake=None, confirm_exposure=None)` in
  `src/sportsbet/execution/__init__.py` with the refusal order from the venue contract in design.md: kill
  switch, then confirmation, then per intent sequentially kill switch, ceilings, `min_price`, place. Dry run is the
  ABSENCE of confirmation, not a flag, so no default can be misconfigured into spending. Enforced IN CODE, never in a
  docstring. (T023)
- [x] 3.11 Implement the sequential loop with the running exposure total in `src/sportsbet/execution/__init__.py`. One
  bet at a time is what makes FR-011's ceiling readable and FR-014's once-only tractable, since concurrent in-flight
  stakes make both unenforceable without a lock (research D7). (T024)
- [x] 3.12 Implement `BetfairVenue.authenticate` in `src/sportsbet/execution/_betfair.py`: certificate login at
  `identitysso-cert.betfair.com/api/certlogin` over `aiohttp` with `ssl.SSLContext.load_cert_chain`, then
  `X-Application` and `X-Authentication` on every call. Constructor takes `*_env` NAMES with name defaults, so nothing
  sensitive has a default. (T025)
- [x] 3.13 Implement `BetfairVenue.place` in `src/sportsbet/execution/_betfair.py`: `placeOrders` with `customerRef` and
  `customerOrderRef` both set to `identity.ref`, ONE instruction per request. On a duplicate inside 60 seconds Betfair
  deduplicates on `customerRef`, and past it, recover with `listCurrentOrders(customerOrderRefs=[ref])` and return
  `ALREADY_PLACED` (FR-014a). Emitting a colliding ref is our defect to prevent, since Betfair validates no uniqueness
  on it. (T026)
- [x] 3.14 Implement `BetfairVenue.list_markets` in `src/sportsbet/execution/_betfair.py` via `listMarketCatalogue` and
  `listMarketBook`. Respect 5 requests/sec per market ID. (T027)
- [x] 3.15 Add `build_venue(venue)` beside `build_bettor` in `src/sportsbet/_selection.py`, resolving `betfair` to
  `BetfairVenue()` and `venue.py:VENUE` through the existing `_load_object`, exactly as `--model` resolves
  `models.py:BETTOR`. A venue is the model case rather than the dataloader case: its URLs, notes and certificate paths
  are not a closed list of names, so no argument list describes one. Import `sportsbet.execution` LAZILY inside the
  function, since the extra is optional and `_selection.py` is imported on every run, and name the extra to install when
  it is missing (SC-010). (T027a)
- [x] 3.16 Add the `execution quote` and `execution place` commands to a new `src/sportsbet/cli/_execution.py`, taking
  `--venue` and resolving it with `build_venue`. Every parameter passed with `--`, no positional arguments. No
  `--dry-run` flag: dry run is the absence of `--confirm-stake` and `--confirm-exposure`. `place` without them prints
  the full quote, stakes nothing, and exits non-zero. (T028)
- [x] 3.17 Add `execution_venue_info`, `execution_authenticate`, `execution_quote` and `execution_place` tools to
  `src/sportsbet/mcp/_server.py`, each taking a `venue` reference resolved by `build_venue`, exactly as the existing
  tools take a `model`. `execution_place` enforces the confirmation IN CODE and states the real figures when it refuses,
  following the `prepare`/`confirm_cost` pattern from feature 005. DataFrames cross as records. (T029)
- [x] 3.18 Test the MCP refusal in `tests/test_mcp.py`: `execution_place` without confirmations refuses, and with wrong
  figures refuses and names the real ones, and with exact figures places against the fake. An agent is the caller most
  likely to skim, which is why FR-009 binds every surface. (T030)
- [x] 3.19 Extend `tests/cli/test_parity.py` to cover the execution group, asserting every US1 capability is reachable
  from the Python API, the CLI and the MCP server (SC-006). (T031)

## 4. User Story 2: follow the money after placement

- [x] 4.1 Test status reporting in `tests/test_execution.py`: bets are reported open, matched, rejected or settled,
  including a partial match. (T032)
- [x] 4.2 Test traceability in `tests/test_execution.py`: every receipt identifies the venue, the market, the selection,
  the stake, the price obtained and the value bet (FR-016). Assert the trace works FROM THE VENUE ALONE, with no local
  state, which is SC-005 and the reason identity excludes the run. (T033)
- [x] 4.3 Test cancellation honesty in `tests/test_execution.py`: a venue with `can_cancel=False` raises
  `CancellationUnsupported` rather than returning a receipt implying it cancelled (FR-008). (T034)
- [x] 4.4 Implement `BetfairVenue.read_status` in `src/sportsbet/execution/_betfair.py` via
  `listCurrentOrders(customerOrderRefs=[...])`. Chunk conservatively: the documented 250 cap applies to `betIds` and
  `marketIds`, and the limit for `customerOrderRefs` is undocumented, so do not assume it generalises (research D3).
  (T035)
- [x] 4.5 Implement `BetfairVenue.read_balance` in `src/sportsbet/execution/_betfair.py` via `getAccountFunds`,
  returning balance and open exposure. (T036)
- [x] 4.6 Implement `BetfairVenue.cancel` in `src/sportsbet/execution/_betfair.py` via `cancelOrders`, with
  `can_cancel = True`. (T037)
- [x] 4.7 Add `execution status`, `execution balance` and `execution cancel` to `src/sportsbet/cli/_execution.py`.
  (T038)
- [x] 4.8 Add `execution_read_status`, `execution_read_balance` and `execution_cancel` tools to
  `src/sportsbet/mcp/_server.py`. (T039)
- [x] 4.9 Extend `tests/cli/test_parity.py` for the US2 capabilities. (T040)

## 5. User Story 3: place at a bookmaker that publishes no API, driven by an agent

- [x] 5.1 Test that the guarantee is unreachable in `tests/test_browser.py`: `BrowserSession` is not a `BaseVenue` and
  exposes no `place`, `read_status` or `cancel`. A caller must not be able to reach a guarantee that does not exist.
  This test IS the FR-005/FR-007 resolution, so it stays permanently. (T041)
- [x] 5.2 Test actionability in `tests/test_browser.py` against a mock bet slip served over loopback: `click` on a
  DISABLED confirm control fails rather than reporting success, and the same for a hidden one. This is a permanent
  regression test and the property that decided Playwright over injected JavaScript, which silently "clicked" a disabled
  Place-bet button under test (research D4). A false success on a confirm button is a receipt that lies about money.
  (T042)
- [x] 5.3 Test the pinned session in `tests/test_browser.py`: serve a page whose odds widget re-renders on a timer,
  changing every snapshot ref. Explore, `fix` the stake and confirm locators, act after a re-render, and assert the
  pinned locators still resolve. Assert `fix` rejects a raw ref and stores no price. (T043)
- [x] 5.4 Test the block path in `tests/test_browser.py`: a page returning a block response yields `BLOCKED` and stops.
  Assert no retry with different timing, headers, or presentation (FR-023). (T044)
- [x] 5.5 Test `notes` in `tests/test_browser.py`: stored and returned verbatim through `execution_venue_info`, never
  parsed. Assert the package holds no site table, no selector pack, and no default naming a bookmaker. (T045)
- [x] 5.6 Implement `BrowserSession.__init__` in `src/sportsbet/execution/_browser.py` taking `key`, `url`, `notes`,
  `credential_env`, `user_data_dir` and `min_interval`. Parameters stored unmodified and unvalidated per Constitution
  Principle I. `notes` is ONE free-text blob rather than structured URL fields, because nothing in the library parses it
  and the only reader is the agent. (T046)
- [x] 5.7 Implement session lifecycle in `src/sportsbet/execution/_browser.py`: one `BrowserContext` per process via
  `launch_persistent_context(user_data_dir=...)`, held across tool calls so a login survives them. One instance per
  `user_data_dir`. Playwright's API is not thread-safe, so manage `start()`/`stop()` explicitly rather than using the
  documented `with sync_playwright()` idiom (research D6). (T047)
- [x] 5.8 Implement `navigate` and `snapshot` in `src/sportsbet/execution/_browser.py` returning `PageSnapshot` from
  `locator.aria_snapshot(mode='ai')`. Scope by locator or depth rather than snapshotting `body`, since it is the token
  cost of every agent turn. Note `page.accessibility.snapshot()` was removed in Playwright 1.57 and plain
  `aria_snapshot()` without `mode='ai'` omits the refs. (T048)
- [x] 5.9 Implement `click`, `type` and `select` in `src/sportsbet/execution/_browser.py`, each taking a ref and
  returning the resulting snapshot so the agent sees what its action did without a second call. Pace with
  `min_interval`. (T049)
- [x] 5.10 Implement `fix(match, locators)` and `FixedSession` in `src/sportsbet/execution/_browser.py`. Store roles and
  accessible names, NEVER refs, which go stale on the re-render an odds widget does constantly. Store NO price: the
  price is read at placement and checked against `min_price`, and pinning it would defeat the only protection between
  quoting and landing. A `FixedSession` is navigation state, not placement state, so it does not conflict with FR-015.
  (T050)
- [x] 5.11 Add `execution page read`, `execution page act` and `execution page fix` to
  `src/sportsbet/cli/_execution.py`. (T051)
- [x] 5.12 Add `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_select` and
  `browser_fix` tools to `src/sportsbet/mcp/_server.py`. They carry no site knowledge: the agent supplies which site,
  which control, which market. (T052)
- [x] 5.13 Extend `tests/cli/test_parity.py` for the US3 capabilities. (T053)

## 6. Polish and cross-cutting concerns

- [x] 6.1 Add `docs/overview/user_guide/execution.md` leading with the money and terms-of-service risks, ahead of any
  instruction to use it (FR-024). State plainly that driving a bookmaker's website breaches essentially every
  bookmaker's terms and risks account closure and loss of the balance. State that once-only and the ceilings hold on the
  API path and are the agent's on the site path. No bold, no dashes, no semicolons, and say what things do rather than
  what they do not. (T054)
- [x] 6.2 Document in `docs/overview/user_guide/execution.md` that no exchange offers a placement sandbox, and that
  Betfair's delayed application key places REAL bets on the live exchange. Users will otherwise believe the folklore and
  test with real money. (T055)
- [x] 6.3 Register the execution page in the `properdocs.yml` nav. (T056)
- [x] 6.4 The CHANGELOG is generated from the commits by `git-changelog` under the angular convention, so the `feat:`
  commits of this feature are its entries. Nothing is hand written into `CHANGELOG.md`, and the `changelog` session
  renders it at release. (T057)
- [x] 6.5 Re-run the public-API docs audit: every public name has a runnable example. Network-touching classes keep
  non-executable examples. (T058)
- [x] 6.6 Run every scenario in the validation scenarios of design.md and confirm each success criterion. (T059)
- [x] 6.7 Verify `git diff --stat -- src/sportsbet/dataloaders src/sportsbet/evaluation` is EMPTY (FR-025, SC-007). This
  is an additive feature. (T060)
- [x] 6.8 Verify SC-010 in a clean environment: `pip install sports-betting` pulls no `playwright`, and `sportsbet
  execution` reports the extra to install rather than raising `ImportError`. Install `[mcp]` alone and confirm the
  execution tools name `[execution]`. (T061)
