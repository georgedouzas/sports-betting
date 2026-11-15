# Tasks

Each group ends green on `pdm run formatting`, `pdm run checks`, `pdm run docs build` and `pdm run tests`, read from
the nox session summary line. Every test follows `test_<function>_<behavior>` with a one-line docstring and imports only
from public surfaces.

## 1. Venue contract

- [x] 1.1 In `execution/_base.py`, make `read_price` abstract beside `authenticate` and `place`. Make `list_markets`,
  `read_balance` and `read_status` concrete and raise `ExecutionError` naming the operation, make `cancel` raise
  `CancellationUnsupportedError`, and add a no-op `close`. Repeat the full docstring on each method. Verify with
  `tests/execution/test_base.py` covering each unsupported operation.
- [x] 1.2 Add the reference API venue to `tests/execution/conftest.py`: an in-memory price book, recorded placements,
  and switches for price drift, a limit refusal and no confirmation. Its receipts carry the book's price and a
  generated venue bet id. Verify that its own tests in `tests/execution/test_base.py` pass for each switch.
- [x] 1.3 Add the `Confirmation` dataclass to `_base.py` and re-export it. Verify with its doctest.

## 2. Browser venue

- [x] 2.1 Remove `fix`, `resolve` and `FixedSession` from `_browser.py`, add the `page_` property, and drop the
  removed names from `execution/__init__.py`. Verify that `tests/execution/test_browser.py` loses the pinning tests,
  gains a `page_` test, and that grep finds no reference to `FixedSession` in `src/` or `tests/`.
- [x] 2.2 Add `BaseBrowserVenue` in a new `execution/_browser_venue.py` with the seven abstract hooks, the concrete
  `match_selection`, and `authenticate`, `read_price`, `place` and `close` in the order from design D2. Re-export it.
  Verify that a doctest shows a subclass missing `add_selection` raising `TypeError`.
- [x] 2.3 Add the fake bookmaker pages under `tests/execution/bookmaker/` and a `served_site` fixture that serves them with
  `http.server` on a free port. The pages cover login, event, selections, slip, stake, confirm, and a confirmation that
  can appear, refuse or never appear. Add the test subclass in `tests/execution/conftest.py`. Verify that a smoke test
  navigates to the served event page.
- [x] 2.4 Test `BaseBrowserVenue` against the served site: the logged-in check passes and fails with the profile
  message, a blocked status raises `VenueBlockedError`, an unknown event and a mismatched slip give `rejected` with no
  stake typed, a price below the minimum gives `refused_price`, a confirmation gives the site's bet id and price, and a
  missing confirmation gives `refused_unconfirmed`. Verify that `tests/execution/test_browser_venue.py` passes.

## 3. Decision and moment

- [x] 3.1 Change `find_betting_moment` in `_schedule.py` to take `lead`, return kickoff minus lead, and raise
  `ExecutionError` for an in-play dataloader or a negative lead. Verify with `tests/execution/test_schedule.py` and its
  doctest.
- [x] 3.2 Add the public `decide_bet` with the private market-to-selection helper for `home_win`, `away_win`, `draw`,
  and over and under lines. It takes the first backed market in the bettor's order, sets the minimum price from
  `find_latest_odds_column`, and returns `None` with a reason when no market is backed, the market is unknown, or no
  price exists. Remove `FALLBACK_PRICE`. Re-export `decide_bet`. Verify that the decision tests cover each spec
  scenario: away win, over line, two markets backed, no price, no value.

## 4. Ledger and run record

- [x] 4.1 Add `_Ledger` in `execution/_ledger.py`: append JSON Lines with flush and fsync, validate on read with a
  pandera schema, skip a torn last line with a log line, and return the latest status per ref. Verify with
  `tests/execution/test_ledger.py` covering placed, attempted, refused_unconfirmed, dry_run and a torn line.
- [x] 4.2 Add `RunOutcome`, the steps schema and `EventRunRecord` to `_base.py`, and re-export them. Verify with a
  doctest building an empty record and a test that an invalid steps frame fails validation.

## 5. Driver

- [x] 5.1 Rewrite `execute_event` in `_event.py` as the thin driver from design D3 and D4. It takes `venue: BaseVenue`,
  `stake`, `ledger_path`, `lead`, `poll`, `live`, `clock` and `wait`, and it loses `urls` and `placer`. It returns
  `EventRunRecord` and closes the venue in `finally`. Verify that `tests/execution/test_event.py` is rewritten against
  the reference venue and fake clock and covers every `execution/event-run` scenario, including fresh data at the
  moment, an event leaving the fixtures, the source failing at the moment, a moment already passed, and the three
  ledger reruns.
- [x] 5.2 Add `tests/execution/test_contract.py`, which runs the same dry run and armed run against the reference API
  venue and the served browser venue, and asserts that both runs record the same steps and outcome. Verify that it
  passes.
- [x] 5.3 Add a test that the ledger and the run record from an armed run with a venue reading a token from a named
  variable do not contain the token's value. Verify that it passes.

## 6. Surfaces

- [x] 6.1 Update `cli/_execution.py`: `run` takes `--lead --poll --ledger` and drops `--url`, prints steps and the
  receipt, and writes `run_steps.csv` and `receipts.csv`. Remove `page fix`. Let `page read` and `page act` accept a
  `BaseBrowserVenue`, and let `page act` check its action options before opening the browser. Let `venue`, `markets`,
  `balance`, `status` and `cancel` accept any `BaseVenue`. Make the execution commands' error reporting catch
  `ExecutionError` and print it as a red panel with exit status 1, instead of a traceback. Verify with
  `tests/cli/test_execution.py`, including an unsupported `balance` that exits 1 with the message.
- [x] 6.2 Update `mcp/_server.py`: `execution_run` mirrors the command and returns `{outcome, moment, steps,
  receipts}`, `browser_fix` is removed, and `_open_session` and `_load_venue` accept browser venues. Verify with
  `tests/mcp/test_mcp.py`.
- [x] 6.3 Update `tests/cli/test_parity.py` and `tests/mcp/test_parity_surfaces.py` for the new parameter sets and
  assert that `page fix` and `browser_fix` no longer exist. Verify that both pass.

## 7. Documentation

- [x] 7.1 Rewrite `docs/overview/user_guide/execution.md` in the constitution's style: risks once, preplay only, the
  lead time, dry run and armed, the ledger, the run record, the API venue contract, and writing a browser venue with
  the exploring commands. The main example is a markdown-exec block that defines an API venue and a custom source with
  one match dated tomorrow inline, and runs a dry run with an injected clock. The browser subclass is reference code.
  Verify that `pdm run docs build` executes the block and shows the run record.
- [x] 7.2 Update `docs/interfaces/cli.md` and `docs/interfaces/agents.md` for the changed command and tool and the
  removed `page fix` and `browser_fix`. Verify that `pdm run docs build` passes and grep finds no `page fix` or
  `browser_fix` in `docs/`.

## 8. Integration and phase 1 acceptance

- [x] 8.1 Run the full gate in order from a clean nox environment. Verify that all four session summary lines report
  success.
- [ ] 8.2 Manual acceptance with a real bookmaker: write a `BaseBrowserVenue` subclass for the user's own bookmaker,
  log in once with a visible browser on its profile, run `sportsbet execution run` as a dry run on a real upcoming
  match, and confirm that the run record shows the venue's real price. Then run it armed with the bookmaker's minimum
  stake, and confirm that the receipt's venue bet id matches the bet in the account history and that a rerun records
  `already_placed`. Record the outcome in the PR description.

## Workflow follow-up

- Archive this change once 8.2 has passed and the PR is merged, so the two capabilities land in `openspec/specs/`.
- Propose phase 2, multiple events, as its own change only after 8.2 has passed.
