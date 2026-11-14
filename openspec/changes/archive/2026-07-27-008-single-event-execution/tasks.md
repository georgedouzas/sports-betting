# Tasks

## 1. Setup

- [x] 1.1 Add the shared test fakes in `tests/execution/conftest.py`: a fake dataloader scripted through preplay,
  inplay, and postplay snapshots for one event with a kickoff and `target_event_status_`/`target_event_time_`, a stub
  `BrowserSession` (navigate, snapshot, fix, no network), a recording placer, and an injected clock and wait. Create the
  empty `tests/execution/test_event.py` module. (T001)

## 2. Retire the batch path

- [x] 2.1 In `src/sportsbet/execution/_schedule.py`, remove `execute` and `select_feasible`. Keep `find_betting_moment`,
  `_now`, and the `Clock`, `Wait`, `Placer`, `Staking`, `Scheduled` type aliases and `TOLERANCE`, which the runner
  reuses. (T002)
- [x] 2.2 Delete `src/sportsbet/execution/_place.py` (the batch `quote`, `place`, `build_value_bet_intents`,
  `_build_dry_run_receipts`, and helpers). Nothing from it is kept. (T003)
- [x] 2.3 In `src/sportsbet/execution/__init__.py`, drop `execute`, `select_feasible`, `quote`, `place`, and
  `build_value_bet_intents` from the imports and `__all__`. Keep the rest. (Depends on T002, T003.) (T004)
- [x] 2.4 In `src/sportsbet/cli/_execution.py`, remove the `run`, `quote`, and `place` commands and their imports and
  helpers (`run_execute`, `run_place`, `run_quote`, `_write_quote`, `_read_quote`, `_render_quote`, `_build_limits`).
  Keep `venue`, `markets`, `balance`, `status`, `cancel`, and the `page` group. (T005)
- [x] 2.5 In `src/sportsbet/mcp/_server.py`, remove the `execution_run`, `execution_quote`, and `execution_place` tools
  and their helpers (`_build_intents`, `_to_quote_records`, `_build_quote`, `_load_fixtures`, `_place_run`). Keep the
  other `execution_*` and all `browser_*` tools. (T006)
- [x] 2.6 In `tests/mcp/test_parity_surfaces.py`, drop the `(["execution","quote"], "execution_quote")` and
  `(["execution","place"], "execution_place")` pairs and the `execution run` pair for now, and any `SPELLED`/`RENAMED`
  entries for removed params. (T007)
- [x] 2.7 Delete the batch tests: the `execute`/`quote`/`place` cases in `tests/execution/test_execution.py` and
  `tests/execution/test_schedule.py` (keep the `find_betting_moment` tests), and any removed-command or removed-tool
  cases in `tests/cli/test_execution.py` and `tests/mcp/test_mcp.py`. (T008)
- [x] 2.8 Run the gate (`pdm run formatting`, `pdm run checks`, `pdm run tests`) and confirm green on 3.11, 3.12, and
  3.13 with the batch path gone, reading the nox session summary lines. (T009)

## 3. User Story 1: set up from URLs, place at the right moment

- [x] 3.1 In `tests/execution/test_event.py`, write the runner tests, one behavior each, mirroring the validation
  scenarios 1 to 5 of design.md: `test_execute_event_places_one_bet_when_armed`,
  `test_execute_event_places_nothing_without_value`, `test_execute_event_dry_run_stakes_nothing`,
  `test_execute_event_stops_when_moment_passed`, `test_execute_event_stops_when_no_url_matches`. Use the injected clock
  and wait and the fakes. They must fail before T011. (T010)
- [x] 3.2 Create `src/sportsbet/execution/_event.py` with `execute_event` and the default placer, per the Python API
  contract in design.md: explore the URLs and match the event, pin controls as a `FixedSession`, ensure login, poll the
  dataloader to the moment from `find_betting_moment`, apply `bettor.bet` to the one-row features and odds, place once
  through the placer when armed or record a dry-run receipt, and return the receipts frame. (Depends on T002 for
  `find_betting_moment` and `Placer`.) (T011)
- [x] 3.3 Re-export `execute_event` (and `Placer` if made public) from `src/sportsbet/execution/__init__.py` and
  `__all__`. (Depends on T011.) (T012)
- [x] 3.4 Add the single-event `execution run` command in `src/sportsbet/cli/_execution.py` with `--venue` (browser
  session), `-d`, `-b`, `--event`, `--stake`, repeatable `--url`, `--live`, `--poll`, and `-o`, placing through the
  default placer over the controls pinned by `execution page fix`. (Depends on T011, T012.) (T013)
- [x] 3.5 Add the `execution_run` MCP tool in `src/sportsbet/mcp/_server.py` mirroring the command params (`venue`,
  `dataloader`, `bettor`, `event`, `stake`, `urls`, `live`, `poll`, `output`). (Depends on T011, T012.) (T014)
- [x] 3.6 Restore the `(["execution","run"], "execution_run")` pair and the new params in
  `tests/mcp/test_parity_surfaces.py`. (Depends on T013, T014.) (T015)
- [x] 3.7 Add the CLI test `test_run_*` for the dry-run path in `tests/cli/test_execution.py`, offline with the fakes.
  (T016)
- [x] 3.8 Add the MCP test for `execution_run` in `tests/mcp/test_mcp.py`, offline with the fakes. (T017)
- [x] 3.9 Run the gate and confirm green on 3.11, 3.12, and 3.13. (T018)

## 4. User Story 2: see the event as it unfolds

- [x] 4.1 In `tests/execution/test_event.py`, write `test_execute_event_logs_status_and_decision` asserting, with
  `caplog` on the `sportsbet.execution` logger, the status transitions, the price where available, and the pre-place
  announce line. (T019)
- [x] 4.2 Emit the per-step log (event, status, price, decision) and the pre-place line on the `sportsbet.execution`
  logger in `src/sportsbet/execution/_event.py` (FR-009, FR-010). (T020)
- [x] 4.3 Ensure the `execution run` command wraps the run in the terminal log handler (reuse `_logging_to_terminal`) in
  `src/sportsbet/cli/_execution.py`. (T021)
- [x] 4.4 Run the gate and confirm green. (T022)

## 5. User Story 3: stay within one event

- [x] 5.1 In `tests/execution/test_event.py`, write `test_execute_event_places_at_most_one_bet` and
  `test_execute_event_takes_one_event_and_one_bettor`. (T023)
- [x] 5.2 Enforce the at-most-one-placement guarantee and the single-event, single-bettor inputs in
  `src/sportsbet/execution/_event.py` (FR-001, FR-008, US3). (T024)
- [x] 5.3 Run the gate and confirm green. (T025)

## 6. Polish and cross-cutting concerns

- [x] 6.1 Rewrite the runnable example around the single-event unit in
  `docs/examples/execution/plot_place_value_bets.py` (rename to `plot_single_event.py` if clearer), offline where it can
  be. (T026)
- [x] 6.2 Rewrite `docs/overview/user_guide/execution.md` for the single-event flow: explore, log in, monitor, dry run
  versus `--live`, and the one-event responsibility. (T027)
- [x] 6.3 Regenerate `docs/generated` through the docs build. Never hand-edit it. (T028)
- [x] 6.4 Land the retirement and the feature as conventional commits, with a `BREAKING CHANGE` footer listing the
  removed public names from the retirement ledger in design.md, so `pdm run changelog` records the API change. (T029)
- [x] 6.5 Run the full gate on 3.11, 3.12, and 3.13 and walk the validation scenarios 1 to 6 of design.md. Confirm
  green. (T030)
