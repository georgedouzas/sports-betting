---

description: "Task list for the single-event execution unit"
---

# Tasks: Single-Event Execution Unit

**Input**: Design documents from `specs/008-single-event-execution/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Included. The constitution gates on `pytest` with `--doctest-modules`, and research D11 and quickstart.md
call for coverage. Tests follow the conventions: mirrored path, `test_<function>_<behavior>`, one-line docstring,
public API, no network.

**Organization**: Grouped by user story. US1 is the MVP, the whole runner. US2 hardens the terminal log, US3 the
one-event guardrail.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel, different files, no dependency on an incomplete task
- **[Story]**: US1, US2, US3

## Path Conventions

Single project, `src/` and `tests/` at the repository root, per plan.md.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: The offline test scaffolding the runner tests need.

- [x] T001 [P] Add the shared test fakes in `tests/execution/conftest.py`: a fake dataloader scripted through preplay,
  inplay, and postplay snapshots for one event with a kickoff and `target_event_status_`/`target_event_time_`, a stub
  `BrowserSession` (navigate, snapshot, fix, no network), a recording placer, and an injected clock and wait. Create
  the empty `tests/execution/test_event.py` module.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Retire the multi-match batch path so the single-event unit is built on a clean base (FR-014). Per
contracts/retirement.md. The tree stays green at the end of the phase.

**⚠️ CRITICAL**: No user story work begins until this phase is complete and the gate is green.

- [x] T002 [P] In `src/sportsbet/execution/_schedule.py`, remove `execute` and `select_feasible`. Keep
  `find_betting_moment`, `_now`, and the `Clock`, `Wait`, `Placer`, `Staking`, `Scheduled` type aliases and
  `TOLERANCE`, which the runner reuses.
- [x] T003 [P] Delete `src/sportsbet/execution/_place.py` (the batch `quote`, `place`, `build_value_bet_intents`,
  `_build_dry_run_receipts`, and helpers). Nothing from it is kept.
- [x] T004 In `src/sportsbet/execution/__init__.py`, drop `execute`, `select_feasible`, `quote`, `place`, and
  `build_value_bet_intents` from the imports and `__all__`. Keep the rest. (Depends on T002, T003.)
- [x] T005 [P] In `src/sportsbet/cli/_execution.py`, remove the `run`, `quote`, and `place` commands and their imports
  and helpers (`run_execute`, `run_place`, `run_quote`, `_write_quote`, `_read_quote`, `_render_quote`,
  `_build_limits`). Keep `venue`, `markets`, `balance`, `status`, `cancel`, and the `page` group.
- [x] T006 [P] In `src/sportsbet/mcp/_server.py`, remove the `execution_run`, `execution_quote`, and `execution_place`
  tools and their helpers (`_build_intents`, `_to_quote_records`, `_build_quote`, `_load_fixtures`, `_place_run`). Keep
  the other `execution_*` and all `browser_*` tools.
- [x] T007 [P] In `tests/mcp/test_parity_surfaces.py`, drop the `(["execution","quote"], "execution_quote")` and
  `(["execution","place"], "execution_place")` pairs and the `execution run` pair for now, and any `SPELLED`/`RENAMED`
  entries for removed params.
- [x] T008 [P] Delete the batch tests: the `execute`/`quote`/`place` cases in `tests/execution/test_execution.py` and
  `tests/execution/test_schedule.py` (keep the `find_betting_moment` tests), and any removed-command or removed-tool
  cases in `tests/cli/test_execution.py` and `tests/mcp/test_mcp.py`.
- [x] T009 Run the gate (`pdm run formatting`, `pdm run checks`, `pdm run tests`) and confirm green on 3.11, 3.12, and
  3.13 with the batch path gone, reading the nox session summary lines.

**Checkpoint**: the execution package is trimmed to its reusable primitives and green.

---

## Phase 3: User Story 1 - Set up from URLs, place at the right moment (Priority: P1) 🎯 MVP

**Goal**: The single-event runner. Given a fitted bettor, a dataloader, a headless session, candidate URLs, and a
stake, it explores and matches the event, ensures login, monitors, and at the fitted moment places one bet if the
model finds value.

**Independent Test**: With the fakes from T001, an armed run against a preplay model places exactly one receipt for the
model's selection at the configured stake at the preplay moment, and a dry run places none.

### Tests for User Story 1

- [x] T010 [US1] In `tests/execution/test_event.py`, write the runner tests, one behavior each, mirroring quickstart
  Scenarios 1 to 5: `test_execute_event_places_one_bet_when_armed`,
  `test_execute_event_places_nothing_without_value`, `test_execute_event_dry_run_stakes_nothing`,
  `test_execute_event_stops_when_moment_passed`, `test_execute_event_stops_when_no_url_matches`. Use the injected clock
  and wait and the fakes. They must fail before T011.

### Implementation for User Story 1

- [x] T011 [US1] Create `src/sportsbet/execution/_event.py` with `execute_event` and the default placer, per
  contracts/python-api.md: explore the URLs and match the event, pin controls as a `FixedSession`, ensure login, poll
  the dataloader to the moment from `find_betting_moment`, apply `bettor.bet` to the one-row features and odds, place
  once through the placer when armed or record a dry-run receipt, and return the receipts frame. (Depends on T002 for
  `find_betting_moment` and `Placer`.)
- [x] T012 [US1] Re-export `execute_event` (and `Placer` if made public) from `src/sportsbet/execution/__init__.py` and
  `__all__`. (Depends on T011.)
- [x] T013 [US1] Add the single-event `execution run` command in `src/sportsbet/cli/_execution.py` with `--venue`
  (browser session), `-d`, `-b`, `--event`, `--stake`, repeatable `--url`, `--live`, `--poll`, and `-o`, placing
  through the default placer over the controls pinned by `execution page fix`. (Depends on T011, T012.)
- [x] T014 [US1] Add the `execution_run` MCP tool in `src/sportsbet/mcp/_server.py` mirroring the command params
  (`venue`, `dataloader`, `bettor`, `event`, `stake`, `urls`, `live`, `poll`, `output`). (Depends on T011, T012.)
- [x] T015 [US1] Restore the `(["execution","run"], "execution_run")` pair and the new params in
  `tests/mcp/test_parity_surfaces.py`. (Depends on T013, T014.)
- [x] T016 [P] [US1] Add the CLI test `test_run_*` for the dry-run path in `tests/cli/test_execution.py`, offline with
  the fakes.
- [x] T017 [P] [US1] Add the MCP test for `execution_run` in `tests/mcp/test_mcp.py`, offline with the fakes.
- [x] T018 [US1] Run the gate and confirm green on 3.11, 3.12, and 3.13.

**Checkpoint**: the MVP works end to end for one event, from all three surfaces.

---

## Phase 4: User Story 2 - See the event as it unfolds (Priority: P2)

**Goal**: The terminal log carries the event, its status across preplay, inplay, and postplay, the price where
available, and the decision at each step, and announces the bet before placing.

**Independent Test**: A dry run against a scripted event produces a log with the status transitions and a pre-place
announce line, with no stake placed.

### Tests for User Story 2

- [x] T019 [US2] In `tests/execution/test_event.py`, write `test_execute_event_logs_status_and_decision` asserting,
  with `caplog` on the `sportsbet.execution` logger, the status transitions, the price where available, and the
  pre-place announce line.

### Implementation for User Story 2

- [x] T020 [US2] Emit the per-step log (event, status, price, decision) and the pre-place line on the
  `sportsbet.execution` logger in `src/sportsbet/execution/_event.py` (FR-009, FR-010).
- [x] T021 [US2] Ensure the `execution run` command wraps the run in the terminal log handler (reuse
  `_logging_to_terminal`) in `src/sportsbet/cli/_execution.py`.
- [x] T022 [US2] Run the gate and confirm green.

**Checkpoint**: the run is legible in the terminal before money moves.

---

## Phase 5: User Story 3 - Stay within one event (Priority: P3)

**Goal**: The unit takes exactly one event and one bettor and places at most one bet over the run.

**Independent Test**: After the unit acts on its event, no later poll or price move produces a second bet, and the
inputs cannot describe more than one event.

### Tests for User Story 3

- [x] T023 [US3] In `tests/execution/test_event.py`, write `test_execute_event_places_at_most_one_bet` and
  `test_execute_event_takes_one_event_and_one_bettor`.

### Implementation for User Story 3

- [x] T024 [US3] Enforce the at-most-one-placement guarantee and the single-event, single-bettor inputs in
  `src/sportsbet/execution/_event.py` (FR-001, FR-008, US3).
- [x] T025 [US3] Run the gate and confirm green.

**Checkpoint**: the one-event promise is enforced and tested.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T026 [P] Rewrite the runnable example around the single-event unit in
  `docs/examples/execution/plot_place_value_bets.py` (rename to `plot_single_event.py` if clearer), offline where it
  can be.
- [x] T027 [P] Rewrite `docs/overview/user_guide/execution.md` for the single-event flow: explore, log in, monitor,
  dry run versus `--live`, and the one-event responsibility.
- [x] T028 Regenerate `docs/generated` through the docs build. Never hand-edit it.
- [x] T029 Land the retirement and the feature as conventional commits, with a `BREAKING CHANGE` footer listing the
  removed public names from contracts/retirement.md, so `pdm run changelog` records the API change.
- [x] T030 Run the full gate on 3.11, 3.12, and 3.13 and walk quickstart.md Scenarios 1 to 6. Confirm green.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: after Setup. Blocks all user stories. Ends green with the batch path removed.
- **User Stories (Phase 3 to 5)**: after Foundational. US1 is the MVP and must land first, since US2 and US3 harden
  the runner it creates.
- **Polish (Phase 6)**: after the user stories that are being shipped.

### User Story Dependencies

- **US1 (P1)**: after Foundational. Creates `_event.py` and the surfaces.
- **US2 (P2)**: builds on US1's runner, hardening the log. Not independent of US1 in code, only in test.
- **US3 (P3)**: builds on US1's runner, enforcing the guardrail.

### Within Each User Story

- Tests are written first and fail before implementation.
- The runner (`_event.py`) before the surfaces (CLI, MCP).
- The surfaces before the parity test restore.

### Parallel Opportunities

- Phase 2: T002, T003, T005, T006, T007, T008 touch different files and run in parallel. T004 waits on T002 and T003.
- Phase 3: T016 and T017 run in parallel (different test files). T010 is one file, so its cases are one task.
- Phase 6: T026 and T027 run in parallel.

---

## Parallel Example: Phase 2

```bash
# Retire the batch path across independent files together:
Task: "Trim src/sportsbet/execution/_schedule.py"
Task: "Delete src/sportsbet/execution/_place.py"
Task: "Remove batch commands from src/sportsbet/cli/_execution.py"
Task: "Remove batch tools from src/sportsbet/mcp/_server.py"
Task: "Drop the retired pairs from tests/mcp/test_parity_surfaces.py"
Task: "Delete the batch tests"
# Then, after the two source removals: update src/sportsbet/execution/__init__.py
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 Setup.
2. Phase 2 Foundational, retire the batch path, gate green.
3. Phase 3 User Story 1, the runner and its surfaces.
4. **STOP and VALIDATE**: run quickstart Scenarios 1 to 5 offline with the fakes, and a manual dry run.

### Incremental Delivery

1. Setup and Foundational, the base is clean and green.
2. US1, the single-event runner, MVP.
3. US2, the terminal log is complete.
4. US3, the one-event guardrail is enforced.
5. Polish, docs, changelog, full gate.

---

## Notes

- [P] tasks are different files with no incomplete dependency.
- Every phase ends with the gate green on 3.11, 3.12, and 3.13, read from the nox session summary, not a piped exit
  code.
- The browser stays headless with no live-preview window. All information goes to the terminal.
- The stake is the configured parameter. The bettor decides whether to bet and the selection, never the stake.
- Commit after each logical group. Do not add a Claude co-author trailer.
