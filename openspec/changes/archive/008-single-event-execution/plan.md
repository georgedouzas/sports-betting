# Implementation Plan: Single-Event Execution Unit

**Branch**: `008-single-event-execution` | **Date**: 2026-07-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/008-single-event-execution/spec.md`

## Summary

Replace the multi-match batch execution path with a single execution unit that watches exactly one betting event,
logs everything to the terminal, and lets a fitted bettor decide and place one bet at the moment the model was fitted
for. The unit self-polls a dataloader for the event's evolving data across preplay, inplay, and postplay, drives a
headless browser to explore the given URLs, match the event, and place, and stakes a configured amount only after the
run is armed. It reuses the existing execution primitives (`BrowserSession`, `FixedSession`, `BetIdentity`,
`PlacementReceipt`, `build_receipts_frame`, `find_betting_moment`) and retires the batch `execute`/`quote`/`place`
stack.

## Technical Context

**Language/Version**: Python `>=3.11, <3.14`, targeting `py311`.

**Primary Dependencies**: existing only, no new dependency. The fitted bettor from `scikit-learn` estimators, `pandas`,
`playwright` (the optional `execution` extra, headless Chromium) for the browser, `click` for the CLI, `rich` plus the
standard `logging` module for the terminal log, `pandera` for the receipt schema.

**Storage**: N/A. A browser user-data directory persists the bookmaker login between runs.

**Testing**: `pytest` with `--doctest-modules`. Unit tests inject a clock and a wait, feed a fake source scripted with
preplay, inplay, and postplay snapshots, and drive a stub session and placer. No network and no real browser.

**Target Platform**: a local developer terminal through the CLI `sportsbet`, the Python API, and the MCP server.

**Project Type**: a library with a CLI and an MCP surface.

**Performance Goals**: place within a few seconds of the fitted moment. Poll the source on a configurable cadence
between log lines, defaulting to a coarse interval so a long preplay wait is cheap.

**Constraints**: exactly one event per unit. The browser runs headless with no visible window. All information goes to
the terminal. Reuse the existing execution primitives rather than a parallel stack, and retire the multi-match batch
quote-and-confirm path. A run is a no-stakes dry run unless armed.

**Scale/Scope**: one event per unit. The user runs and parallelizes units and accepts the bookmaker's response.

## Constitution Check

*GATE: checked before Phase 0 and re-checked after Phase 1. Constitution v2.2.0.*

- **I. Honor the Ecosystem Contract**: PASS. The fitted bettor is used only through its `bet(X, O)` surface. No
  estimator name, attribute, or constructor is changed. The unit is a client of the estimator, not an extension of it.
- **II. Type Safety & Schema Validation**: PASS. The placed bets come back as the existing pandera-validated receipt
  frame. The event's snapshots are validated by the existing source and dataloader schemas.
- **III. Tests & Doctest Discipline**: PASS. The runner is network- and browser-touching, so it carries no executable
  doctest, matching `BrowserSession`. Behaviour is covered by unit tests with an injected clock, a fake source, and a
  stub session and placer. No test touches the network.
- **IV. Automated Quality Gates**: PASS by construction. `pdm run formatting`, `pdm run checks`, and `pdm run tests`
  stay green on 3.11, 3.12, and 3.13 at every commit.
- **V. Documentation as a First-Class Artifact**: PASS. The runnable execution example and the execution user guide are
  rewritten for the single-event unit. `docs/generated` is regenerated, never hand-edited.
- **VI. A Library, Not an Application**: PASS. The unit handles one event and does not orchestrate a fleet, so no agent
  loop enters the package. The capability is reached from the Python API, the CLI, and the MCP server, and the parity
  test is updated to assert the new command and tool stay in step. Errors raise the named `ExecutionError` family.
  Naming and module structure follow the Code Conventions.

No violations. The Complexity Tracking table stays empty.

Note on the retirement: the multi-match batch path (`execute`, `quote`, `place`, `build_value_bet_intents`,
`select_feasible`) is recent and its removal is a behaviour change to a surface the spec deems not useful. The public
API change is recorded in the contracts and the changelog, and the gate stays green because the surfaces, tests, and
docs move with it.

## Project Structure

### Documentation (this feature)

```text
specs/008-single-event-execution/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── python-api.md
│   ├── surfaces.md
│   └── retirement.md
└── tasks.md             # Phase 2 output (/speckit-tasks, not created here)
```

### Source Code (repository root)

```text
src/sportsbet/execution/
├── __init__.py          # re-export the new runner, drop the retired batch names
├── _base.py             # KEEP: BaseVenue, BetIdentity, Placement*, ExposureLimits, build_receipts_frame
├── _browser.py          # KEEP: BrowserSession (headless), FixedSession, PageSnapshot
├── _credentials.py      # KEEP: CredentialRef, resolve
├── _factory.py          # KEEP: build_venue
├── _event.py            # NEW: execute_event and its helpers, the single-event runner
├── _place.py            # RETIRE the batch quote/place/build_value_bet_intents, keep the placing helpers reused
└── _schedule.py         # RETIRE execute/select_feasible, keep find_betting_moment where it belongs

src/sportsbet/cli/_execution.py    # replace the batch `run` with the single-event command, keep `page` explore/fix
src/sportsbet/mcp/_server.py       # replace `execution_run` with the single-event tool, keep the browser_* tools

tests/execution/     # test_event.py for the runner, trim the batch tests
tests/cli/           # update the execution command test
tests/mcp/           # update the tool and the parity test

docs/examples/execution/plot_place_value_bets.py   # rewrite around the single-event unit
docs/overview/user_guide/execution.md              # rewrite for the single-event flow
```

**Structure Decision**: Single project, `src` layout. The feature lives in the existing `execution` package, adds one
module `_event.py` for the runner, retires the two batch modules' multi-match functions, and moves the surfaces and
docs onto the new capability. The layering stays `core` to domain to surfaces, imports run downward, and the browser
stays behind the optional `execution` extra.

## Complexity Tracking

No constitution violations, so this table is empty.
