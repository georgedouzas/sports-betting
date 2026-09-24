# Implementation Plan: Surfaces that reach the whole library

**Branch**: `005-agent-facing-surfaces` | **Date**: 2026-07-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/005-agent-facing-surfaces/spec.md`

## Summary

Three moves, in order, each shippable on its own:

1. **Delete the GUI.** 1,951 lines — a quarter of the codebase — with **zero tests**, excluded from the test run by name,
   pinned to an exact framework version it cannot safely leave, and needing a Node toolchain. It reaches one sport.
2. **Give the CLI a configured dataloader.** This is the *actual* defect. The config hands over a `DATALOADER_CLASS` — a
   **class** — so no source can be attached, so no credential can be carried, so basketball is unreachable and paid odds
   are unreachable. Handing over an **instance** fixes every one of those at once.
3. **Add an MCP server**, as an optional extra, so an assistant can drive the library. The assistant lives **outside**
   the package; no model, no model key and no model call enters it.

The headline is not "remove the GUI". It is that **two of the three surfaces reach about a tenth of the library**, and
the constitution says they must reach all of it. Deleting the surface that reaches least, and repairing the one that can
reach everything, is what makes Principle I true for the first time since sources became injectable.

## Technical Context

**Language/Version**: Python `>=3.11, <3.14`, targeting `py311`

**Primary Dependencies**: `click` and `rich` (CLI, already present). **`mcp`** becomes a new **optional** extra —
exactly the slot `gui` occupied, so the count of optional extras does not even change. **No new required dependency**
(SC-006, FR-008). Removed: `reflex==0.7.0`, `reflex-ag-grid`, `nest-asyncio`, and the Node toolchain they dragged in.

**Storage**: Unchanged. The store, the sources and the evaluation layer are not touched.

**Testing**: `pytest`, branch coverage, `pytest-randomly`, `--doctest-modules`, socket guard. **The `--ignore` for the
GUI disappears**, so the test run stops excluding a quarter of the package (SC-004). The MCP server is tested, unlike
the surface it replaces (FR-009).

**Target Platform**: A library with three surfaces: the Python API, the `sportsbet` CLI, and a `sportsbet-mcp` server.

**Project Type**: Library (`src/`-based).

**Performance Goals**: None. Every surface delegates to the same core; nothing here is on a hot path.

**Constraints**:

- **The core must not change.** `src/sportsbet/datasets/**` and `src/sportsbet/evaluation/**` show a **zero diff**
  (FR-015, SC-007). This is a change to how the library is *reached*, not to what it *does*.
- **Spending cannot be a surprise.** Any surface that can fetch metered odds must make the exact cost available first
  (FR-007, SC-003). The library already computes it exactly and for free; no surface may skip it.
- **No model in the box** (FR-014, SC-006).
- No test touches the network. No executable doctest on a network-touching class.

**Scale/Scope**: −1,951 lines (the GUI), ~+40 lines changed in the CLI, ~+200 lines for the MCP server and its tests.
**The feature is net negative in size and strictly positive in capability**, which is the shape a healthy refactor has.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design — still passing.*

A reviewer will read "delete a delivery surface" as an attack on Principle I. It is the opposite, and this is the most
important row in the table.

| Principle | Status | How |
| --- | --- | --- |
| **I. scikit-learn-compatible API** (*"the three delivery surfaces MUST expose the same underlying capabilities without one surface holding logic the others cannot reach"*) | **PASS — this feature is what makes it true** | Today the rule is **violated**: the Python API reaches 2 sports, 3 leagues and 4 sources; the CLI and the GUI reach **soccer via football-data, and nothing else**. Deleting the surface that reaches least, and giving the CLI a configured dataloader, means **every** surface reaches **every** sport and source. The MCP server is built on the same contract, so it inherits parity rather than reinventing it. |
| **II. Type safety & schema validation** | PASS | Full annotations; `mypy` clean. No DataFrame crosses a new boundary — the MCP server returns what the CLI already prints. |
| **III. Test coverage & doctest discipline** (*"Every behavioral change MUST ship with tests"*) | **PASS — this feature repairs a standing violation** | `addopts` contains `--ignore=src/sportsbet/gui`: a quarter of the package has been exempt from pytest, coverage **and** doctests since it was written. Removing it deletes the exemption; the MCP server that replaces it **is** tested. |
| **IV. Automated quality gates** | PASS | The GUI's `bandit` suppressions (B404/B603/B607 — it shells out to `node`) go with it, so the security gate gets *stricter*, not looser. |
| **V. Documentation first-class** | PASS | README loses the GUI screenshots and gains the agent story; the CLI page documents the new config; the MCP server is documented. `CHANGELOG.md` records **two breaking changes**. The public-API docs audit must still report an example for every public name. |

**No violations. Complexity Tracking is empty and omitted.**

## Project Structure

### Documentation (this feature)

```text
specs/005-agent-facing-surfaces/
├── plan.md              # This file
├── spec.md
├── research.md          # D1-D6: the survey, and why an agent goes outside the box
├── data-model.md        # The config contract, before and after
├── quickstart.md
├── contracts/
│   ├── cli.md           # The configuration contract
│   └── mcp.md           # The tool set, and the spending rule
├── checklists/
└── tasks.md             # /speckit-tasks output
```

### Source Code (repository root)

```text
DELETED
src/sportsbet/gui/                    # 1,951 lines, zero tests, one sport

CHANGED
src/sportsbet/cli/_utils.py           # get_dataloader_cls + get_param_grid -> get_dataloader
src/sportsbet/cli/_data.py            # params, prepare, odds_types, training, fixtures
src/sportsbet/cli/_betting.py         # backtest, bet
tests/conftest.py                     # the CONFIG string: the contract it tests is the one that changed
pyproject.toml                        # -gui extra, -sportsbet-gui, -pytest ignore, -bandit skips, +mcp extra
README.md                             # -screenshots, +the agent story
docs/generate_api.py                  # -the GUI
CHANGELOG.md                          # two breaking changes, with before/after

NEW
src/sportsbet/mcp/                    # the server: ~150 lines
tests/test_mcp.py
docs/overview/user_guide/agent.md     # driving the library with an assistant
```

**Explicitly NOT touched.** A diff against either means this stopped being a surfaces change:

```text
src/sportsbet/datasets/**
src/sportsbet/evaluation/**
```

**Structure Decision**: The existing `src/`-based layout. `src/sportsbet/mcp/` sits beside `src/sportsbet/cli/` as a
peer: a surface, not a layer. It imports the public API and nothing private, which is the same discipline the CLI keeps
and the reason a surface can be added or removed without the core noticing.

## Implementation Phases

Each phase ends green: `pdm run formatting`, `pdm run checks`, `pdm run tests`.

### Phase 1 — Delete the GUI (US3 — P1)

Self-contained, and it makes the test configuration honest: after it, **the test run excludes nothing** (SC-004), and
`bandit` stops skipping three checks it only skipped for the GUI's `node` launcher.

**Checkpoint**: `--ignore` is gone from `addopts`, the suite is green, and the package installs with no Node.

### Phase 2 — The CLI takes a configured dataloader (US1 — P1)

The actual bug fix. `get_dataloader_cls` + `get_param_grid` become one `get_dataloader`, and every consumer follows.

`tests/conftest.py`'s `CONFIG` string **changes with it, and that is correct here** — unlike feature 004, this feature's
*purpose* is to change the CLI contract, so the tests of that contract must change too. What must **not** change is any
test of the core.

The `params` command is the subtlety: it exists so a user can learn what to select **before** selecting it, so
`param_grid=None` must remain a valid config and `params` must still work.

**Checkpoint**: the CLI prepares and extracts an NBA dataset with a paid odds key — something it has never been able to
do.

### Phase 3 — The MCP server (US2 — P1)

The tools, and the rule that an assistant cannot spend by accident.

**Checkpoint**: an assistant goes from "what exists?" to "here are the value bets" with no Python written by the user,
and cannot buy metered odds without the cost having been surfaced.

### Phase 4 — Docs, and the breaking changes stated plainly

README, the CLI page, a new agent page, and a changelog entry with a before/after for **both** breaks. Re-run the
public-API docs audit — every public name must still have a runnable example.

## Risks

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| **Someone loses the GUI and wanted it.** | A real capability disappears for a real user. | Say so plainly rather than pretend the trade is free (spec Assumptions). The replacement reaches *more* of the library, not less. If a GUI is wanted later it belongs in its own package, built on this library like any other consumer. |
| **The config break strands existing users.** | An upgrade fails with an obscure error. | The old contract must fail with a message that **says what to change** (FR-002 edge case), and the changelog carries a before/after snippet. One clean break, not two. |
| **An assistant spends a user's odds credits by accident.** | Real money, silently. | The spending rule (FR-007): the cost is exact, free to compute, and must be surfaced before any fetch. Designed into the tool signatures, not left to the assistant's good manners. |
| **The MCP server drifts from the CLI, recreating the parity bug.** | Two surfaces, two capabilities — exactly the disease being cured. | The server reuses the CLI's **configuration contract**, so a dataloader is described once and both surfaces read it. See research D3. |
| **Scope creep into the core.** | A "surfaces" change quietly becomes a rewrite. | `datasets/**` and `evaluation/**` must show a zero diff. If they need to change, stop and report. |
