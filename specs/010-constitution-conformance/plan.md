# Implementation Plan: Constitution conformance

**Branch**: `010-constitution-conformance` | **Date**: 2026-09-21 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/010-constitution-conformance/spec.md`

## Summary

Bring the existing tree to the constitution as it stands at version 9.0.0. Nothing the library does changes. What
changes is what a contributor reads: every public name documents what it takes, returns, and raises, every name a
package keeps to itself is named as private, no comment stands in for a name, every example a reader sees has been run
by the build, and the Project Profile describes the repository it claims to describe.

The approach is the sweep itself, ordered so that each pass leaves the gate green and no pass rewrites lines the next
one touches. Enforcement afterwards is review plus what the gate already covers, since the maintainer declined a new
tool for it.

## Technical Context

**Language/Version**: Python `>=3.11, <3.14`, targeting `py311`

**Primary Dependencies**: `scikit-learn`, `pandas`, `pandera`, `click`, `rich`, `aiohttp`, with the optional `mcp` and
`execution` extras

**Storage**: N/A, the library reads feeds and returns frames

**Testing**: `pytest` with branch coverage, randomized ordering, and `--doctest-modules`, so every docstring example is
a test. A `network` marker deselects the live-feed tests by default.

**Target Platform**: any platform the supported Pythons run on

**Project Type**: library, with a CLI and an MCP server as additional delivery surfaces

**Performance Goals**: N/A, no runtime behaviour changes

**Constraints**: the public behaviour of the library MUST NOT change, so the existing test suite passes unchanged. The
full gate MUST be green at the end of each user story.

**Scale/Scope**: 53 source modules, 45 test modules, 192 public functions and classes, 95 re-exported names across 7
packages

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Section | Verdict |
| --- | --- |
| Contract | PASS. The rename touches only names no package re-exports, so no fixed estimator name moves |
| Types | PASS. No signature changes, and a rename is name-only |
| Schema | PASS. No boundary changes |
| Tests | PASS by vacancy. A test that has to change is the signal a behaviour change crept in |
| Gates | PASS. The plan adds a check rather than a suppression |
| Documentation | PASS. User Story 4 is what makes it true |
| Library | PASS. Nothing moves in or out of the library |
| Style | PASS. It governs what this plan writes, too |
| Naming | PASS. The rename adds an underscore and changes nothing else |
| Structure | PASS. Removing the type-checking guards is how the no-papering rule is honoured |
| Surface | This is User Story 2 |
| Docstrings | This is User Story 1 |
| Comments | This is User Story 3 |
| Suppressions | PASS today. All 8 carry a rule code and a reason |
| Errors | Checked as part of User Story 5 |
| Control Flow | Out of scope. See the note below |
| Duplication | PASS. The conformance check is the one place each rule is enforced |
| Credentials | PASS today, unchanged |
| Toolchain | PASS. The check adds no dependency, since it uses the standard library |

No violations to justify, so Complexity Tracking stays empty. The one judgement worth recording: `Control Flow` is the
single section this feature does not sweep, because its rules are the only ones that cannot be honoured without
changing what code does, and the spec forbids that.

## Project Structure

### Documentation (this feature)

```text
specs/010-constitution-conformance/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output, the rule model
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── public-surface.md  # Phase 1 output, what must not change
├── checklists/
│   └── requirements.md
├── spec.md
└── tasks.md             # Phase 2 output, not created here
```

### Source Code (repository root)

```text
src/sportsbet/
├── __init__.py          # front page, names every public surface
├── __main__.py
├── core/                # shared leaves, the only package with __all__ today
├── sources/             # 64 public names, the largest docstring and example surface
│   ├── _common/
│   ├── _odds/
│   └── _stats/
├── dataloaders/         # 13 public names
├── evaluation/          # 22 public names
├── execution/           # 39 public names, and the four type-checking guards
├── cli/                 # 28 public names
└── mcp/                 # 22 public names

tests/                   # mirrors the source tree, 45 modules

docs/                    # pages already execute their blocks
```

**Structure Decision**: the layout does not change. This feature edits the modules in place and adds no directory.

## Constitution Check, after Phase 1

Re-run against the design rather than the intent. The design added a conformance check under `tools/`, which the
maintainer removed, so the design is now the sweep alone and adds nothing to the repository. No new violations, and
Complexity Tracking stays empty.

## Complexity Tracking

No constitution violations require justification.
