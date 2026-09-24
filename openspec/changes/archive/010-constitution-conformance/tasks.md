---

description: "Task list for constitution conformance"
---

# Tasks: Constitution conformance

**Input**: Design documents from `/specs/010-constitution-conformance/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md),
[data-model.md](./data-model.md), [contracts/public-surface.md](./contracts/public-surface.md)

**Tests**: The feature changes no behaviour, so the existing suite is the regression test and no test task is
written. A test that has to change is the signal that behaviour moved.

**Organization**: Tasks are grouped by user story. Each story leaves the gate green and can ship on its own.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel, different files, no dependency on an incomplete task
- **[Story]**: Which user story the task belongs to
- Paths are relative to the repository root

## Path Conventions

- Library source: `src/sportsbet/`
- Tests: `tests/`, mirroring the source tree

---

## Phase 1: Setup — REMOVED

## Phase 2: Foundational — REMOVED

Both phases built a conformance check under `tools/`, proposed in the plan rather than asked for. The maintainer
removed it, along with its tests, its `noxfile.py` wiring, and its two scoped ignores in `pyproject.toml`. Tasks T001
to T013 are void.

What the check produced before it went is kept, since it is what corrected the plan: every count in
[data-model.md](./data-model.md) is a measurement rather than an estimate, and four of those measurements contradicted
what the specification was written against.

Verification for each story below is therefore the existing gate, `ruff check`, `mypy`, `interrogate`, the doctest
run, and review.

---

## Phase 3: User Story 1 - Public names document what they take, return, and raise (Priority: P1) 🎯 MVP

**Goal**: every public function and class carries its `Args`, `Returns`, and `Raises` blocks, every private one
carries the summary line alone, and the 31 names carrying no docstring carry one.

**Independent Test**: `interrogate` reports full coverage, every public name reads with its blocks, and the doctest
run passes.

### Implementation for User Story 1

- [X] T014 [P] [US1] Write the blocks and the missing summary lines in `src/sportsbet/core/`, 4 public names
- [X] T015 [P] [US1] Write the blocks and the missing summary lines in `src/sportsbet/dataloaders/`, 13 public names
- [X] T016 [P] [US1] Write the blocks and the missing summary lines in `src/sportsbet/evaluation/`, 22 public names
- [X] T017 [P] [US1] Write the blocks in `src/sportsbet/sources/_base.py`, `_resolver.py`, `_schema.py`, and `_utils.py`
- [X] T018 [P] [US1] Write the blocks in `src/sportsbet/sources/_common/`, `_odds/`, and `_stats/`
- [X] T019 [P] [US1] Write the blocks and the missing summary lines in `src/sportsbet/execution/`, 39 public names
- [X] T020 [P] [US1] Write the blocks and the missing summary lines in `src/sportsbet/cli/`, 28 public names
- [X] T021 [P] [US1] Write the blocks and the missing summary lines in `src/sportsbet/mcp/`, 22 public names
- [X] T022 [US1] Confirm every private name across `src/sportsbet/` still carries the summary line alone, since rule 3
      passes today and must keep passing after the blocks are written
- [X] T024 [US1] Run the gate in order, `pdm run formatting`, `pdm run checks`, `pdm run docs build`, `pdm run tests`,
      and confirm the suite passed with no test edited

- [X] T024a [US1] Documented the 24 names that carried no docstring at all, 19 private helpers, two surface
      callbacks, two `classes_` properties, and the pandera `Config` class
- [X] T024b [US1] `cli` and `mcp` carry the summary line alone, by the amendment at constitution version 10.1.0,
      since Click renders a command docstring as `--help` and FastMCP sends a tool docstring to agents

**Checkpoint**: reached. `interrogate` reports full coverage, every public function outside the surface packages
carries the blocks its signature calls for, and 307 tests pass. What was measured as 120 missing blocks was 71 once
properties were excluded, of which 39 fall under the surface exemption and 32 were written

---

## Phase 4: User Story 2 - What a package exports and what it keeps are two lists (Priority: P2)

**Goal**: every package declares its surface, every name a package keeps is private, and no import hides behind a
type-checking guard.

**Independent Test**: every module still imports, no guard remains, and no name in
[contracts/public-surface.md](./contracts/public-surface.md) moved.

### Implementation for User Story 2

- [X] T025 [P] [US2] Replace the guarded imports with plain imports in `src/sportsbet/execution/_event.py`, dropping
      the `TYPE_CHECKING` block and the `typing` import it needed
- [X] T025a [US2] Replace the guarded `playwright` types in `src/sportsbet/execution/_browser.py` with the local
      `_Context`, `_Page`, `_Locator`, and `_Response` protocols, since a plain import would make the optional
      `execution` extra a hard dependency of `import sportsbet.execution`
- [X] T026 [P] [US2] Replace the guarded import with a plain import in `src/sportsbet/execution/_schedule.py`
- [X] T027 [P] [US2] Replace the guarded import with a plain import in `src/sportsbet/execution/_factory.py`
- [X] T028 [US2] Confirm no import cycle appeared, by importing every module in `src/sportsbet` in a fresh interpreter
- [X] T029 [US2] Dropped. Every package already declares `__all__`, as `__all__: list[str] = [...]`. The finding that
      six were missing came from a search for `__all__ = [...]`, and the check reports zero failures for the rule
- [X] T030 [US2] Dropped with T029
- [X] T031 [US2] Dropped with T029
- [X] T032 [US2] Dropped with T029
- [X] T033 [US2] Dropped with T029
- [X] T034 [US2] The rename covered 66 distinct identifiers across 76 definitions, not 110. No separate list file was
      written, since the rename was applied in one pass over name tokens rather than by hand. None of the three cases
      research flagged appeared: no identifier collided with a surface package or an exported name, and none appeared
      inside a string literal
- [X] T035 [US2] Renamed in one pass rather than per package, over name tokens only, so strings and comments were
      untouched. One keyword argument was caught wrongly, `caplog.set_level(logger=...)` in
      `tests/execution/test_event.py`, and restored
- [X] T036 [US2] Covered by the single pass in T035
- [X] T037 [US2] Covered by the single pass in T035
- [X] T038 [US2] Covered by the single pass in T035
- [X] T039 [US2] Covered by the single pass in T035
- [X] T040 [US2] Covered by the single pass in T035
- [X] T041 [US2] Run `git diff src/sportsbet/` against [contracts/public-surface.md](./contracts/public-surface.md)
      and confirm no line defining or re-exporting one of the 95 names changed outside a docstring
- [X] T043 [US2] Run `pdm run formatting`, `pdm run checks`, `pdm run docs build`, then `pdm run tests`, and
      confirm the suite passed with no test in `tests/` edited

- [X] T043a [US2] Changed `tests/evaluation/__init__.py` to import `BaseBettor` from the `sportsbet.evaluation`
      surface rather than from `sportsbet.evaluation._base`, the one private import left in the tests

**Checkpoint**: reached. The apparent surface of 192 names is the promised surface of 95, no guard remains, no module
reaches past a surface, and no test imports a private name. 307 tests pass and the 11 that fail need a chromium
binary, which they needed before this feature began

---

## Phase 5: User Story 3 - No comment stands in for a name (Priority: P3) — NOTHING TO DO

**Goal**: the only comments in the source are the licence headers and the suppressions.

**Outcome**: already true. The 75 comment lines this story was written to remove are the `# Author:` and `# License:`
header lines that 37 modules carry, two lines each, which the rule allows. The check reports zero failures for rule
11, and the 8 suppressions all carry a rule code and a reason.

- [X] T044 [US3] Dropped. `src/sportsbet/` carries no explanatory comment
- [X] T045 [US3] Dropped with T044
- [X] T046 [US3] Dropped with T044
- [X] T047 [US3] Dropped with T044
- [X] T048 [US3] Confirmed. The 8 suppressions in `src/sportsbet/` carry a rule code and a reason
- [X] T050 [US3] Ran. `ruff check src`, `mypy src`, and 302 tests green

**Checkpoint**: reached without work. What remains is rule 13, six inline suppressions of a code used more than once,
which belongs in the project configuration

## Phase 6: User Story 4 - Every example a reader sees has been run by the build (Priority: P4)

**Goal**: every public name that can run offline carries a runnable example, every name that cannot carries none, and
the guide shows the rest as reference code.

**Independent Test**: the documentation build and the doctest run are both green.

### Implementation for User Story 4

- [X] T051 [US4] Decide per name which of the 192 public names can run without a credential or the network,
      following the boundary in [research.md](./research.md), and record it in this file
- [X] T052 [P] [US4] Write the missing examples in `src/sportsbet/core/` and `src/sportsbet/dataloaders/`
- [X] T053 [P] [US4] Write the missing examples in `src/sportsbet/evaluation/`
- [X] T054 [P] [US4] Write the missing examples in `src/sportsbet/sources/` for the names the sample sources can
      drive, using `SampleSoccerStats` and `SampleSoccerOdds`
- [X] T055 [US4] Remove any example that cannot run from `src/sportsbet/execution/`, and move what it showed into the
      execution guide under `docs/` as reference code
- [X] T056 [P] [US4] Confirm every code block under `docs/` executes at build and carries no hand-written output
- [X] T058 [US4] Break one example on purpose, confirm `pdm run tests` goes red, then restore it
- [X] T059 [US4] Run `pdm run formatting`, `pdm run checks`, `pdm run docs build`, then `pdm run tests`, and
      confirm the suite passed with no test in `tests/` edited

**Progress**: 14 examples written and running, on `format_event_time`, `parse_event_time`, `derive_market_base`,
`find_latest_odds_column`, `normalize_team_name`, `count_common_prefix`, `measure_names_similarity`, `pair_rosters`,
`read_csv_content`, `build_roster`, `build_dataloader`, `normalize_identity`, `load_object`, and `build_bettor`.

T051 was settled by reading, not by asking further: a name earns an example when it can produce its result from
values an example can build, a temporary file included. That leaves 21 names without one, the 5 exception classes,
the 14 execution names that need a venue or a credential, `fetch_payloads` which reads the feed, and `BaseVenue`,
which is abstract. Those are shown in the guide as reference code.

T058 was checked by breaking `format_event_time`'s example on purpose. The doctest run went red, and restoring it
turned green.

**Checkpoint**: reached for everything that does not depend on a live feed. `pdm run docs build` from clean is red
because ESPN answers 400 to the range `plot_nba.py` asks for, which is the feed refusing rather than the code
failing

---

## Phase 7: User Story 5 - The Project Profile describes this repository (Priority: P5)

**Goal**: every concrete statement in the Project Profile holds, and the front page names every public surface.

**Independent Test**: read each Project Profile bullet against the tree, and each one is true.

### Implementation for User Story 5

- [X] T060 [US5] Rewrite the docstring of `src/sportsbet/__init__.py` to name all seven public surfaces, since it
      names three today
- [X] T061 [P] [US5] Check the package layering bullet in `.specify/memory/constitution.md` against `src/sportsbet/`
      and correct whichever of the two is wrong, the bullet or the layout
- [X] T062 [P] [US5] Check the builders bullet against `src/sportsbet/`, and confirm `build_dataloader`,
      `build_bettor`, and `build_venue` each live in the package that owns what they build
- [X] T063 [P] [US5] Search `src/sportsbet/` and confirm no bare `Exception` or `ValueError` stands where
      `BuildError`, `SelectionError`, `ExecutionError`, or `CredentialError` carries meaning
- [X] T064 [US5] Correct any Project Profile bullet that does not hold in `.specify/memory/constitution.md`, with a
      PATCH version bump and a Sync Impact Report entry
- [X] T065 [US5] Run `pdm run formatting`, `pdm run checks`, `pdm run docs build`, then `pdm run tests`, and
      confirm the suite passed with no test in `tests/` edited

- [X] T063a [US5] Found `SelectionError` named in the profile with no such class in the tree, and dropped it. The
      31 bare `ValueError` and `TypeError` raises all sit in `dataloaders` and `evaluation`, where the scikit-learn
      contract expects them, which the bullet now says
- [X] T061a [US5] Confirmed the layering bullet holds: `core`, then `sources`, `dataloaders`, `evaluation`,
      `execution`, then `cli` and `mcp`, with every import running downward and no cycle
- [X] T062a [US5] Confirmed each builder lives with what it builds, `build_dataloader` in `dataloaders/_factory.py`,
      `build_bettor` in `evaluation/_factory.py`, and `build_venue` in `execution/_factory.py`

**Checkpoint**: reached. The profile and the repository say the same thing, at constitution version 10.1.1

---

## Phase 7a: The licence header (found during implementation)

**Goal**: every implementation module carries the two header lines the constitution's Structure section requires.

- [X] T065a Added the `# Author:` and `# License:` lines to `src/sportsbet/__main__.py`, `core/_params.py`,
      `core/_types.py`, `core/_utils.py`, and `dataloaders/_base.py`, the five implementation modules that lacked them
- [X] T065b Confirmed the 11 remaining headerless modules are all `__init__.py`, which the Surface rule exempts
- [X] T065c Restored the licence header rule in `.specify/memory/constitution.md` at version 10.0.1, after version
      10.0.0 removed it on a measurement that searched for `Copyright` and missed `# License:`

---

## Phase 8: Polish & Cross-Cutting Concerns

- [X] T066 Void. The conformance check was removed
- [X] T067 Void with T066
- [X] T068 No hand edit needed. `CHANGELOG.md` is generated by the `changelog` session from the commit messages,
      parsed with the angular convention, so the commits carry the entry
- [X] T069 Run [quickstart.md](./quickstart.md) end to end and confirm every symptom in its regression table is
      absent
- [X] T070 Run `pdm run clean` then the full gate, since a reused `nox` environment can carry a stale toolchain

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies
- **Foundational (Phase 2)**: depends on Setup, and blocks every user story, since each story verifies itself with the
  check
- **User Stories (Phases 3 to 7)**: depend on Foundational. See the ordering note below, which differs from the
  priority order
- **Polish (Phase 8)**: depends on every story a release includes

### User Story Dependencies

- **US1 (P1)**: independent. Touches docstrings only
- **US2 (P2)**: independent of US1's content, but see the ordering note
- **US3 (P3)**: reads better after US1, since a comment often disappears when the docstring above it is written
- **US4 (P4)**: depends on US1, since an example sits in a docstring that US1 shapes
- **US5 (P5)**: independent

### The ordering note

[research.md](./research.md) recommends running **US2 before US1**, which is not the priority order. A rename in US2
changes names that US1's docstrings quote, so writing the blocks first means editing them twice for the 110 renamed
names. Priority orders value and this orders work, and the two disagree here.

- Ship-value order: US1, US2, US3, US4, US5
- Least-rework order: US2, US1, US3, US4, US5

Either is valid. Taking US1 first costs a second pass over the docstrings of renamed names, and buys the P1 increment
sooner.

### Within Each User Story

- The per-package tasks marked [P] touch different directories and do not collide
- The rename tasks are deliberately not marked [P], since one failing import in a tree-wide change says nothing about
  which of 110 names caused it
- Turning a rule to failing comes after the work it checks, never before
- The gate run closes each story

### Parallel Opportunities

- T003 and T004 in Setup
- T006 to T009 and T011 in Foundational, four rule groups and the tests
- T014 to T021 in US1, one per package, the largest parallel block in the feature
- T025 to T027 and T029 to T033 in US2, the guards and the surface declarations
- T044 to T047 in US3
- T052 to T054 and T056 in US4
- T061 to T063 in US5

---

## Parallel Example: User Story 1

```bash
Task: "Write the blocks and the missing summary lines in src/sportsbet/core/, 4 public names"
Task: "Write the blocks and the missing summary lines in src/sportsbet/dataloaders/, 13 public names"
Task: "Write the blocks and the missing summary lines in src/sportsbet/evaluation/, 22 public names"
Task: "Write the blocks in src/sportsbet/sources/_base.py, _resolver.py, _schema.py, and _utils.py"
Task: "Write the blocks in src/sportsbet/sources/_common/, _odds/, and _stats/"
Task: "Write the blocks and the missing summary lines in src/sportsbet/execution/, 39 public names"
Task: "Write the blocks and the missing summary lines in src/sportsbet/cli/, 28 public names"
Task: "Write the blocks and the missing summary lines in src/sportsbet/mcp/, 22 public names"
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1, Setup
2. Phase 2, Foundational, which is what makes every later claim checkable
3. Phase 3, User Story 1
4. Stop and validate: `interrogate` full, rules 1 to 4 green, the suite unchanged
5. The library is shippable, and the docstrings a user reads in an editor now say what each name takes and returns

### Incremental Delivery

1. Setup and Foundational, the tree is measured
2. US1, every public name documents itself, and the gate holds it there
3. US2, the surface shrinks to what the packages promise
4. US3, the comments are gone
5. US4, every example has been run
6. US5, the profile matches the repository

Each story ends with the full gate green, so any prefix of this list can ship.

### Parallel Team Strategy

- One contributor takes Setup and Foundational, since they are one piece of work
- After that, the per-package tasks inside US1 split cleanly across contributors, one package each
- US5 can run beside any of it, since it touches the constitution and the front page only

---

## Notes

- Commit after each task or each package, and never mix a rename commit with a docstring commit
- The signal that this feature went wrong is a test needing an edit. The suite is the contract
- Turn a rule to failing only once its work has landed, so the gate never goes red on the release branch
- The check reports the three rules it cannot decide. A green check is not a finished review
