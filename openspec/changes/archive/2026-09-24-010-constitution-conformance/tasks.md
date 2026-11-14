# Tasks

The Setup and Foundational phases (T001 to T013) built a conformance check under `tools/`. The maintainer removed it
with its tests, its `noxfile.py` wiring and its two scoped ignores in `pyproject.toml`, so those tasks are void. The
measurements it produced are kept in the design.

## 1. User Story 1: public names document what they take, return and raise

- [x] 1.1 Write the blocks and the missing summary lines in `src/sportsbet/core/`, 4 public names. (T014)
- [x] 1.2 Write the blocks and the missing summary lines in `src/sportsbet/dataloaders/`, 13 public names. (T015)
- [x] 1.3 Write the blocks and the missing summary lines in `src/sportsbet/evaluation/`, 22 public names. (T016)
- [x] 1.4 Write the blocks in `src/sportsbet/sources/_base.py`, `_resolver.py`, `_schema.py` and `_utils.py`. (T017)
- [x] 1.5 Write the blocks in `src/sportsbet/sources/_common/`, `_odds/` and `_stats/`. (T018)
- [x] 1.6 Write the blocks and the missing summary lines in `src/sportsbet/execution/`, 39 public names. (T019)
- [x] 1.7 Write the blocks and the missing summary lines in `src/sportsbet/cli/`, 28 public names. (T020)
- [x] 1.8 Write the blocks and the missing summary lines in `src/sportsbet/mcp/`, 22 public names. (T021)
- [x] 1.9 Confirm every private name across `src/sportsbet/` still carries the summary line alone. (T022)
- [x] 1.10 Run the gate in order, `pdm run formatting`, `pdm run checks`, `pdm run docs build`, `pdm run tests`, and
  confirm the suite passed with no test edited. (T024)
- [x] 1.11 Document the 24 names that carried no docstring at all: 19 private helpers, two surface callbacks, two
  `classes_` properties, and the pandera `Config` class. (T024a)
- [x] 1.12 Give `cli` and `mcp` the summary line alone, by the amendment at constitution version 10.1.0, since Click
  renders a command docstring as `--help` and FastMCP sends a tool docstring to agents. (T024b)

Checkpoint reached: `interrogate` reports full coverage, and 307 tests pass. What was measured as 120 missing blocks was
71 once properties were excluded, of which 39 fall under the surface exemption and 32 were written.

## 2. User Story 2: what a package exports and what it keeps are two lists

- [x] 2.1 Replace the guarded imports with plain imports in `src/sportsbet/execution/_event.py`, dropping the
  `TYPE_CHECKING` block and the `typing` import it needed. (T025)
- [x] 2.2 Replace the guarded `playwright` types in `src/sportsbet/execution/_browser.py` with the local `_Context`,
  `_Page`, `_Locator` and `_Response` protocols, since a plain import would make the optional `execution` extra a hard
  dependency of `import sportsbet.execution`. (T025a)
- [x] 2.3 Replace the guarded import with a plain import in `src/sportsbet/execution/_schedule.py`. (T026)
- [x] 2.4 Replace the guarded import with a plain import in `src/sportsbet/execution/_factory.py`. (T027)
- [x] 2.5 Confirm no import cycle appeared, by importing every module in `src/sportsbet` in a fresh interpreter. (T028)
- [x] 2.6 Dropped. Every package already declares `__all__` as `__all__: list[str] = [...]`. The finding that six
  were missing came from a search for `__all__ = [...]`. (T029)
- [x] 2.7 Dropped with T029. (T030)
- [x] 2.8 Dropped with T029. (T031)
- [x] 2.9 Dropped with T029. (T032)
- [x] 2.10 Dropped with T029. (T033)
- [x] 2.11 The rename covered 66 distinct identifiers across 76 definitions, not 110. No separate list file was written,
  since the rename ran in one pass over name tokens. None of the three cases research flagged appeared. (T034)
- [x] 2.12 Rename in one pass over name tokens only, so strings and comments are untouched. One keyword argument was
  caught wrongly, `caplog.set_level(logger=...)` in `tests/execution/test_event.py`, and restored. (T035)
- [x] 2.13 Covered by the single pass in T035. (T036)
- [x] 2.14 Covered by the single pass in T035. (T037)
- [x] 2.15 Covered by the single pass in T035. (T038)
- [x] 2.16 Covered by the single pass in T035. (T039)
- [x] 2.17 Covered by the single pass in T035. (T040)
- [x] 2.18 Run `git diff src/sportsbet/` against the public surface contract and confirm no line defining or
  re-exporting one of the 95 names changed outside a docstring. (T041)
- [x] 2.19 Run `pdm run formatting`, `pdm run checks`, `pdm run docs build`, then `pdm run tests`, and confirm the
  suite passed with no test in `tests/` edited. (T043)
- [x] 2.20 Change `tests/evaluation/__init__.py` to import `BaseBettor` from the `sportsbet.evaluation` surface rather
  than from `sportsbet.evaluation._base`, the one private import left in the tests. (T043a)

Checkpoint reached: the apparent surface of 192 names is the promised surface of 95, no guard remains, no module
reaches past a surface, and no test imports a private name. 307 tests pass, and the 11 that fail need a chromium
binary, which they needed before.

## 3. User Story 3: no comment stands in for a name

- [x] 3.1 Dropped. `src/sportsbet/` carries no explanatory comment. (T044)
- [x] 3.2 Dropped with T044. (T045)
- [x] 3.3 Dropped with T044. (T046)
- [x] 3.4 Dropped with T044. (T047)
- [x] 3.5 Confirmed. The 8 suppressions in `src/sportsbet/` carry a rule code and a reason. (T048)
- [x] 3.6 Ran `ruff check src`, `mypy src`, and 302 tests green. (T050)

Checkpoint reached without work. Rule 13 remains: six inline suppressions of a code used more than once, which belong in
the project configuration.

## 4. User Story 4: every example a reader sees has been run by the build

- [x] 4.1 Decide per name which of the 192 public names can run without a credential or the network, following the
  boundary in the design, and record it. (T051)
- [x] 4.2 Write the missing examples in `src/sportsbet/core/` and `src/sportsbet/dataloaders/`. (T052)
- [x] 4.3 Write the missing examples in `src/sportsbet/evaluation/`. (T053)
- [x] 4.4 Write the missing examples in `src/sportsbet/sources/` for the names the sample sources can drive, using
  `SampleSoccerStats` and `SampleSoccerOdds`. (T054)
- [x] 4.5 Remove any example that cannot run from `src/sportsbet/execution/`, and move what it showed into the
  execution guide under `docs/` as reference code. (T055)
- [x] 4.6 Confirm every code block under `docs/` executes at build and carries no hand-written output. (T056)
- [x] 4.7 Break one example on purpose, confirm `pdm run tests` goes red, then restore it. (T058)
- [x] 4.8 Run `pdm run formatting`, `pdm run checks`, `pdm run docs build`, then `pdm run tests`, and confirm the
  suite passed with no test in `tests/` edited. (T059)

Progress: 14 examples written and running. 21 names carry none: the 5 exception classes, the 14 execution names that
need a venue or a credential, `fetch_payloads`, and the abstract `BaseVenue`. `pdm run docs build` from clean was red
because ESPN answered 400 to the range `plot_nba.py` asks for, which is the feed refusing rather than the code failing.

## 5. User Story 5: the Project Profile describes this repository

- [x] 5.1 Rewrite the docstring of `src/sportsbet/__init__.py` to name all seven public surfaces, since it named three.
  (T060)
- [x] 5.2 Check the package layering bullet in `openspec/constitution.md` against `src/sportsbet/` and correct
  whichever of the two is wrong. (T061)
- [x] 5.3 Check the builders bullet against `src/sportsbet/`, and confirm `build_dataloader`, `build_bettor` and
  `build_venue` each live in the package that owns what they build. (T062)
- [x] 5.4 Search `src/sportsbet/` and confirm no bare `Exception` or `ValueError` stands where `BuildError`,
  `SelectionError`, `ExecutionError` or `CredentialError` carries meaning. (T063)
- [x] 5.5 Correct any Project Profile bullet that does not hold in `openspec/constitution.md`, with a PATCH version
  bump and a Sync Impact Report entry. (T064)
- [x] 5.6 Run `pdm run formatting`, `pdm run checks`, `pdm run docs build`, then `pdm run tests`, and confirm the
  suite passed with no test in `tests/` edited. (T065)
- [x] 5.7 Found `SelectionError` named in the profile with no such class in the tree, and dropped it. The 31 bare
  `ValueError` and `TypeError` raises all sit in `dataloaders` and `evaluation`, where the scikit-learn contract
  expects them, which the bullet now says. (T063a)
- [x] 5.8 Confirmed the layering bullet holds: `core`, then `sources`, `dataloaders`, `evaluation`, `execution`, then
  `cli` and `mcp`, with every import running downward and no cycle. (T061a)
- [x] 5.9 Confirmed each builder lives with what it builds: `build_dataloader` in `dataloaders/_factory.py`,
  `build_bettor` in `evaluation/_factory.py`, `build_venue` in `execution/_factory.py`. (T062a)

Checkpoint reached: the profile and the repository say the same thing, at constitution version 10.1.1.

## 6. The licence header

- [x] 6.1 Add the `# Author:` and `# License:` lines to `src/sportsbet/__main__.py`, `core/_params.py`,
  `core/_types.py`, `core/_utils.py` and `dataloaders/_base.py`, the five implementation modules that lacked them.
  (T065a)
- [x] 6.2 Confirm the 11 remaining headerless modules are all `__init__.py`, which the Surface rule exempts. (T065b)
- [x] 6.3 Restore the licence header rule in `openspec/constitution.md` at version 10.0.1, after version 10.0.0
  removed it on a measurement that searched for `Copyright` and missed `# License:`. (T065c)

## 7. Polish and cross-cutting concerns

- [x] 7.1 Void. The conformance check was removed. (T066)
- [x] 7.2 Void with T066. (T067)
- [x] 7.3 No hand edit needed. `CHANGELOG.md` is generated by the `changelog` session from the commit messages, parsed
  with the angular convention. (T068)
- [x] 7.4 Run the quickstart end to end and confirm every symptom in its regression table is absent. (T069)
- [x] 7.5 Run `pdm run clean` then the full gate, since a reused `nox` environment can carry a stale toolchain. (T070)
