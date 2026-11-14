# Design

## Context

The target is the constitution as it stood when the work was done. It started at version 8.0.3 and was amended during
the work, first to 9.0.0. This builds on the conventions sweep of change 007, which conformed the tree to the naming and
module rules as they stood then. This work is the delta the later amendments introduced.

Scope at planning: 53 source modules, 45 test modules, 192 public functions and classes, 95 re-exported names across 7
packages. The package import graph is acyclic:

```text
cli          -> core, dataloaders, evaluation, execution
dataloaders  -> core, sources
evaluation   -> core
execution    -> core, dataloaders, evaluation
mcp          -> dataloaders, evaluation, execution
sources      -> core
```

### Rule model

The rules were modelled as predicates over names, modules, packages and suppressions.

- A name is a function, class or module-level assignment. It is private when its identifier starts with one underscore
  and is not a dunder. It is exported when its package `__init__` imports it. It carries a docstring and a set of
  blocks (`Args`, `Returns`, `Raises`, `Attributes`, `Examples`), and it is offline when it can run without a credential
  or the network.
- A module is private when its file name starts with `_`, `__init__` and `__main__` aside. A module records its imports
  and whether each sits inside an `if TYPE_CHECKING:` block.
- A package is a directory with an `__init__.py`. A package and its subpackages count as one package for the import
  rule.
- A suppression is an inline exemption with a rule code and a reason. Two or more of the same code mean it belongs in
  the project configuration.

Rules and the places the tree failed them, measured by a check rather than estimated:

| # | Rule | Fails |
| --- | --- | ---: |
| 1 | Everything carries a docstring | 24 |
| 2 | A public name carries its blocks | 120 |
| 3 | A private name carries the line alone | 0 |
| 4 | An empty block is omitted | 8 |
| 5 | A name a package keeps is private | 76 |
| 6 | No reaching past a surface | 0 |
| 7 | A public name leaves through the surface | 0 |
| 8 | A package declares `__all__` | 0 |
| 9 | An `__init__` holds no logic | 0 |
| 10 | No type-checking import guard | 4 |
| 11 | Only licence and suppression comments | 0 |
| 12 | A suppression says which rule and why | 0 |
| 13 | A recurring suppression is configured | 6 |
| 14 | An offline public name carries an example | at least 27 |
| 15 | A name that needs the feed carries none | to be measured |
| 16 | An implementation module carries the header | 0 |

Four of these corrected an estimate: rule 8 was estimated at 6 because the search looked for `__all__ = [...]` while
every package writes `__all__: list[str] = [...]`. Rule 11 was estimated at 75 because the `# Author:` and `# License:`
header lines were counted as comments. Rule 5 was estimated at 110 before the surface packages were carved out and the
annotated constants counted. Rule 1 was estimated at 31 before constructors were exempted.

Three rules need a human: whether a summary line says what the thing does, whether a name is honest, and whether a name
can run offline.

### The public surface contract

No name re-exported by a package is renamed, removed, or given a different signature, return or exception, except the
changes the constitution amendments required during the work. The 88 re-exported names at the end:

- `sportsbet.core`, 26: `ALIASES`, `BoolData`, `BuildError`, `CancellationUnsupportedError`, `CredentialError`, `Data`,
  `EVENT_COLS`, `ExecutionError`, `FixturesData`, `GROUPS_COLS`, `IDENTITY_COLS`, `IDENTITY_FIELDS`, `Indices`,
  `MATCH_COLS`, `NON_PREPLAY_EVENT_STATUSES`, `NotExtractedError`, `PREPLAY_EVENT_STATUSES`, `ParamGrid`, `STATUSES`,
  `STATUS_RANK`, `TEAMS_COLS`, `TrainData`, `VenueBlockedError`, `format_event_time`, `load_object`,
  `parse_event_time`.
- `sportsbet.sources`, 26: `BaseOddsSchema`, `BaseOddsSource`, `BaseSource`, `BaseStatsSchema`, `BaseStatsSource`,
  `EuroLeagueStats`, `FootballDataOdds`, `FootballDataStats`, `NBAStats`, `OddsApi`, `RawItem`, `RawPayload`,
  `SampleSoccerOdds`, `SampleSoccerStats`, `build_roster`, `count_common_prefix`, `derive_market_outcomes`,
  `fetch_payloads`, `measure_names_similarity`, `normalize_identity`, `normalize_team_name`, `optional_col`,
  `pair_rosters`, `read_csv_content`, `required_col`, `resolve_odds`.
- `sportsbet.execution`, 15: `BaseVenue`, `BetIdentity`, `BrowserSession`, `CredentialRef`, `FixedSession`,
  `PageSnapshot`, `PlacementIntent`, `PlacementReceipt`, `PlacementStatus`, `Placer`, `build_receipts_frame`,
  `build_venue`, `execute_event`, `find_betting_moment`, `resolve`.
- `sportsbet.evaluation`, 11: `BaseBettor`, `BettorGridSearchCV`, `ClassifierBettor`, `OddsComparisonBettor`,
  `backtest`, `build_bettor`, `derive_complementary_events`, `derive_market_base`, `find_latest_odds_column`,
  `load_bettor`, `save_bettor`.
- `sportsbet.dataloaders`, 8: `BaseDataLoader`, `DEFAULT_KEY_ENV`, `DataLoader`, `ODDS_SOURCES`, `STATS_SOURCES`,
  `build_dataloader`, `build_extraction_settings`, `load_dataloader`.
- `sportsbet.mcp`, 2: `run`, `server`.
- `sportsbet.cli`, 1: `main`.

The `sportsbet` command line keeps 21 commands and the `sportsbet-mcp` server keeps 21 tools. No command name, option
name, argument order or exit code changes, and no tool name, parameter name or returned shape changes. The parity test
is the contract for the surfaces.

The contract is checked by the existing test suite passing unchanged, the parity test, every module importing from its
documented surface, and a `git diff` that touches no line defining or re-exporting a listed name outside a docstring.

## Goals / Non-Goals

**Goals:**

- 100% of public functions and classes document their parameters, return and exceptions, up from 41% (SC-001).
- 100% of functions and classes carry a docstring, up from 91% (SC-002).
- 100% of private functions and classes carry the summary line alone, which held and still holds (SC-003).
- Zero names in a private module are public without their package re-exporting them (SC-004).
- Zero modules import another package's private name, and zero type-checking guards remain (SC-011).
- Zero comment lines beyond licence headers and suppressions (SC-005).
- Every example a reader sees has been executed by the build (SC-006).
- 100% of public names that can run offline carry a runnable example, up from 18%, and none that cannot run offline
  carries one (SC-010).
- Every statement in the Project Profile holds against the repository (SC-007).
- The full gate passes at the end of every user story, and the existing suite confirms unchanged behaviour (SC-008).
- A reviewer can check one module against the rules in under five minutes (SC-009).

**Non-Goals:**

- Changing what any public name does, returns or accepts.
- The constitution's Control Flow section, whose rules cannot be honoured without changing what code does.
- Changing the rules to fit the code. A wrong rule is amended separately.
- A new enforcement tool. The planned check under `tools/` was removed by the maintainer.

## Decisions

- Clarified: Which names must carry a runnable doctest example? -> Every public function, class and method, not only
  the names a package re-exports. A name that touches the network or needs a credential is the exception and is shown
  as reference code in the guide.
- Clarified: How should a private module obtain a name that exists only as a public name in another package? -> It
  imports it from that package's surface. The constitution was amended to say so in version 9.0.0. The rule a module
  breaks is importing another package's private name, and the rule a name breaks is staying public when its package
  never re-exports it. A package and its subpackages count as one package.
- Clarified: Are the `TYPE_CHECKING` import guards kept? -> No. None prevents a cycle, since the import graph is
  acyclic. They become plain imports, and version 9.0.0 forbids the guard.

### R1. Enforcement after the sweep (withdrawn)

The plan proposed a conformance check under `tools/`, run from the `checks` session. The maintainer did not ask for a
new tool and removed it. The rules are enforced by review and the existing gate: `ruff` with the `D` rules for
docstring shape, `interrogate` for coverage, `mypy`, and the doctest run for examples. The measurements the check
produced before removal stand.

### R2. Which public names carry a runnable example

A name carries a runnable example when it can run offline with no credential. That covers `core`, `dataloaders` and
`evaluation`, plus the part of `sources` that works against `SampleSoccerStats` and `SampleSoccerOdds`. A name that
reads a live feed or needs a key is shown as reference code in the guide. `cli` and `mcp` names are command callbacks
and tool functions, documented by the guide as a user types them.

| Package | Public names | With an example before | Can run offline |
| --- | ---: | ---: | --- |
| `core` | 4 | 0 | yes |
| `dataloaders` | 13 | 4 | yes |
| `evaluation` | 22 | 8 | yes |
| `sources` | 64 | 18 | partly, through the sample sources |
| `execution` | 39 | 3 | no, every path needs a venue and a credential |
| `cli` | 28 | 0 | as commands, not as calls |
| `mcp` | 22 | 0 | as tools, not as calls |

In the end, a name earned an example when it can produce its result from values an example can build, a temporary
file included. 21 names have none: the 5 exception classes, the 14 execution names that need a venue or a credential,
`fetch_payloads` which reads the feed, and the abstract `BaseVenue`.

Alternatives considered: an example on all 192 names with a skip directive, rejected because the constitution forbids
an example that does not run. Recorded payloads for live feeds, rejected because the fixtures would drift from the feed.

### R3. The order of the passes

Guards and `__all__` first, then the conformance check, the rename, the docstring blocks, the comments, the examples,
and the Project Profile. A rename before the docstrings avoids writing blocks against names that then change. Comments
after docstrings, since a comment usually disappears when the docstring above it is written.

Alternative considered: strict priority order P1 to P5, rejected because it writes 111 docstring blocks against names
the rename then changes.

### R4. Renaming the internal names

A script rewrote definitions and references together over name tokens only, so strings and comments were untouched.
The plan called for one package at a time. The work ran one pass, which covered 66 distinct identifiers across 76
definitions. One keyword argument was caught wrongly, `caplog.set_level(logger=...)` in
`tests/execution/test_event.py`, and restored. None of the cases research flagged for a human appeared.

Alternatives considered: renaming by hand, rejected at that size, and renaming all at once without the gate between
packages.

### R5. The type-checking guards were not load-bearing

`dataloaders` and `evaluation` never import `execution`, so the guards in `execution/_event.py` and
`execution/_schedule.py` guarded nothing. `execution/_browser.py` does not import `execution/_factory.py`. Each guard
became a plain import, except the guarded `playwright` types in `_browser.py`, which became local protocols so the
optional `execution` extra stays optional.

Alternative considered: keeping the guards for import cost, rejected since the modules are imported at runtime
elsewhere in the same package.

### R6. The surface declarations, corrected

Research first claimed that only `core` declared `__all__`. All seven packages declare it as
`__all__: list[str] = [...]`. The 75 comment lines User Story 3 targeted were the `# Author:` and `# License:` headers
on 37 modules, so the story had nothing to remove. The claim that no module carried a licence header missed
`# License: MIT`. 37 modules carry it, 11 of the rest are package surfaces the rule exempts, and the 5 implementation
modules that lacked it received it. A count from a search for a guessed spelling is not a measurement.

### Constitution amendments during the work

- 9.0.0: a private module imports another package's public name from that package's surface, and the type-checking
  guard is forbidden.
- 10.0.0 removed the licence header rule on a wrong measurement, and 10.0.1 restored it.
- 10.1.0: `cli` and `mcp` command and tool functions carry the summary line alone.
- 10.1.1: Project Profile corrections. `SelectionError` was dropped since no such class exists. The 31 bare
  `ValueError` and `TypeError` raises sit in `dataloaders` and `evaluation`, where the scikit-learn contract expects
  them. The layering is `core`, then `sources`, `dataloaders`, `evaluation`, `execution`, then `cli` and `mcp`. The
  builders live with what they build: `build_dataloader` in `dataloaders/_factory.py`, `build_bettor` in
  `evaluation/_factory.py`, `build_venue` in `execution/_factory.py`.
- 10.3.0: a package re-exports only the names something outside it uses, and a name nothing uses is removed rather
  than made private. `Param`, `Schema`, `OutputsMapping`, `PlacementQuote` and `ExposureLimits` were removed.
  `DATE_COLS` and `PlacementReceiptSchema` became private. `Placer` stays, since it types the `placer` parameter of
  `execute_event`.
- 12.0.0: every named exception lives in `sportsbet.core`.

### Assumptions recorded in the spec

- A constructor documents its parameters under the class `Args`. A property carries the one-line summary.
- A method that overrides a documented base carries its own docstring.
- The scikit-learn estimator contract wins over the naming rules, so fixed method and parameter names are documented
  rather than renamed.
- Third-party imports are outside the privacy rule.
- Adding an underscore to a name no package re-exports is not a public API change.

### Requirement coverage

FR-006, FR-007, FR-009, FR-019 and FR-021 produce the surface changes in the `execution/venue-contract` and
`dataloaders/data-extraction` deltas. FR-001 to FR-005, FR-008, FR-010 to FR-018, FR-020 and FR-022 govern docstrings,
imports, comments, suppressions, examples, the Project Profile and the gate. They change no behaviour a user observes
and are recorded in this design.

## Risks / Trade-offs

- A test needing an edit signals that a public name moved. Only `tests/evaluation/__init__.py` changed, to import
  `BaseBettor` from the public surface.
- The removal of the dead names and the move of the exceptions to `sportsbet.core` break imports from the old place.
- After the sweep, review holds the rules in place, since the conformance check was removed.
- At the end, `pdm run docs build` from clean was red because ESPN answered 400 to the range `plot_nba.py` asks for.
  That is the feed refusing, not the code failing. 11 tests need a chromium binary, as they did before.

## Migration Plan

Each user story ends with the gate green: `pdm run formatting`, `pdm run checks`, `pdm run docs build`,
`pdm run tests`, read from the session summary. Run with no credential set, since a set secret can hide an example that
needs one, and run `pdm run clean` first, since a reused `nox` environment can carry a stale toolchain.

A user migrates by importing `ExecutionError`, `CancellationUnsupportedError`, `VenueBlockedError` and
`CredentialError` from `sportsbet.core`, and by removing uses of `PlacementQuote`, `ExposureLimits`,
`PlacementReceiptSchema`, `Param`, `Schema`, `OutputsMapping` and `DATE_COLS`.

What a regression looks like:

- A test needed an edit: a public name moved. Check it against the surface contract.
- The documentation build went red: an example stopped running, or a page referenced a renamed name.
- The doctest run went red but the suite passed: an example is wrong.
- `interrogate` fell: a docstring was removed rather than written.
- Coverage changed: something other than a docstring moved, since this work adds no logic.
