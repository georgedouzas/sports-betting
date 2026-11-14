# Proposal

## Why

The user asked to "make project compatible to the constitution". The constitution had moved on since the conventions
sweep of change 007, and the tree no longer met it. Measured against the tree at the start:

- Of the 189 public functions and classes, 111 carried no `Args`, `Returns` or `Raises` block, and 31 functions and
  classes carried no docstring at all.
- 110 names looked public but no package re-exported them, so the apparent surface was 192 names where the promised one
  was 82.
- Four `TYPE_CHECKING` import guards existed, and none prevented a cycle, since the package import graph is acyclic.
- 33 of 187 public names carried a runnable example.
- The package front page named three submodules while the package shipped more public surfaces, and the Project Profile
  named a `SelectionError` that did not exist.

Several of these counts were later corrected by measurement (see the design). The work is mostly invisible to a user:
docstrings, underscores on names no package re-exports, plain imports in place of guards. Those parts change nothing a
user of the API, the command line or the MCP server observes.

Some parts change the public surface that a user imports from, so this change carries spec deltas rather than
`skip_specs`:

- `ExecutionError`, `CancellationUnsupportedError`, `VenueBlockedError` and `CredentialError` moved from
  `sportsbet.execution` to `sportsbet.core`.
- `PlacementQuote` and `ExposureLimits` were removed from `sportsbet.execution` as dead code, and
  `PlacementReceiptSchema` became private.
- `Param`, `Schema` and `OutputsMapping` were removed from `sportsbet.core` as dead code, and `DATE_COLS` became
  private.

The constitution amendments made during this work are history. They are recorded here and in the design, not as specs:
version 9.0.0 (import privacy and the ban on type-checking guards), 10.0.0 and 10.0.1 (the licence header rule removed
and restored), 10.1.0 (command and tool docstrings carry the summary line alone), 10.1.1 (Project Profile
corrections), 10.3.0 (a package re-exports only what something outside it uses), and 12.0.0 (every named exception
lives in `sportsbet.core`).

## What Changes

- Every public function and class documents its parameters under `Args`, its return under `Returns`, and its
  exceptions under `Raises`. A class documents its initialization parameters under `Args` and learned state under
  `Attributes`. Every private name carries the summary line alone. Every module, class and function carries a
  docstring.
- `cli` and `mcp` command and tool functions carry the summary line alone, since Click renders a command docstring as
  `--help` and FastMCP sends a tool docstring to agents.
- Every name defined in a private module that its package does not re-export gets a leading underscore. The rename
  covered 66 distinct identifiers across 76 definitions.
- The four `TYPE_CHECKING` guards become plain imports. In `execution/_browser.py`, local `_Context`, `_Page`,
  `_Locator` and `_Response` protocols replace the guarded `playwright` types, so the optional `execution` extra is not
  a hard dependency of `import sportsbet.execution`.
- `tests/evaluation/__init__.py` imports `BaseBettor` from `sportsbet.evaluation`.
- 14 new runnable examples: `format_event_time`, `parse_event_time`, `derive_market_base`, `find_latest_odds_column`,
  `normalize_team_name`, `count_common_prefix`, `measure_names_similarity`, `pair_rosters`, `read_csv_content`,
  `build_roster`, `build_dataloader`, `normalize_identity`, `load_object`, `build_bettor`.
- The front page docstring of `sportsbet` names all seven public surfaces.
- The licence header lines are added to the five implementation modules that lacked them.
- **BREAKING**: `ExecutionError`, `CancellationUnsupportedError`, `VenueBlockedError` and `CredentialError` are
  imported from `sportsbet.core`, not `sportsbet.execution`.
- **BREAKING**: `PlacementQuote` and `ExposureLimits` are removed, and `PlacementReceiptSchema` is no longer exported
  from `sportsbet.execution`.
- **BREAKING**: `Param`, `Schema` and `OutputsMapping` are removed, and `DATE_COLS` is no longer exported from
  `sportsbet.core`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `execution/venue-contract`: the named execution exceptions are imported from `sportsbet.core`, the public names of
  `sportsbet.execution` are fixed, and the exposure limits are removed.
- `dataloaders/data-extraction`: the public names of `sportsbet.core`, which holds the shared data model, are fixed.

## Impact

- Code: every package under `src/sportsbet/` (`core`, `sources`, `dataloaders`, `evaluation`, `execution`, `cli`,
  `mcp`), 53 source modules and 45 test modules. The layout does not change.
- API: 88 re-exported names remain. `sportsbet.core` 26, `sportsbet.sources` 26, `sportsbet.execution` 15,
  `sportsbet.evaluation` 11, `sportsbet.dataloaders` 8, `sportsbet.mcp` 2, `sportsbet.cli` 1. The `sportsbet` command
  line keeps its 21 commands and the `sportsbet-mcp` server its 21 tools, with no name, option, parameter, order, exit
  code or returned shape changed.
- Docs: the execution guide shows the capabilities that need a venue or a credential as reference code. Every page
  block runs at build.
- Constitution: `openspec/constitution.md` amended as listed under Why.
- Changelog: generated from the commit messages by the `changelog` session.
- Dependencies: none.
- A conformance check under `tools/` was planned and then removed by the maintainer, with its tests, its `noxfile.py`
  wiring and its two scoped ignores in `pyproject.toml`. Enforcement after the sweep is review plus the existing gate.
- Specification quality checklist: every item passed. The reader of the spec is a contributor to the library rather
  than an end user, and the content quality items were checked against that reader. The second clarification changed
  the constitution rather than the specification.
