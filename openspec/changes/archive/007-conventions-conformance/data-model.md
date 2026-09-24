# Phase 1 Violation Inventory: Conventions conformance

**Feature**: [spec.md](./spec.md) | **Date**: 2026-07-20

The "entities" of a conventions refactor are the violations. This is the inventory that the per-package
tasks work from. It is the state at the green baseline; each package task re-scans its own files, since a
sweep of a lower package may change what a higher one imports.

## Module docstrings (FR-005)

28 of 34 modules carry a non-conforming module docstring. Grouped by package, in sweep order.

| Package | Module | Current docstring opening |
| --- | --- | --- |
| sources | `_utils.py` | `Includes utilities shared by the sports.` |
| sources | `_odds/_football_data.py`, `_stats/_football_data.py` | `Implements the sources backed by the football-data.co.uk feed.` |
| sources | `_odds/_odds_api.py` | `Implements the odds source backed by The Odds API.` |
| sources | `_odds/_sample.py`, `_stats/_sample.py` | `Implements the sources of the sample data …` |
| sources | `_stats/_euroleague.py` | `Implements the statistics source backed by the EuroLeague's official API.` |
| sources | `_stats/_nba.py` | `Implements the statistics source of the NBA, backed by ESPN.` |
| sources | `__init__.py` | `It provides the sources the data comes from.` |
| dataloaders | `_base.py` | `Implements the base dataloader class shared by all dataloaders.` |
| dataloaders | `_sourced.py` | `Implements the dataloader shared by every sport whose data comes from sources.` |
| dataloaders | `__init__.py` | `It provides the dataloaders that shape the data for modelling.` |
| evaluation | `_base.py`, `_model_selection.py` | `Includes base class and functions for evaluating betting strategies.` |
| evaluation | `_classifier.py` | `Create a bettor based on a classifier.` (verb-first, tighten only) |
| evaluation | `_rules.py` | `Create a bettor based on betting rules.` (verb-first, tighten only) |
| evaluation | `__init__.py` | `It provides the tools to evaluate the performance of predictive models.` |
| execution | `__init__.py` | `It provides the placing of the bets a bettor found.` |
| selection & artifacts | `_artifacts.py` | `Implements the file a surface writes so that the data is downloaded once and reused.` |
| selection & artifacts | `_selection.py` | `Implements the selection, which is what a surface is told instead of being handed a file.` |
| cli | `_cli.py`, `_data.py`, `_betting.py`, `_execution.py`, `_options.py`, `_utils.py` | `Module that contains the … of the CLI.` |
| mcp | `_server.py` | `Implements the server that lets an agent drive the library.` |

Already conforming (leave): `execution/_base.py`, `_browser.py`, `_credentials.py`, `_place.py`,
`_schedule.py`; `sources/_base.py`, `_resolver.py`, `_schema.py`; `dataloaders/_schema.py`.

## Other violation categories (re-scanned per package task)

| Category | Rule | How to find it |
| --- | --- | --- |
| Noun-first / empty-verb function name | FR-001 | Read each `def` in the package; a name not starting with a verb, or starting with `process`/`handle`/`transform`-without-object, is a candidate. Exclude the D1 contract surface. |
| Private helper with Args/Returns block | FR-006 | A `def _name` whose docstring has an `Args:`/`Returns:` section, or a body paragraph, that is not a public entry point. Collapse to one line. |
| Base module importing a sibling | FR-009 | `from ._x import` in a `_base.py`. Merge the definition into the base. |
| In-function import not deferring an extra | FR-010 | `import` inside a `def` body without a `# noqa: PLC0415` optional-extra reason. |
| Explanatory inline comment | FR-011 | A `#` comment inside a function body that explains the next line. Remove it or fix the names. |
| Test not at the mirrored path | FR-012 | A source module `X` with no `test_X` at the mirrored test path, or a `test_` file mirroring no module. |
| Test name not `test_<function>_<behavior>` | FR-013 | A `def test_` whose name does not begin with the function under test. |
| Multi-line test docstring | FR-013 | A `def test_` whose docstring spans more than one line. |
| Test importing a private name from a private module | FR-014 | `from sportsbet.<pkg>._<mod> import` in a test. Import from the public API, or export the helper. |

## Rename candidates known at baseline

These are the renames visible now; each package task confirms and extends its own list, recording them in
[contracts/rename-ledger.md](./contracts/rename-ledger.md).

| Package | Old name | Note |
| --- | --- | --- |
| sources | `derive_market_outcomes` | already verb-first (kept); listed because it was the last rename |
| evaluation | `complementary_events` (public) | noun-first; rename to a verb phrase if it is a function, or leave if it is data |
| evaluation | `latest_odds_column`, `market_base` (private, tested) | export or test through the public API (FR-014), and check verb-first |

The bulk of renames are discovered when each package is read in its task; the ledger is the running record.
