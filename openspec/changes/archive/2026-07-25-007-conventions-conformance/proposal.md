# Proposal

## Why

The input was: conform the whole codebase to Principle VI (`CONVENTIONS.md`), covering naming, docstrings, module
structure and test layout. Constitution v1.2.0 added Principle VI and made it a gate, so the code had to meet it. At the
green baseline (399 tests, the full gate green on 3.11, 3.12 and 3.13), 28 of 34 modules carried a module docstring
that narrated (`Implements the ...`, `It provides ...`, `Module that contains ...`), some public names were noun-first,
some private helpers carried Args and Returns blocks, and some tests imported private names from private modules.

Most of the work changes nothing a user observes. Three things are user-visible, so this change carries spec deltas
rather than `skip_specs`:

- The public function `complementary_events` in `sportsbet.evaluation` was renamed to `derive_complementary_events`.
- Two odds grammar helpers became public in `sportsbet.evaluation` under verb-first names: `derive_market_base` and
  `find_latest_odds_column`.
- The inert `max_unmatched_rate` parameter was removed from `DataLoader`, `build_dataloader`, the
  `--max-unmatched-rate` command line option and the MCP tool signatures.

The constitution change that defined Principle VI is history. It is recorded here and in the design, not as a spec.

## What Changes

- **BREAKING**: Rename `complementary_events` to `derive_complementary_events` in `sportsbet.evaluation`. Behaviour is
  unchanged. The fitted attribute `complementary_events_` and the class constant `COMPLEMENTARY_EVENTS` are part of the
  scikit-learn contract surface and keep their names.
- Export `derive_market_base` (formerly the private `market_base`) and `find_latest_odds_column` (formerly the private
  `latest_odds_column`) from `sportsbet.evaluation`.
- **BREAKING**: Remove the `max_unmatched_rate` parameter from the `DataLoader` constructor and `build_dataloader`,
  the `--max-unmatched-rate` command line option, and the MCP tool signatures. `resolve_odds` no longer consumed it.
- Reword every non-conforming module docstring to one imperative line, collapse private helper docstrings to one line,
  remove explanatory inline comments, and make each base module import no sibling. In `dataloaders`, `_schema` merged
  into `_base`, and the merged helpers were never public.
- Move tests to the paths that mirror the source, name them `test_<function>_<behavior>` with one-line docstrings, and
  import from the public API.
- Rewrite the reconciliation documentation (the `plot_reconciliation` example and the `sources.md` and `dataloader.md`
  guide sections) from the removed `ReconciliationReport`, `UnmatchedError` and `resolve` API to the current
  `resolve_odds(stats, odds, aliases)`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `evaluation/bettors`: `complementary_events` is renamed to `derive_complementary_events`, and `derive_market_base` and
  `find_latest_odds_column` become public.
- `sources/team-reconciliation`: the configurable unmatched rate tolerance is removed.

## Impact

- Code: every package under `src/sportsbet` (`sources` with `_params`, `dataloaders`, `evaluation`, `execution`, the
  `_selection` and `_artifacts` glue, `cli`, `mcp`) and the mirrored tests. 34 source modules across 6 packages plus 4
  top-level modules.
- API: `sportsbet.evaluation.__all__` drops `complementary_events` and gains `derive_complementary_events`,
  `derive_market_base` and `find_latest_odds_column`. `DataLoader` and `build_dataloader` lose `max_unmatched_rate`.
  The command line loses `--max-unmatched-rate`. The MCP tools lose the matching parameter.
- Callers updated for the renames: `_base`, `_rules`, `_model_selection`, `execution/_place.py`,
  `tests/evaluation/test_base.py`, `tests/dataloaders/test_basketball.py`,
  `docs/examples/modelling/plot_custom_bettor.py`, `docs/overview/user_guide/bettor.md`.
- Docs: `docs/examples` and the user guide follow every public rename. `docs/generated` is regenerated, never
  hand-edited.
- Changelog: `CHANGELOG.md` is generated from conventional commits (`pdm run changelog`), so the before and after of
  each rename lives in the package commit body.
- Dependencies: none added. The extras install what they installed before.
- Specification quality checklist: every item passed. The spec names `pdm run` gate commands and package names, which
  are domain vocabulary for this library rather than leaked implementation detail. The judgement a reviewer should
  weigh is which names count as the fixed scikit-learn contract surface.
