# Rename ledger: Conventions conformance

**Feature**: [spec.md](../spec.md)

Every public-symbol rename, recorded as it is made, so the changelog and the callers stay in step (FR-003,
FR-016, SC-007). A rename changes the identifier only; behaviour is held constant. Filled per package as the
sweep proceeds; empty rows are packages not yet done.

| Package | Old name | New name | Callers / exports / tests / docs updated | In CHANGELOG |
| --- | --- | --- | --- | --- |
| sources | _none_ (docstrings and test names only) | | | n/a |
| dataloaders | _none_ (`_schema` merged into `_base`; the merged helpers were never public) | | | n/a |
| evaluation | `complementary_events` | `derive_complementary_events` | `evaluation.__all__`, `_base` def/doctest/internal call, `tests/evaluation/test_base.py`, `tests/dataloaders/test_basketball.py`, `docs/examples/modelling/plot_custom_bettor.py`, `docs/overview/user_guide/bettor.md` | commit body |
| evaluation | `market_base` (private) | `derive_market_base` (now exported) | `evaluation.__all__`, `_base`, `_rules`, `tests/evaluation/test_base.py` | commit body |
| evaluation | `latest_odds_column` (private) | `find_latest_odds_column` (now exported) | `evaluation.__all__`, `_base`, `_rules`, `execution/_place.py`, `tests/evaluation/test_base.py` | commit body |
| execution | _none_ (the __init__ module docstring only) | | | n/a |
| selection & artifacts | _none_ (module docstrings and plain-English wording only) | | | n/a |
| cli | _none_ (the command functions are the CLI's command names, the contract) | | | n/a |
| mcp | _none_ (the tool functions are the MCP tool names, the contract) | | | n/a |

## Follow-up cleanup made during the sweep

Two removed-feature remnants of the earlier resolver refactor were cleaned up so the code, the docs and the
behaviour agree. These are removals, not renames, and are recorded here for completeness:

- The reconciliation documentation (the `plot_reconciliation` example and the `sources.md` / `dataloader.md`
  guide sections) was rewritten from the removed `ReconciliationReport` / `UnmatchedError` / `resolve` API to
  the current `resolve_odds(stats, odds, aliases)`, which pairs by fuzzy name and drops an unmatched match
  silently.
- The dead `max_unmatched_rate` parameter was removed from the `DataLoader` constructor, `build_dataloader`,
  the `--max-unmatched-rate` CLI option, the MCP tool signatures, the parity test, and the `DataLoader`
  docstring (which also documented a `reconciliation_` attribute that was never set). It was inert:
  `resolve_odds` no longer consumed it. This is a public-API change, recorded in its commit body for the
  generated changelog.

The fitted attribute `complementary_events_` and the class constant `COMPLEMENTARY_EVENTS` are the
scikit-learn contract surface (research D1) and are NOT renamed, even though the function that fills the
attribute was. `CHANGELOG.md` is generated from the conventional commits (`pdm run changelog`), so the
before/after lives in the package commit body rather than being hand-edited into the generated file.

## Rules

- A name the scikit-learn contract fixes (research D1) never appears here: it is not renamed.
- A rename lands in the same commit as its package, with every caller, the `__init__` export and `__all__`,
  the tests, the docstrings and doctests, and `docs/` updated, and a `CHANGELOG.md` line with before and after.
- If a rename would collide with an existing public name, it is reconsidered rather than forced, and the
  collision is noted in the package's task.
