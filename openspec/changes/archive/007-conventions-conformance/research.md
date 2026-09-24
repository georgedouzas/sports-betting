# Phase 0 Research: Conventions conformance

**Feature**: [spec.md](./spec.md) | **Date**: 2026-07-20

Three decisions, all settled by reading the code rather than by preference.

## D1: The fixed scikit-learn contract surface (what NOT to rename)

**Decision**: These names are fixed by Principle I and are not renamed, even where Principle VI's verb-first
rule would otherwise apply:

- Estimator methods: `fit`, `predict`, `predict_proba`, `bet`, `score`, `get_params`, `set_params`.
- Fitted attributes: every trailing-underscore attribute (`betting_markets_`, `target_event_status_`,
  `feature_names_out_`, `reconciliation_`, …).
- Constructor parameter names (`param_grid`, `stats`, `odds`, `alpha`, `stake`, …).
- Public estimator classes, which are nouns because a class is a noun: `BaseBettor`, `ClassifierBettor`,
  `OddsComparisonBettor`, `BettorGridSearchCV`, `BaseDataLoader`, `DataLoader`, and the source/venue base
  classes `BaseSource`, `BaseStatsSource`, `BaseOddsSource`, `BaseVenue`.

**Rationale**: Principle I makes interoperability with the scikit-learn ecosystem the library's core value.
Renaming `fit` to `fit_model` to satisfy a verb-object rule would break every pipeline and grid search
downstream. The public API surfaces confirm the set: `evaluation.__all__` is `['BaseBettor',
'BettorGridSearchCV', 'ClassifierBettor', 'OddsComparisonBettor', 'backtest', 'complementary_events',
'load_bettor', 'save_bettor']` and `dataloaders.__all__` is `['BaseDataLoader', 'DataLoader',
'load_dataloader']`. Of the non-class names, `backtest`, `load_bettor`, `save_bettor`, `load_dataloader`
already begin with a verb. `complementary_events` is a noun-first public function and is the one candidate
to weigh in the evaluation package (rename to `list_complementary_events` or similar, if it is a function).

**Alternatives considered**: renaming the estimator classes to verb-y factory functions was rejected as a
behaviour and API change, out of scope by FR-016.

## D2: A module docstring is reworded, not re-scoped

**Decision**: A non-conforming module docstring is rewritten to one imperative line that says what the
module does, which is the same thing the old sentence said. The module's content does not change.

- `"""Implements the base classes of the data sources."""` → an imperative of what it does.
- `"""It provides the sources the data comes from."""` (a package `__init__`) → an imperative line. A
  package `__init__` still gets a one-liner; `It provides ...` is meta narration and is reworded.
- `"""Module that contains the evaluation commands of the CLI."""` → `"""Run the evaluation commands from
  the command line."""` or similar.
- `"""Create a bettor based on a classifier."""` already begins with a verb and is close; it is kept or
  tightened, not rewritten.

**Rationale**: FR-005 forbids `Implements`, `It provides`, `Module that contains` as meta narration. The
fix is a wording change, never a scope change, which keeps the work behaviour-preserving.

**Inventory**: 28 of 34 modules carry a non-conforming module docstring today (see data-model). The six
already-conforming ones are the recently refactored `execution/*`, `sources/_base.py`, `sources/_resolver.py`,
and `sources/_schema.py` / `dataloaders/_schema.py`.

## D3: Bottom-up package order

**Decision**: Sweep the packages in dependency order, foundation first:
`sources` (with `_params`) → `dataloaders` → `evaluation` → `execution` → selection & artifacts
(`_selection`, `_artifacts`, top `__init__`) → `cli` → `mcp`.

**Rationale**: A rename in a lower package propagates upward to its consumers. Doing the foundation first
means a consumer package is refactored after the names it imports are already final, so it is touched once
rather than twice. Each commit is still self-contained: FR-003 requires a rename to update every caller in
the same commit, so the tree is green at every boundary regardless of order.

**Test-side findings** (raw material for User Story 2 tasks):

- `tests/evaluation/test_base.py` imports `from sportsbet.evaluation._base import BaseBettor,
  latest_odds_column, market_base` — a private-module import. `BaseBettor` is public and should come from
  `sportsbet.evaluation`; `latest_odds_column` and `market_base` are private helpers tested directly, so they
  are exported (the resolver-helper precedent) or tested through the public API.
- `tests/evaluation/__init__.py` imports `BaseBettor` from `_base`; it should come from the public API.
- `tests/mcp/test_parity_surfaces.py` imports `from sportsbet.cli._cli import main`; `main` is the CLI entry
  point and is exported from `sportsbet.cli`, so the import should be public.
- Test names across the older suites are not uniformly `test_<function>_<behavior>` (e.g.
  `test_required_items_is_deterministic`, `test_stastics_schema` which is also a typo). These are brought to
  the pattern package by package.
