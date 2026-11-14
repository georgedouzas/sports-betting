# Design

## Context

Constitution v1.2.0 added Principle VI, "Naming, Docstrings, and Module Structure", with `CONVENTIONS.md` as its
authoritative detail. Where the two could be read differently, `CONVENTIONS.md` governs. This change brings every
package under `src/sportsbet` and its mirrored tests into conformance. It is behaviour-preserving. The public API stays
the same except for the Principle VI renames, and each package is one commit that ends with the full gate green on
3.11, 3.12 and 3.13.

### Violation inventory at the green baseline

28 of 34 modules carried a non-conforming module docstring:

- sources, `_utils.py`: `Includes utilities shared by the sports.`
- sources, `_odds/_football_data.py`, `_stats/_football_data.py`: `Implements the sources backed by the
  football-data.co.uk feed.`
- sources, `_odds/_odds_api.py`: `Implements the odds source backed by The Odds API.`
- sources, `_odds/_sample.py`, `_stats/_sample.py`: `Implements the sources of the sample data ...`
- sources, `_stats/_euroleague.py`: `Implements the statistics source backed by the EuroLeague's official API.`
- sources, `_stats/_nba.py`: `Implements the statistics source of the NBA, backed by ESPN.`
- sources, `__init__.py`: `It provides the sources the data comes from.`
- dataloaders, `_base.py`: `Implements the base dataloader class shared by all dataloaders.`
- dataloaders, `_sourced.py`: `Implements the dataloader shared by every sport whose data comes from sources.`
- dataloaders, `__init__.py`: `It provides the dataloaders that shape the data for modelling.`
- evaluation, `_base.py`, `_model_selection.py`: `Includes base class and functions for evaluating betting strategies.`
- evaluation, `_classifier.py`: `Create a bettor based on a classifier.` (verb-first, tighten only)
- evaluation, `_rules.py`: `Create a bettor based on betting rules.` (verb-first, tighten only)
- evaluation, `__init__.py`: `It provides the tools to evaluate the performance of predictive models.`
- execution, `__init__.py`: `It provides the placing of the bets a bettor found.`
- glue, `_artifacts.py`: `Implements the file a surface writes so that the data is downloaded once and reused.`
- glue, `_selection.py`: `Implements the selection, which is what a surface is told instead of being handed a file.`
- cli, `_cli.py`, `_data.py`, `_betting.py`, `_execution.py`, `_options.py`, `_utils.py`: `Module that contains the ...
  of the CLI.`
- mcp, `_server.py`: `Implements the server that lets an agent drive the library.`

Already conforming: `execution/_base.py`, `_browser.py`, `_credentials.py`, `_place.py`, `_schedule.py`,
`sources/_base.py`, `_resolver.py`, `_schema.py`, and `dataloaders/_schema.py`.

Other violation categories, rescanned per package:

- Noun-first or empty-verb function name (FR-001). A `def` not starting with a verb, or starting with `process`,
  `handle` or `transform` with no object. Exclude the contract surface.
- Private helper with Args or Returns (FR-006). A `def _name` whose docstring has a section or body paragraph.
- Base module importing a sibling (FR-009). `from ._x import` in a `_base.py`. Merge the definition into the base.
- In-function import (FR-010). `import` inside a `def` body without a `# noqa: PLC0415` optional-extra reason.
- Explanatory inline comment (FR-011). A `#` comment inside a function body that explains the next line.
- Test not at the mirrored path (FR-012). A source module with no test module at the mirrored path.
- Test name not `test_<function>_<behavior>` (FR-013). A `def test_` that does not begin with the function under test.
- Multi-line test docstring (FR-013). A `def test_` whose docstring spans more than one line.
- Test importing a private name (FR-014). `from sportsbet.<pkg>._<mod> import` in a test.

### Rename ledger

- sources: no rename. Docstrings and test names only.
- dataloaders: no rename. `_schema` merged into `_base`, and the merged helpers were never public.
- evaluation: `complementary_events` became `derive_complementary_events`. Updated: `evaluation.__all__`, the `_base`
  definition, doctest and internal call, `tests/evaluation/test_base.py`, `tests/dataloaders/test_basketball.py`,
  `docs/examples/modelling/plot_custom_bettor.py`, `docs/overview/user_guide/bettor.md`.
- evaluation: the private `market_base` became the exported `derive_market_base`. Updated: `evaluation.__all__`,
  `_base`, `_rules`, `tests/evaluation/test_base.py`.
- evaluation: the private `latest_odds_column` became the exported `find_latest_odds_column`. Updated:
  `evaluation.__all__`, `_base`, `_rules`, `execution/_place.py`, `tests/evaluation/test_base.py`.
- execution: no rename. The `__init__` module docstring only.
- glue: no rename. Module docstrings and plain wording only.
- cli: no rename. The command functions are the command names and so the contract.
- mcp: no rename. The tool functions are the tool names and so the contract.

Follow-up cleanup during the sweep, recorded as removals and not renames:

- The reconciliation documentation was rewritten from the removed `ReconciliationReport`, `UnmatchedError` and
  `resolve` API to the current `resolve_odds(stats, odds, aliases)`, which pairs by fuzzy name and drops an unmatched
  match silently.
- The dead `max_unmatched_rate` parameter was removed from the `DataLoader` constructor, `build_dataloader`, the
  `--max-unmatched-rate` command line option, the MCP tool signatures, the parity test, and the `DataLoader` docstring,
  which also documented a `reconciliation_` attribute that was never set.

## Goals / Non-Goals

**Goals:**

- Every module docstring is one imperative line (SC-001).
- Every public function name begins with a verb, except the scikit-learn contract names (SC-002).
- Every private helper and ordinary function has a one-line docstring. Args and Returns blocks appear only on public
  entry points and public classes (SC-003).
- No base module imports a sibling, and no function body contains an import except a deferred optional extra (SC-004).
- Tests mirror the source and are named `test_<function>_<behavior>` with a one-line docstring (SC-005).
- The full gate is green on 3.11, 3.12 and 3.13 at every package commit (SC-006).
- Every runnable example still runs (SC-008).

**Non-Goals:**

- Any change to what a symbol does, returns or accepts. That is a behaviour change and out of scope.
- Renaming the scikit-learn contract surface.
- A new runtime dependency or a change to what the extras install.
- Hand edits to `docs/generated`.

## Decisions

- Clarified: What is an allowed public API rename versus a forbidden behaviour change? -> A rename changes only the
  identifier of a public symbol and updates every caller, the `__init__` export and `__all__`, the tests and the docs in
  the same commit. Inputs, outputs, side effects and semantics stay identical.
- Clarified: How is a module docstring that says "Implements the X that ..." or "It provides ..." rewritten? -> As one
  imperative line that says what the module does, the same thing the old sentence described. For example
  `"""Implements the base classes of the data sources."""` becomes
  `"""Read the raw content a data source needs, and define the base a source implements."""`.
- Clarified: Do the scikit-learn estimator names conform to Principle VI? -> Principle I wins on the estimator contract.
  `fit`, `predict`, `predict_proba`, `bet`, `score`, `get_params`, `set_params`, the trailing-underscore fitted
  attributes and the constructor parameter names are not renamed.

### D1. The fixed scikit-learn contract surface

These names are fixed by Principle I: the estimator methods `fit`, `predict`, `predict_proba`, `bet`, `score`,
`get_params`, `set_params`, every trailing-underscore fitted attribute (`betting_markets_`, `target_event_status_`,
`feature_names_out_`, `reconciliation_`, and so on), the constructor parameter names (`param_grid`, `stats`, `odds`,
`alpha`, `stake`, and so on), and the public classes `BaseBettor`, `ClassifierBettor`, `OddsComparisonBettor`,
`BettorGridSearchCV`, `BaseDataLoader`, `DataLoader`, `BaseSource`, `BaseStatsSource`, `BaseOddsSource`, `BaseVenue`.
Renaming `fit` to satisfy a verb-object rule would break every downstream pipeline and grid search. Of the non-class
names in `evaluation.__all__` and `dataloaders.__all__`, `backtest`, `load_bettor`, `save_bettor` and `load_dataloader`
already begin with a verb. `complementary_events` was the one noun-first public function.

Alternative considered: renaming the estimator classes to factory functions. Rejected as an API change.

### D2. A module docstring is reworded, not rescoped

The fix for a narrating module docstring is a wording change, never a scope change, which keeps the work
behaviour-preserving. A docstring that already starts with a verb, such as `"""Create a bettor based on a
classifier."""`, is kept or tightened.

### D3. Bottom-up package order

The sweep runs `sources` (with `_params`), `dataloaders`, `evaluation`, `execution`, the glue (`_selection`,
`_artifacts`, top `__init__`), `cli`, `mcp`. A consumer package is refactored after the names it imports are final, so
it is touched once. Each commit is still self-contained, because a rename updates every caller in the same commit.

### D4. Tested private helpers become public

A private helper that a test exercises directly is exported so the test uses the public API, following the resolver
helper precedent. This is why `derive_market_base` and `find_latest_odds_column` joined `sportsbet.evaluation`.

### D5. The changelog is generated

`CHANGELOG.md` is generated from conventional commits, so each before and after lives in the package commit body rather
than a hand edit to the generated file.

### Test-side findings at the baseline

- `tests/evaluation/test_base.py` imported `BaseBettor`, `latest_odds_column` and `market_base` from
  `sportsbet.evaluation._base`.
- `tests/evaluation/__init__.py` imported `BaseBettor` from `_base`.
- `tests/mcp/test_parity_surfaces.py` imported `main` from `sportsbet.cli._cli` although `sportsbet.cli` exports it.
- Older test names did not follow the pattern, for example `test_required_items_is_deterministic` and the misspelled
  `test_stastics_schema`.

### Requirement coverage

FR-001, FR-003 and FR-016 produce the user-visible renames, which land in the `evaluation/bettors` delta. FR-004 is the
reason `complementary_events_` and `COMPLEMENTARY_EVENTS` keep their names. The removal of the inert
`max_unmatched_rate` parameter lands in the `sources/team-reconciliation` delta. FR-002 and FR-005 to FR-015 are
conventions for names, docstrings, module structure and tests, and FR-017 to FR-019 are process rules. None of them
changes what the library does, so they are recorded in this design and not as requirements.

## Risks / Trade-offs

- A name the scikit-learn contract fixes can look noun-like. It is left alone because Principle I wins.
- A rename could collide with an existing public name. The rename is reconsidered rather than forced, and the collision
  is called out.
- A base module could need a sibling's code where merging would mix two separate concerns. The conflict is surfaced and
  the merge is not forced.
- A docstring body could carry a reason a single line would lose. The reason moves to the module docstring, or is
  dropped only if it restated the code.
- The renames break callers of `complementary_events` and of `max_unmatched_rate`. The commit bodies carry the before
  and after for the generated changelog.

## Migration Plan

Each package follows the same recipe: rescan against the inventory, reword module docstrings, rename and propagate,
collapse helper docstrings, make the base self-contained, bring the tests to the mirror, add the changelog entry for a
public rename, run `pdm run formatting`, `pdm run checks` and `pdm run tests` on 3.11, 3.12 and 3.13, then commit. The
baseline needs the browser installed for the execution tests:
`pdm run python -m playwright install --only-shell chromium`.

A user migrates by replacing `complementary_events(markets)` with `derive_complementary_events(markets)` and by removing
any `max_unmatched_rate` argument or `--max-unmatched-rate` option.
