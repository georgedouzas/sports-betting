# Implementation Plan: Conventions conformance

**Branch**: `007-conventions-conformance` | **Date**: 2026-07-20 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/007-conventions-conformance/spec.md`

## Summary

Bring every package under `src/sportsbet` and its mirrored tests into conformance with Principle VI
(`CONVENTIONS.md`): one-line imperative module docstrings, verb-first exact function names, single-line
function docstrings, bottom-up self-contained modules, mirrored tests named `test_<function>_<behavior>`.
Behaviour-preserving. One package per commit, each ending with the full gate green on three versions.

## Technical Context

**Language/Version**: Python `>=3.11, <3.14`, targeting `py311`. No new dependency.

**Primary Dependencies**: The existing toolchain only: `black`, `docformatter`, `ruff`, `mypy`,
`interrogate`, `bandit`, `pip-audit`, `pytest` with `--doctest-modules`, `nox`, `pdm`.

**Storage**: N/A. This is a refactor.

**Testing**: The existing suite, unchanged in behaviour. Test files are renamed, moved to the mirrored
path, and their names and docstrings brought to `test_<function>_<behavior>` with one line. No test
changes what it asserts. No test touches the network (the socket guard stays).

**Target Platform**: Linux, macOS, Windows, on 3.11/3.12/3.13, matching the current CI matrix.

**Project Type**: Single library with three surfaces.

**Performance Goals**: None. The gate's wall-clock is the only constraint and it is unchanged.

**Constraints**: Behaviour-preserving (FR-016). Public API unchanged except Principle VI renames, which
propagate to callers, `__all__`, tests, docs and `CHANGELOG.md` in the same commit (FR-003). One package
per commit (FR-017). No new dependency (FR-018). `docs/generated` is regenerated, never hand-edited (FR-019).

**Scale/Scope**: 34 source modules across 6 packages plus 4 top-level modules, and their mirrored tests.
28 modules already show a non-conforming module docstring (see data-model). The gate is the definition of
done per package.

## Constitution Check

*GATE: checked against constitution v1.2.0. This feature is defined BY the constitution (Principle VI),
so the check is that the work honours all six principles, not that it introduces a new capability.*

| Principle | Status | Basis |
| --- | --- | --- |
| **I. scikit-learn-Compatible API** | CONSTRAINS this feature | The contract surface is fixed and NOT renamed even where Principle VI's verb-first rule would apply: `fit`, `predict`, `predict_proba`, `bet`, `score`, `get_params`, `set_params`, trailing-underscore fitted attributes, constructor parameter names, and the public estimator classes `BaseBettor`, `ClassifierBettor`, `OddsComparisonBettor`, `BettorGridSearchCV`, `BaseDataLoader`, `DataLoader`. Principle I wins (FR-004, research D1). |
| **II. Type Safety & Schema Validation** | PRESERVED | Annotations and `pandera` schemas are untouched in meaning; a renamed schema helper updates its callers. `mypy` staying green at each commit proves it. |
| **III. Test Coverage & Doctest Discipline** | PRESERVED and improved | No test changes what it asserts, so coverage does not drop. Doctests are reworded with their functions and stay executable (`--doctest-modules`). Test layout is brought to Principle VI, which is User Story 2. |
| **IV. Automated Quality Gates** | IS the definition of done | Every package commit ends with `formatting`, `checks`, `tests` green on three versions (FR-017, SC-006). Nothing is silenced with an inline disable. |
| **V. Documentation as a First-Class Artifact** | PRESERVED | Every public name keeps a runnable example; a public rename updates `docs/examples`, the user guide, and the changelog (FR-019, SC-007, SC-008). `docs/generated` regenerates. |
| **VI. Naming, Docstrings, and Module Structure** | IS this feature | Every FR is a Principle VI rule made testable. |

No violations. The one tension, Principle I vs VI on the estimator surface, is resolved in favour of I and
recorded in the spec's Clarifications and FR-004. Complexity Tracking stays empty.

## Project Structure

### Documentation (this feature)

```text
specs/007-conventions-conformance/
├── plan.md              # This file
├── spec.md              # The requirements
├── research.md          # Phase 0: contract surface, package order, docstring-rewrite rule
├── data-model.md        # Phase 1: the violation inventory, per package
├── quickstart.md        # Phase 1: the per-package gate recipe
├── contracts/
│   └── rename-ledger.md # Phase 1: old name -> new name -> callers, filled as packages are done
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2, one task group per package
```

### Source Code (repository root)

No files move for structural reasons beyond what conformance requires (a test to its mirrored path, a
sibling definition merged into a base). The package layout is unchanged:

```text
src/sportsbet/
├── _params.py _selection.py _artifacts.py __init__.py   # top-level modules
├── sources/     _base.py _resolver.py _schema.py _utils.py _stats/* _odds/*
├── dataloaders/ _base.py _sourced.py _schema.py
├── evaluation/  _base.py _classifier.py _model_selection.py _rules.py
├── execution/   _base.py _browser.py _credentials.py _place.py _schedule.py
├── cli/         _cli.py _data.py _betting.py _execution.py _options.py _utils.py
└── mcp/         _server.py

tests/  mirrors the above, one test module per source module, subpackages with __init__
```

**Structure Decision**: The unit of work and of review is the package. The sweep runs bottom-up so a
rename in a lower package is already propagated when its consumers' turn comes:

1. **sources** (with the top-level `_params` it depends on) — the foundation, imported by everything below.
2. **dataloaders** — imports sources.
3. **evaluation** — the estimators; Principle I constrains it most.
4. **execution** — imports evaluation and dataloaders for types.
5. **selection & artifacts** (`_selection`, `_artifacts`, top `__init__`) — the glue both surfaces share.
6. **cli** — a surface over the glue.
7. **mcp** — a surface over the glue.

Each is one commit. Within a package the source conforms first, then its mirrored tests, then the gate runs.

## Notable design decisions carried from research

- **The contract surface is fixed** (research D1). The sweep must not rename `fit`/`predict`/`bet`/… or the
  estimator classes. Everything else conforms.
- **A module docstring is reworded, not re-scoped** (research D2). `"""Implements the X that Y."""` becomes a
  one-line imperative of Y. The module keeps its content.
- **Bottom-up order minimizes churn** (research D3), but each commit is self-contained because a rename
  propagates to all callers in the same commit (FR-003).

## Complexity Tracking

> Fill ONLY if Constitution Check has violations that must be justified.

No violations. Table intentionally empty.
