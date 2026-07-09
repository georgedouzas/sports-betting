# Implementation Plan: In-Play (Live) Betting Support

**Branch**: `001-in-play-betting` | **Date**: 2026-07-08 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-in-play-betting/spec.md`

## Summary

Complete the partially-built `development`-branch redesign so users can model and bet on a
match outcome at any chosen moment — pre-match or in-play — on top of an event-snapshot data
model with schema validation. The work finishes the base dataloader (fixtures extraction),
brings the concrete `SoccerDataLoader` and a new sample dataloader onto the new base,
reintroduces the public API, and adapts the bettor/backtest layer to consume the new
moment-aware `X`/`Y`/`O` tables. Data is long-format snapshots (`event_status` ×
`event_time`) validated by `pandera` schemas whose column metadata drives a pivot-based
extraction into wide feature/target/odds tables. Extraction supports supervised and
unsupervised modes only; reinforcement learning is removed from the extraction method and
specified as a separate future method (`make_env()`), designed but not implemented here
(see research.md R1).

## Technical Context

**Language/Version**: Python `>=3.11, <3.14` (target `py311`)

**Primary Dependencies**: scikit-learn (estimator contract), pandas, pandera[pandas]
(schema validation), aiohttp (async download), cloudpickle (persistence), click + rich
(CLI), reflex (GUI, optional extra)

**Storage**: No database. Data loaded from remote CSVs (football-data-style feed on the
repo's `data` branch) and bundled sample CSVs; configured dataloaders persisted to disk via
cloudpickle.

**Testing**: pytest with branch coverage, `--doctest-modules`, randomized ordering
(pytest-randomly), xdist; nox sessions (`tests`, `checks`, `formatting`). GUI excluded from
the default pytest run.

**Target Platform**: Cross-platform Python library + `sportsbet` CLI + `sportsbet-gui` app.

**Project Type**: Single Python package (`src/sportsbet`) with library, CLI, and GUI
surfaces.

**Performance Goals**: Not latency-critical. Extraction over a full multi-season selection
must complete within typical interactive time (seconds–low minutes, dominated by network
download). No new hard performance targets introduced.

**Constraints**: Must preserve the scikit-learn-style public interface (Principle I); all
public DataFrames validated by pandera before use (Principle II); doctested examples must
run offline against the sample dataloader (Principle III); full quality gate must pass
(Principle IV).

**Scale/Scope**: Soccer only. Data volume bounded by available leagues/seasons/divisions
(tens of thousands of matches). One new sample dataloader; ~4 existing bettors adapted.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Evaluated against `.specify/memory/constitution.md` v1.0.0:

- **I. scikit-learn-Compatible API** — PASS (by design). Dataloaders keep the `param_grid`
  selection style and `get_all_params`/`get_odds_types` discovery; bettors remain
  scikit-learn estimators (`fit`/`predict`/`predict_proba`/`bet`, trailing-underscore
  fitted state). API/CLI/GUI parity is an explicit user story (US4) and requirement
  (FR-019). No principle deviation.
- **II. Type Safety & Schema Validation** — PASS (central to the feature). Every public
  stats/odds frame validated against a `pandera` `DataFrameModel` before extraction
  (FR-003); full type annotations required; `mypy` clean.
- **III. Test Coverage & Doctest Discipline** — PASS. The sample dataloader (FR-020) makes
  all docstring examples runnable offline under `--doctest-modules`; regression tests added
  for extraction, schema rejection, and bettor migration; coverage must not regress
  (SC-006).
- **IV. Automated Quality Gates** — PASS. No new tooling exceptions requested; changes go
  through black/docformatter/ruff/interrogate/bandit/safety.
- **V. Documentation as a First-Class Artifact** — PASS. User guide (`dataloader.md`,
  `bettor.md`) and examples currently describe the OLD design and MUST be updated as part
  of this feature; CHANGELOG entry required.

**Initial gate result: PASS** — no violations; Complexity Tracking not required.

**Open clarification carried into Phase 0**: FR-011 (reinforcement learning). Resolved in
research.md R1: RL is removed from `extract_train_data` (keeping a uniform, type-consistent
`(X, Y, O)` return — `Y` is `None` when unsupervised — that upholds Principles I and II) and
specified as a separate future method with a documented forward design, no code this feature.
No gate violation — the capability is scoped and documented, not silently dropped.

**Post-Design re-check (after Phase 1): PASS** — the data model, contracts, and quickstart
preserve the scikit-learn interface and schema-first validation; `extract_train_data` returns
a uniform three-tuple `(X, Y, O)` for both supervised and unsupervised modes (`Y=None` for the
latter), and RL lives in a documented future-method contract (`make_env()`). No new violations
introduced; Complexity Tracking remains empty.

## Project Structure

### Documentation (this feature)

```text
specs/001-in-play-betting/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── datasets-api.md
│   ├── evaluation-api.md
│   └── cli.md
├── checklists/
│   └── requirements.md
└── tasks.md             # /speckit-tasks output (NOT created here)
```

### Source Code (repository root)

```text
src/sportsbet/
├── __init__.py                      # Public type aliases (TrainData, FixturesData, ...)
├── datasets/
│   ├── __init__.py                  # RE-EXPORT: BaseDataLoader, SoccerDataLoader,
│   │                                #   DummySoccerDataLoader, load_dataloader, schemas
│   ├── _base/
│   │   ├── _dataloader.py           # BaseDataLoader: finish extract_fixtures_data()
│   │   └── _schema.py               # Base stats/odds schemas + metadata helpers
│   ├── _soccer/
│   │   ├── _dataloader.py           # SoccerDataLoader conformed to new base
│   │   ├── _schema.py               # Concrete SoccerStatsSchema / SoccerOddsSchema (NEW)
│   │   └── _utils.py                # Re-homed outcome/odds mapping helpers (RESTORE)
│   └── _dummy.py                    # DummySoccerDataLoader w/ in-play sample data (RESTORE)
├── evaluation/
│   ├── _base.py                     # Bettor odds-column parsing → new naming contract
│   ├── _classifier.py               # Adapt to new X/Y/O
│   ├── _model_selection.py          # backtest / BettorGridSearchCV adapt
│   └── _rules.py                    # OddsComparisonBettor odds-type parsing adapt
├── cli/                             # CLI commands surface new target-moment options
└── gui/app/                         # GUI surfaces target-moment selection

tests/
├── conftest.py                      # Fix CLI CONFIG (currently imports deleted loader)
├── datasets/
│   ├── base/                        # test_dataloader.py, test_schema.py (extend)
│   ├── test_soccer.py               # NEW: soccer loader + schema
│   └── test_dummy.py                # RESTORE: sample-data-driven workflow
├── evaluation/                      # Update bettor tests for new X/Y/O
└── samples/                         # stats.csv, odds.csv (in-play sample data)
```

**Structure Decision**: Single-package layout is unchanged. The feature completes and
reconciles the in-flight refactor already present under `src/sportsbet/datasets/_base/` and
`_soccer/`, restores the deleted dummy/utils modules, re-establishes the public API in
`datasets/__init__.py`, and propagates the new `X`/`Y`/`O` column contract into
`src/sportsbet/evaluation/` and the CLI/GUI surfaces.

## Complexity Tracking

> No constitution violations to justify. Table intentionally empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
