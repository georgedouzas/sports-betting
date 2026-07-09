<!--
SYNC IMPACT REPORT
==================
Version change: (unversioned template) → 1.0.0
Rationale: Initial ratification. First concrete constitution replacing the
placeholder template; establishes five core principles derived from the
project's established, enforced engineering practices. MAJOR baseline.

Modified principles: none (initial definition)
Renamed principles:
  [PRINCIPLE_1_NAME] → I. scikit-learn-Compatible API
  [PRINCIPLE_2_NAME] → II. Type Safety & Schema Validation
  [PRINCIPLE_3_NAME] → III. Test Coverage & Doctest Discipline
  [PRINCIPLE_4_NAME] → IV. Automated Quality Gates (NON-NEGOTIABLE)
  [PRINCIPLE_5_NAME] → V. Documentation as a First-Class Artifact

Added sections:
  - Technology & Tooling Standards (was [SECTION_2_NAME])
  - Development Workflow & Quality Gates (was [SECTION_3_NAME])
  - Governance (filled)

Removed sections: none

Templates requiring updates:
  ✅ .specify/templates/plan-template.md — Constitution Check gate is generic
     ("[Gates determined based on constitution file]"); aligns, no edit needed.
  ✅ .specify/templates/spec-template.md — generic; no principle-specific
     mandatory sections conflict.
  ✅ .specify/templates/tasks-template.md — generic; task categories are
     compatible with testing/quality principles.
  ✅ .specify/templates/checklist-template.md — generic; no conflict.

Follow-up TODOs:
  - RATIFICATION_DATE set to first-adoption date (today). If an earlier formal
    adoption date exists, amend this field.
-->

# sports-betting Constitution

## Core Principles

### I. scikit-learn-Compatible API

Public estimators (dataloaders, bettors) MUST conform to the scikit-learn
estimator contract: constructor parameters are stored unmodified, state learned
during fitting uses trailing-underscore attributes, and behavior is configured
through explicit parameters (including `param_grid`) rather than hidden global
state. The three delivery surfaces — Python API, CLI (`sportsbet`), and GUI
(`sportsbet-gui`) — MUST expose the same underlying capabilities without one
surface holding logic the others cannot reach.

Rationale: Interoperability with the scikit-learn ecosystem (pipelines, model
selection, cross-validation) is the library's core value proposition; drift from
the estimator contract silently breaks downstream user code.

### II. Type Safety & Schema Validation

All public and internal code MUST carry complete type annotations and pass
`mypy` with no new ignored errors (`warn_unused_ignores` is enforced). Every
DataFrame that crosses a public boundary (training data, fixtures, odds) MUST be
validated against an explicit `pandera` schema. Data-shape assumptions MUST be
declared as schemas, not enforced by ad-hoc runtime checks scattered through the
code.

Rationale: The package is distributed as typed (`Typing :: Typed`) and operates
on tabular data whose column contracts are easy to break; static types plus
schema validation catch integration errors before they reach users' models.

### III. Test Coverage & Doctest Discipline

Every behavioral change MUST ship with tests under `tests/` or `src/`. The test
suite runs with `pytest`, branch coverage enabled, randomized ordering
(`pytest-randomly`), and `--doctest-modules`: therefore every code example in a
docstring MUST be correct and executable. New logic MUST NOT reduce coverage of
the module it touches. Bug fixes MUST include a regression test that fails
before the fix.

Rationale: Randomized, doctest-inclusive testing keeps examples honest and
guards against order-dependent flakiness in statistical/backtesting code where
subtle regressions are otherwise invisible.

### IV. Automated Quality Gates (NON-NEGOTIABLE)

Code MUST pass the full automated gate before merge: `black` and `docformatter`
formatting, `ruff` linting (the configured rule set, line length 120),
`interrogate` docstring coverage, the `bandit` security check, and the
`pip-audit` dependency audit.
These gates run via `pre-commit` locally and `nox` in CI. Failures MUST be fixed
at the source; disabling a rule inline requires a justifying comment and is the
exception, not the workaround.

Rationale: A single, machine-enforced quality bar removes style debate, keeps
the diff reviewable, and prevents security-sensitive dependencies (this library
handles network data fetching) from silently degrading.

### V. Documentation as a First-Class Artifact

Every public module, class, and function MUST have a Google-style docstring
(enforced by `ruff` pydocstyle and `interrogate`). User-facing behavioral
changes MUST update the affected docs under `docs/` (user guide, examples, or
API generation) and, when they change public behavior, add or amend a
`CHANGELOG.md` entry. Runnable examples in `docs/examples/` are part of the
documented contract and MUST stay working.

Rationale: The library is adopted through its documentation and gallery
examples; undocumented capabilities effectively do not exist for users and rot
quickly without executable coverage.

## Technology & Tooling Standards

- **Language**: Python `>=3.11, <3.14`; code targets `py311` and MUST remain
  compatible across all supported minor versions.
- **Core dependencies**: `scikit-learn`, `pandas`, `pandera`, `click` (CLI),
  `rich`, `aiohttp` (async data fetching); the optional `gui` extra uses
  `reflex`. New runtime dependencies MUST be justified and added to
  `pyproject.toml`, not vendored ad hoc.
- **Build & packaging**: PDM with SCM-derived versioning; the package layout is
  `src/`-based. Do not hand-edit generated version metadata.
- **Task automation**: `nox` sessions (`tests`, `checks`, `formatting`, `docs`,
  `changelog`, `release`) are the canonical entry points; local shortcuts run
  through `pdm run`.
- **Style constants**: line length 120; docstring convention Google; string
  normalization disabled (`black skip-string-normalization`).

## Development Workflow & Quality Gates

- Work happens on feature branches; `main` is the release branch and MUST stay
  green.
- Before opening a PR, contributors MUST run `pdm run formatting`,
  `pdm run checks`, and `pdm run tests` (or the equivalent `pre-commit` +
  `nox` invocations) and resolve all findings.
- CI (GitHub Actions `ci.yml` / `doc.yml`) re-runs the same gates; a red CI run
  blocks merge.
- Every PR MUST state which principles it touches and confirm the gates pass;
  reviewers verify compliance, not just correctness.
- Releases follow semantic versioning and MUST update `CHANGELOG.md` via the
  `changelog` session before tagging.

## Governance

This constitution supersedes ad-hoc conventions and prior undocumented practice.
It applies to all code, documentation, and tooling changes in this repository.

- **Amendments**: Proposed via PR that edits this file, states the rationale,
  and updates the version and Sync Impact Report. Amendments that add or remove
  a principle or governance rule require the maintainer's approval.
- **Versioning policy**: Semantic versioning of the constitution itself.
  MAJOR = backward-incompatible removal or redefinition of a principle or
  governance rule; MINOR = a new principle/section or materially expanded
  guidance; PATCH = clarifications and non-semantic wording fixes.
- **Compliance review**: Every PR and code review MUST verify adherence to the
  Core Principles. Deviations MUST be justified in the PR description and, where
  they represent added complexity, recorded in the plan's Complexity Tracking
  table. Unjustified violations block merge.
- **Runtime guidance**: Contributor-facing operational guidance lives in
  `CONTRIBUTING.md` and `docs/development/`; those documents MUST stay
  consistent with this constitution.

**Version**: 1.0.0 | **Ratified**: 2026-07-08 | **Last Amended**: 2026-07-08
