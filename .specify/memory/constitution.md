<!--
SYNC IMPACT REPORT
==================
Version change: 1.6.0 → 1.7.0
Rationale: Sharpen Principle VI naming for instance state: a method begins with a
verb while a property names a value (a noun phrase, never verb-first); a public
instance name is either a constructor parameter stored unmodified in __init__ with
no underscore, or a runtime derivation carrying the trailing underscore, whether a
stored attribute or a computed property (so `sources`/`sport` become
`sources_`/`sport_`); anything else an instance exposes is private, while
class-level constants that declare what a class is (a source's sport/name/kind)
stay `ClassVar`. MINOR: materially expanded guidance, no principle removed or
redefined.

Modified principles:
  - VI. Naming, Docstrings, and Module Structure — added the method-verb /
    property-noun distinction and the instance-attribute rule (constructor
    parameter, else trailing-underscore derivation, else private).

Templates requiring updates:
  ✅ .specify/templates/plan-template.md — Constitution Check gate is generic.
  ✅ .specify/templates/spec-template.md — generic; no conflict.
  ✅ .specify/templates/tasks-template.md — generic; no conflict.

---- history ----
Version change: 1.5.0 → 1.6.0
Rationale: Add Principle VI rules the codebase needed: the top level of a package
holds subpackages and its __init__, not loose implementation modules; the shared
leaves (types, constants, build primitives) live in a `core` subpackage, while a
builder lives in the package that owns what it builds (build_dataloader in
dataloaders, build_bettor in evaluation, build_venue in execution) so every import
runs downward and no cycle forms; a public name is re-exported once, at its owning
package's surface, and imported from there rather than promoted a second time by a
parent package; and the top-level package __init__ may carry a fuller front-page
docstring. MINOR: materially expanded guidance, no principle removed or redefined.

Modified principles:
  - VI. Naming, Docstrings, and Module Structure — added the core-subpackage and
    builder-with-what-it-builds rules, the re-export-once rule, and the top-__init__
    docstring exception.

Templates requiring updates:
  ✅ .specify/templates/plan-template.md — Constitution Check gate is generic.
  ✅ .specify/templates/spec-template.md — generic; no conflict.
  ✅ .specify/templates/tasks-template.md — generic; no conflict.

---- history ----
Version change: 1.4.0 → 1.5.0
Rationale: Add DRY to Principle VI — a fact or derivation lives in one place,
don't store what can be derived, and don't reimplement a capability the library
already has. Recurring smell here. MINOR: materially expanded guidance, no
principle removed or redefined.

Modified principles:
  - VI. Naming, Docstrings, and Module Structure — added the DRY rule.

Templates requiring updates:
  ✅ .specify/templates/plan-template.md — Constitution Check gate is generic.
  ✅ .specify/templates/spec-template.md — generic; no conflict.
  ✅ .specify/templates/tasks-template.md — generic; no conflict.

---- history ----
Version change: 1.3.0 → 1.4.0
Rationale: Sharpen Principle VI with three rules the codebase was missing: a
module is named for the concern it owns and not for the data it consumes or a
colliding dependency concept; a public entry point documents every parameter and
its return in an Args/Returns block; a public name is re-exported through the
package __init__ and imported from the public surface, never from the private
module that defines it. Adds verb-honesty (load/resolve is not build). MINOR:
materially expanded guidance, no principle removed or redefined.

Modified principles:
  - VI. Naming, Docstrings, and Module Structure — added module naming, the
    public-surface re-export rule, verb honesty, and mandatory Args/Returns on
    public entry points.

Templates requiring updates:
  ✅ .specify/templates/plan-template.md — Constitution Check gate is generic.
  ✅ .specify/templates/spec-template.md — generic; no conflict.
  ✅ .specify/templates/tasks-template.md — generic; no conflict.

---- history ----
Version change: 1.2.0 → 1.3.0
Rationale: Fold the full code conventions into Principle VI and remove the
separate CONVENTIONS.md, so the constitution is the single, self-contained source
of the naming, docstring, module-structure and test rules rather than a brief
that points at an external file. MINOR: materially expanded guidance, no
principle removed or redefined.

Modified principles:
  - VI. Naming, Docstrings, and Module Structure — expanded from a brief that
    bound to CONVENTIONS.md into the complete rules, stated inline.
Modified sections:
  - Development Workflow & Quality Gates — the pre-PR conformance check names
    Principle VI directly rather than CONVENTIONS.md.
Removed files:
  - CONVENTIONS.md — its content now lives in Principle VI.

Templates requiring updates:
  ✅ .specify/templates/plan-template.md — Constitution Check gate is generic.
  ✅ .specify/templates/spec-template.md — generic; no conflict.
  ✅ .specify/templates/tasks-template.md — generic; no conflict.

---- history ----
Version change: 1.1.0 → 1.2.0
Rationale: Fold the code conventions (CONVENTIONS.md) into the constitution as a
sixth principle, so naming, docstrings, module structure and test layout are a
gate the plan's Constitution Check verifies rather than taste enforced by hand.
CONVENTIONS.md remains the detailed, portable companion; the principle states the
rules in brief and names it binding. MINOR: adds a principle, no principle
removed or redefined.

Added principles:
  - VI. Naming, Docstrings, and Module Structure — binds the code to
    CONVENTIONS.md and states its rules in brief.
Modified sections:
  - Development Workflow & Quality Gates — conformance to CONVENTIONS.md is part
    of the pre-PR check and the Constitution Check.

---- history ----
Version change: 1.0.0 → 1.1.0
Rationale: Sync the constitution to the surfaces that exist. Feature 005 removed
the GUI and added the MCP server, so Principle I named a surface that is gone and
omitted the one that replaced it. Adds the rule that an agent is a client of the
surfaces rather than a component of the library, and that credentialled or
side-effecting capabilities live behind an optional extra. MINOR: materially
expanded guidance, no principle removed or redefined.

---- history ----
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
state. The three delivery surfaces — Python API, CLI (`sportsbet`), and MCP
server (`sportsbet-mcp`) — MUST expose the same underlying capabilities without
one surface holding logic the others cannot reach.

An agent is a client of those surfaces, never a component of the library: no
model, model key, model choice, or agent loop enters the package, so estimators
stay deterministic and testable.

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

### VI. Naming, Docstrings, and Module Structure

These rules are binding, not advisory. They are generic Python and hold in any
project; the examples are drawn from this codebase. The automated gate
(Principle IV) checks the mechanical parts; this principle is the taste the gate
cannot check.

**Don't repeat yourself.** A fact, a definition, or a derivation lives in exactly
one place — repetition is the typical smell here, so when the same thing is
expressed twice, collapse it to one. Do not store what can be derived from what
you already keep: persist the source and derive the projection on demand, not
both (a saved dataloader keeps its downloaded snapshots and rebuilds its training
data from them, rather than storing the training data alongside). Do not
reimplement a capability the library already has: save and load a dataloader with
its own `save`/`load`, not a second copy of that logic in another module.

**Files and layout.** A module reads top to bottom in one order: a one-line
imperative module docstring, the license header, `from __future__ import
annotations`, then imports grouped standard library / third party / first party
(`ruff` sorts them, so do not sort by hand), then module constants (`UPPER_CASE`)
and type aliases, then functions in dependency order — a name is defined before
it is used, so the small helpers come first and the function the module exists
for comes last. One module is one concern; when a file grows two, split it. A
definition lives in the module that owns it. A base module is self-contained and
imports no sibling: its purpose is to be imported, not to import, so a base that
needs a sibling's code absorbs it by merging rather than importing. A type-only
alias from `commons` under `TYPE_CHECKING` is not a sibling. No import inside a
function body papers over a cycle — fix the cycle; the only lazy import defers an
optional dependency and carries a `# noqa: PLC0415` with a reason.

The top level of a package holds subpackages and its `__init__`, not loose
implementation modules: the shared leaves the whole tree imports — the type
vocabulary (`_types`), the shared constants (`_params`), the shared building
primitives (the build error, the reference loader) — live in a `core` subpackage,
and the rest of the tree imports them from `core`, never from the package root. A
builder lives in the package that owns what it builds and is re-exported from
there: `build_dataloader` in `dataloaders`, `build_bettor` in `evaluation`,
`build_venue` in `execution`. That keeps every import running downward — `core` →
domain packages → surfaces — so no cycle can form, and the top `__init__` is left
holding nothing but its docstring. Prefer this layering over a `# noqa: E402` and
an explanatory comment that paper over an out-of-order import.

**Naming.** A function name begins with a verb and names what the function
actually does or returns: `count_common_prefix`, not `common_prefix_length`;
`normalize_identity`, not `transform_identity`; `build_roster`, not `roster`. A
name that begins with a noun describes a value, and a function is not a value. The
verb is honest: a function that loads or resolves an object from a reference is
`load_`/`resolve_`, not `build_`. Avoid empty verbs that say nothing — `process`,
`handle`, `manage`, `transform` with no object. A method is an action and begins
with a verb; a property names a value and is a noun phrase (`sport_`, `sources_`),
never verb-first. State the instance derives at runtime carries a trailing
underscore — the scikit-learn convention — whether it is a stored attribute
(`odds_type_`, `target_event_status_`) or a computed property (`sources_`,
`sport_`). A public instance name is therefore one of two things: a constructor
parameter, stored unmodified under its own name in `__init__` with no underscore,
or a derivation, carrying the trailing underscore. Anything else an instance
exposes is private. Class-level constants that declare what a class is — a
source's `sport`, `name`, `kind` — are `ClassVar` and stand apart from this. Names
come from the domain, used consistently.

A module is named for the concern it owns: a descriptive noun for what it does or
holds (`_resolver`, `_schedule`, `_factory`), not for the data it consumes
(`_selection` for a module that builds objects) and not for the surface that
happens to call it. It must not reuse a name that collides with a dependency's
concept — `_selection` shadowed scikit-learn's `model_selection`.

**Public surface.** Implementation modules, classes and helpers are private
(`_name`); the package `__init__` re-exports the public surface with an explicit
`__all__`. A public function, class or exception used outside the module that
defines it is re-exported through its owning package's `__init__` and imported
from that surface — `from sportsbet.dataloaders import build_dataloader`, never
`from sportsbet.dataloaders._factory import build_dataloader`. It is re-exported
once, where it lives: a parent package does not re-export a subpackage's surface a
second time,
so a consumer imports from the package that owns the name. This holds for all
code, production and tests alike; reaching into another package's private module
for a public name is the smell the re-export removes.

**Docstrings.** The summary line is one line, imperative, and says what the thing
does — never `Implements the ...`, `This function ...`, `A class that ...`, which
are meta narration. Write it in plain English: prefer simple, direct words over
clever or roundabout phrasing, and if a line reads awkwardly out loud, rewrite
it. For most functions the one line is the whole docstring, and a private helper
never gets more. A public entry point and a public class carry an
`Args`/`Returns` block (and `Raises` where they raise) that documents every
parameter and what is returned — a public signature is not self-explanatory to a
caller, so the docstring names each argument and the result. Keep each entry
terse and the whole body to a few sentences. Never describe what the code does
not do, never restate the code, no essays, no editorializing. Public API carries
a runnable example checked by the doctest run; a network-touching class does not.

The top-level package `__init__` is the exception to the one-line rule: it is the
library's front page, so it may carry a fuller docstring — a tagline and a short
overview of the submodules — rather than a single imperative line.

**Comments.** Almost none. The names say what, the docstring says why. An inline
comment that explains the next line means the line or its names are unclear — fix
those. The only comments in source are the license header and, rarely, a
`# noqa`/`# type: ignore` with a reason.

**Errors.** Build the message in a variable, then raise it (`ruff EM`/`TRY`).
Raise a specific named exception defined for the module or package
(`SelectionError`, `ExecutionError`), not a bare `Exception`/`ValueError` where a
named one carries meaning. The message tells the reader what to do — the variable
that was missing, the value that did not match. Do not catch and swallow; catch
narrowly or let it propagate.

**Control flow.** Functions are small and do one thing; a function that needs a
paragraph of docstring body to explain its branches is two functions. Return
early with guard clauses. No deep nesting — extract a helper before the third
level of indentation.

**Tests.** The test tree mirrors the source tree (`sources/_stats/_nba.py` is
tested by `tests/sources/stats/test_nba.py`). A test is named
`test_<function>_<behavior>`: it begins with the function it exercises, then a
terse behavior phrase with articles dropped, and its docstring is a single line —
never a multi-line body. A test never imports a private name from a private
module; it uses the public API the way a user does, and if it needs an internal,
the internal wants to be public. No test reaches the network — use a recorded
payload, a fake, or a locally served page. Fixtures are typed, small, and live in
the nearest `conftest.py`. The three surfaces (Python API, CLI, MCP server)
expose the same capabilities, and a parity test asserts it so it cannot drift. A
credential is named, never passed: a function, command flag or tool argument
takes the name of the variable holding the secret and reads it where it is used;
a secret never becomes an argument, a log line or a pickle.

Rationale: these are the rules the maintainer has enforced by hand across the
refactor. Folding them into the constitution, rather than a separate file, makes
them one self-contained source of truth and a gate the Constitution Check
verifies, so a change conforms before review rather than after.

## Technology & Tooling Standards

- **Language**: Python `>=3.11, <3.14`; code targets `py311` and MUST remain
  compatible across all supported minor versions.
- **Core dependencies**: `scikit-learn`, `pandas`, `pandera`, `click` (CLI),
  `rich`, `aiohttp` (async data fetching); the optional `mcp` extra uses `mcp`
  and ships the `sportsbet-mcp` server. New runtime dependencies MUST be
  justified and added to `pyproject.toml`, not vendored ad hoc. A capability
  that needs a credential or performs a real-world side effect MUST live behind
  an optional extra, never in the default install.
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
  `nox` invocations) and resolve all findings, and MUST conform to Principle VI
  (Naming, Docstrings, and Module Structure).
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

**Version**: 1.7.0 | **Ratified**: 2026-07-08 | **Last Amended**: 2026-07-26
