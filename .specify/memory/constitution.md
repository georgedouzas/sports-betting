<!--
SYNC IMPACT REPORT
==================
Version change: 2.2.0 -> 2.3.0
Rationale: The gate is the full sequence in order, formatting, then checks, then the documentation build, then tests.
The documentation build executes the examples, so a broken example fails the gate and the build is never skipped. A
reused build environment can carry a stale toolchain, so a clean run can find a lint rule a cached one missed. Also
restructure the Code Conventions for clarity, splitting `Files & Module Structure` into `Module Structure` and `Package
Layering`, and lifting the credential rule out of `Tests` into its own `Credentials` subsection. No rule text changed.
MINOR: expanded guidance, nothing removed.

Modified sections:
  - Development Workflow: run the full gate in order, including the documentation build.
  - Project Profile: the gate order and commands, the docs build executes the examples, cached environments can hide a
    finding.
  - Code Conventions: split `Files & Module Structure` into `Module Structure` and `Package Layering`, and gave the
    credential rule its own `Credentials` subsection. Content unchanged.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 2.1.0 -> 2.2.0
Rationale: Fold in the lessons of the whole-src conformance sweep. The `from __future__ import annotations` import is
conditional, kept only where a forward reference or a `TYPE_CHECKING`-only name needs it, not carried by every module.
A general `_utils` module is allowed for assorted small helpers, and a helper earns its own module only for a distinct
role. An `__init__` carries no license header. DRY extends from constants to behavior, so two surface serializations
that are distinct contracts stay apart. The gate's verdict is the nox session summary, not a piped exit code. The
`__future__` change redefines a stated rule, the rest is expanded guidance, so MINOR.

Modified sections:
  - Files & Module Structure: made `from __future__ import annotations` conditional, and allowed a general `_utils`.
  - Public Surface: an `__init__` carries no license header.
  - Don't Repeat Yourself: extended the one-fact rule to behavior and distinct surface contracts.
  - Project Profile: added how to read the gate's verdict from the nox session summary.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 2.0.0 -> 2.1.0
Rationale: Sharpen the Code Conventions. An `__init__` holds only its docstring and the re-exports, with no logic, no
function, and no `__getattr__`. A module constant is `UPPER_CASE` and lives in the constants block near the top of the
module, never mid-file. A constant is named for what it holds, not a role it plays. A constant two modules define the
same way is one fact and is lifted to the shared-leaves subpackage, while two that share a value but not a meaning stay
apart. MINOR: expanded guidance, nothing removed or redefined.

Modified sections:
  - Naming: added the constant-named-for-what-it-holds rule.
  - Files & Module Structure: added the UPPER_CASE-constants-at-the-top rule.
  - Public Surface: added the no-logic-in-__init__ rule.
  - Don't Repeat Yourself: added the lift-a-shared-constant rule.

---- history ----
Version change: 1.10.0 -> 2.0.0
Rationale: Structural rewrite. The document is reorganized into a repo-agnostic body (Core Principles, Code
Conventions, Toolchain, Workflow, Governance) plus a single Project Profile that instantiates it for this repository,
so the same constitution can be reused and extended elsewhere. Every rule from 1.x is kept. Scattered guidance is
merged: lint suppressions in one place, the trailing-underscore rule in one place, surface parity and credential
handling in one place. Project-specific facts move into the Project Profile. A Writing Style section is added, and the
whole document is reformatted to obey it: every line is at most 120 characters, no sentence uses a semicolon or a dash
as punctuation, and the prose is plain English. MAJOR: principles are reorganized and regeneralized, none removed.

Modified sections:
  - I. scikit-learn-Compatible API becomes I. Honor the Ecosystem Contract, with the concrete contract in the profile.
  - Surface parity and the agent-is-a-client rule move into a new VI. A Library, Not an Application.
  - Principle VI (naming, docstrings, structure) becomes the Code Conventions section, one subsection per concern.
  - Added Writing Style, which the document itself now follows.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.
-->

# Engineering Constitution

## Purpose & Scope

This constitution states the engineering principles and code conventions that govern a typed Python library. The body
is repo-agnostic, so it can be reused across projects. A single Project Profile at the end instantiates it for the
repository that adopts it, naming the concrete framework contract, toolchain, dependencies, and delivery surfaces. A
project extends the constitution by growing its Project Profile, and by amending the body when a new rule is genuinely
general.

Rules use MUST and MUST NOT for what the gate and review enforce. They use plain statements for the taste an automated
gate cannot check. Both are binding.

## Writing Style

This constitution, and the prose documents in the repository, follow one style so they read consistently and diff
cleanly. The rule below is the same one the Docstrings conventions place on code.

- A line is at most 120 characters, the same limit as the code.
- A sentence uses no semicolon and no dash as punctuation. Each point is its own sentence. Hyphenated words such as
  trailing-underscore and repo-agnostic are fine.
- The English is plain and correct: simple direct words, short sentences, nothing clever or roundabout.

## Core Principles

### I. Honor the Ecosystem Contract

Public objects MUST conform to the contract of the framework they plug into, rather than inventing a parallel
convention that silently breaks downstream code. Constructor parameters are stored unmodified under their own names.
State learned at runtime is exposed only through the framework's convention for derived state. Behavior is configured
through explicit parameters, never through hidden global or ambient state, so an object is deterministic and testable in
isolation. The concrete contract a project conforms to is named in its Project Profile.

Rationale: interoperability with an established ecosystem is a library's core value. Drift from its contract breaks
users' code in ways the tests here cannot see.

### II. Type Safety & Schema Validation

All code, public and internal, MUST carry complete type annotations and pass the static type checker with no new ignored
errors. Data that crosses a public boundary MUST be validated against an explicit, declared schema. Data-shape
assumptions MUST be declared as schemas, not enforced by ad-hoc runtime checks scattered through the code.

Rationale: a typed library that operates on structured data has contracts that are easy to break silently. Static types
and boundary schemas catch integration errors before they reach users.

### III. Tests & Doctest Discipline

Every behavioral change MUST ship with tests. The suite runs with branch coverage, randomized ordering, and executable
docstrings, so every code example in a docstring MUST be correct and runnable. New logic MUST NOT reduce the coverage of
the module it touches. A bug fix MUST include a regression test that fails before the fix.

Rationale: randomized, doctest-inclusive testing keeps examples honest and guards against order-dependent flakiness
where subtle regressions are otherwise invisible.

### IV. Automated Quality Gates (NON-NEGOTIABLE)

Code MUST pass the full automated gate before merge: formatting, linting, static type checking, docstring coverage, a
security scan, and a dependency audit. The gate runs locally through pre-commit and again in CI, and a red run blocks
merge. Failures MUST be fixed at the source. Disabling a rule is the exception governed by Comments & Suppressions, not
a workaround.

Rationale: one machine-enforced bar removes style debate, keeps diffs reviewable, and stops security-sensitive
dependencies from silently degrading.

### V. Documentation as a First-Class Artifact

Every public module, class, and function MUST have a docstring in the project's chosen style. A user-facing behavioral
change MUST update the affected documentation, and MUST add or amend a changelog entry when it changes public behavior.
Runnable examples are part of the documented contract and MUST stay working.

Rationale: a library is adopted through its documentation. An undocumented capability does not exist for users, and it
rots without executable coverage.

### VI. A Library, Not an Application

The package stays a library. Application concerns are the caller's, not the library's: an interactive loop, a long-lived
session, a choice of model or policy, a credential. Keeping them out is what makes the library deterministic and
testable. A capability that needs a credential or performs a real-world side effect MUST live behind an optional extra,
never in the default install.

A project may expose the same capabilities through several delivery surfaces, for example a Python API, a command line,
and a server. The surfaces MUST expose the same underlying capabilities, with none holding logic the others cannot
reach. A parity test MUST assert this, so the surfaces cannot drift.

Rationale: a library free of application state stays composable and testable, and surface parity keeps a capability from
existing on one surface only.

## Code Conventions

These are binding, not advisory. They are generic Python and hold in any project. The automated gate checks the
mechanical parts. These conventions are the taste it cannot check.

### Naming

A function name begins with a verb and names what the function actually does or returns. Write `count_common_prefix`,
not `common_prefix_length`. Write `normalize_identity`, not `transform_identity`. Write `build_roster`, not `roster`. A
name that begins with a noun describes a value, and a function is not a value. The verb is honest: a function that loads
or resolves an object from a reference is `load_` or `resolve_`, not `build_`. Avoid empty verbs that say nothing, such
as `process`, `handle`, `manage`, or `transform` with no object.

A method is an action and begins with a verb. A property names a value and is a noun phrase, never verb-first. State an
instance derives at runtime carries the framework's derived-state marker, which in this ecosystem is a trailing
underscore. This holds whether the state is a stored attribute or a computed property. A public instance name is
therefore one of two things. It is a constructor parameter, stored unmodified under its own name with no marker. Or it
is a derivation, carrying the marker. Anything else an instance exposes is private. A class-level constant that declares
what a class is, such as its kind or its name, is a `ClassVar` and stands apart from this.

A module is named for the concern it owns. Use a descriptive noun for what it does or holds, such as `_resolver`,
`_schedule`, or `_factory`. Do not name it for the data it consumes, and do not name it for the surface that happens to
call it. It MUST NOT reuse a name that collides with a dependency's concept. Names come from the domain, used
consistently.

A constant is named for what it holds, not for a role it happens to play. When the values are the preplay statuses the
name is `PREPLAY_EVENT_STATUSES`, not `INPUT_EVENT_STATUSES`. Read every constant name and check it still describes its
value.

### Module Structure

A module reads top to bottom in one order. First a one-line imperative module docstring, then the license header, then
the imports grouped standard library, third party, first party. The linter sorts the imports, so do not sort them by
hand. A module adds `from __future__ import annotations` above the imports only when it needs it, for a forward
reference or a name that exists only under `TYPE_CHECKING`. The supported language floor decides, and a version that
resolves the annotations without it does not carry it. Then module constants and type aliases. Then functions in
dependency order, so a name is defined before it is used, the small helpers first and the function the module exists
for last.

A module constant is `UPPER_CASE`, and it lives in the constants block near the top of the module, never mid-file among
the functions.

One module is one concern. When a file grows two, split it. Small general helpers may share a `_utils` module, whose one
concern is the assorted helpers a package needs, and a helper earns its own module only when it takes on a distinct role
worth a name, as `_base` or `_types` do, not for every function. A definition lives in the module that owns it. A base
module is self-contained and imports no sibling. Its purpose is to be imported, not to import, so a base that needs a
sibling's code absorbs it by merging rather than importing. A type-only alias under `TYPE_CHECKING` is not a sibling.

### Package Layering

The top level of a package holds subpackages and its `__init__`, not loose implementation modules. The shared leaves the
whole tree imports, the type vocabulary, the shared constants, and the shared building primitives, live in a `core`
subpackage, and the rest of the tree imports them from there. A builder lives in the package that owns what it builds,
and is re-exported from there. This keeps every import running downward, from `core` to domain packages to surfaces, so
no cycle can form. No import inside a function body papers over a cycle. Fix the cycle. The only lazy import defers an
optional dependency and carries a suppression with a reason. Prefer this layering over a suppression and a comment that
paper over an out-of-order import.

### Public Surface

Implementation modules, classes, and helpers are private, named `_name`. The package `__init__` re-exports the public
surface with an explicit `__all__`, and carries only its docstring and those re-exports, with no license header, since
it holds no implementation of its own. A public name used outside the module that defines it is re-exported through its
owning package's `__init__` and imported from that surface, never from the private module that defines it. It is
re-exported once, where it lives, and a parent package does not re-export a subpackage's surface a second time. This
holds for all code, production and tests alike. Reaching into another package's private module for a public name is the
smell the re-export removes.

An `__init__` holds only its docstring and the re-exports. It carries no logic, no function, and no `__getattr__`. A
name that has to be computed to be exposed lives in a module, not the `__init__`.

### Docstrings

The summary line is one line, imperative, and says what the thing does. It is never `Implements the ...`, `This function
...`, or `A class that ...`, which are meta narration. Write it in plain English. Prefer simple, direct words over
clever or roundabout phrasing, and if a line reads awkwardly out loud, rewrite it.

For most functions the one line is the whole docstring, and a private helper never gets more. A public entry point and a
public class carry an `Args` and `Returns` block, and a `Raises` block where they raise, that documents every parameter
and what is returned. A constructor parameter or a dataclass field is documented under `Args`. An `Attributes` block is
only for learned state.

A docstring describes what the thing is and what it holds, plainly. It does not state its virtues, such as `free` or
`needs no key`. It does not give the rationale for its shape, the `since ...` or `so ...` clause. It does not say what
downstream code builds from it. State the content, not the sales pitch, the justification, or the uses. Never describe
what the code does not do. Never restate a self-evident name, so a `url` field needs no "where to read it from". No
essays, and no editorializing. A docstring joins no two clauses with a semicolon or a dash. Each point is its own
sentence.

Public API carries a runnable example checked by the doctest run. A network-touching class does not. The top-level
package `__init__` is the exception to the one-line rule. As the library's front page it may carry a fuller docstring, a
tagline and a short overview of the submodules.

### Comments & Suppressions

Almost no comments. The names say what, the docstring says why. An inline comment that explains the next line means the
line or its names are unclear, so fix those. The only comments in source are the license header and, rarely, a
suppression.

A lint or type suppression, a `# noqa` or a `# type: ignore`, is a last resort for a genuine one-off, and it carries the
rule code and the reason. When the same suppression recurs across the repository, it is not repeated inline. The rule is
configured once in the project configuration, as a scoped ignore, so the decision lives in one place rather than
scattered through the source.

### Errors

Build the message in a variable, then raise it. Raise a specific named exception defined for the module or package,
not a bare `Exception` or `ValueError` where a named one carries meaning. The message tells the reader what to do, the
variable that was missing, or the value that did not match. Do not catch and swallow. Catch narrowly, or let it
propagate.

### Control Flow

Functions are small and do one thing. A function that needs a paragraph of docstring body to explain its branches is two
functions. Return early with guard clauses. Avoid deep nesting, and extract a helper before the third level of
indentation.

### Don't Repeat Yourself

A fact, a definition, or a derivation lives in exactly one place. When the same thing is expressed twice, collapse it
to one. Do not store what can be derived from what you already keep. Persist the source and derive the projection on
demand, not both. Do not reimplement a capability the codebase already has. Reuse it rather than writing a second copy
in another module.

A constant that two modules define the same way is one fact, whatever each names it. Lift it to the subpackage that
holds the shared leaves and import it from there. Two constants that share a value but not a meaning, such as a
column-name separator and an item-key separator that are both `'__'`, are two facts and stay apart.

The same holds for behavior. Two surfaces that serialize an object into different shapes, each a contract its callers
depend on, are two facts, not one duplication. Collapse only the fragment that is identical at every call site. When
call sites differ in what they check or emit, a helper that unifies them changes behavior, so leave them apart.

### Tests

The test tree mirrors the source tree. A test is named `test_<function>_<behavior>`. It begins with the function it
exercises, then a terse behavior phrase with articles dropped, and its docstring is a single line. A test never imports
a private name from a private module. It uses the public API the way a user does, and if it needs an internal, the
internal wants to be public. No test reaches the network. Use a recorded payload, a fake, or a locally served page.
Fixtures are typed, small, and live in the nearest `conftest.py`.

### Credentials

A credential is named, never passed. A function, command flag, or tool argument takes the name of the variable holding
the secret, and reads it where it is used. A secret never becomes an argument value, a log line, or a pickle.

## Toolchain & Standards

- Language: the project declares its supported language versions and MUST remain compatible across all of them.
- Dependencies: a new runtime dependency MUST be justified and declared in the project manifest, not vendored ad hoc. A
  credentialled or side-effecting capability lives behind an optional extra, per Principle VI.
- Build and packaging: a `src`-based layout with SCM-derived versioning. Generated version metadata is not hand-edited.
- Task automation: canonical task-runner sessions for tests, checks, formatting, docs, and release are the entry points
  the gate runs through.
- Style constants: line length, docstring convention, and formatter options are set once in the project configuration
  and never overridden by hand.

The concrete versions, tools, and dependency list are in the Project Profile.

## Development Workflow & Quality Gates

- Work happens on feature branches. The release branch MUST stay green.
- Before opening a PR, contributors MUST run the full gate in order, formatting, then checks, then the documentation
  build, then tests, resolve all findings, and conform to the Code Conventions. The documentation build runs the
  examples, so it is part of the gate and is never skipped.
- CI re-runs the same gates. A red CI run blocks merge.
- Every PR MUST state which principles it touches and confirm the gates pass. Reviewers verify compliance, not just
  correctness.
- Releases follow semantic versioning and MUST update the changelog before tagging.

## Governance

This constitution supersedes ad-hoc conventions and prior undocumented practice. It applies to all code, documentation,
and tooling changes in the repository that adopts it.

- Amendments: proposed through a PR that edits this file, states the rationale, and updates the version and Sync Impact
  Report. An amendment that adds or removes a principle or governance rule requires the maintainer's approval.
- Versioning policy: semantic versioning of the constitution itself. MAJOR is a backward-incompatible removal or
  redefinition of a principle or governance rule, or a structural rewrite. MINOR is a new principle or section, or
  materially expanded guidance. PATCH is a clarification or a non-semantic wording fix.
- Compliance review: every PR and code review MUST verify adherence to the Core Principles and Code Conventions. A
  deviation MUST be justified in the PR, and recorded in the plan where it adds complexity. Unjustified violations block
  merge.
- Runtime guidance: contributor-facing operational guidance lives alongside the code, in a `CONTRIBUTING.md` and
  developer docs, and MUST stay consistent with this constitution.

## Project Profile: sports-betting

This section instantiates the body above for this repository. It is the only repo-specific part. The rest is portable.

- Ecosystem contract, Principle I: the scikit-learn estimator contract. The public estimators, the dataloaders and
  bettors, store constructor parameters unmodified, learn state into trailing-underscore attributes, and keep the
  standard surface of `fit`, `predict`, `predict_proba`, `bet`, `score`, `get_params`, and `set_params`, so they compose
  with pipelines, model selection, and cross-validation. Sources implement the source-plugin contract of
  `list_index_items`, `read_catalogue`, `list_required_items`, `list_fixtures_items`, `to_snapshots`, and `request_url`.
- A library, not an application, Principle VI: no model, model key, model choice, or agent loop enters the package. An
  agent is a client of the surfaces.
- Delivery surfaces, Principle VI: the Python API, the CLI (`sportsbet`), and the MCP server (`sportsbet-mcp`). A parity
  test asserts they stay in step.
- Schema validation, Principle II: every DataFrame that crosses a public boundary, the training data, the fixtures, and
  the odds, is validated against a `pandera` schema.
- Language: Python `>=3.11, <3.14`, targeting `py311`.
- Core dependencies: `scikit-learn`, `pandas`, `pandera`, `click`, `rich`, and `aiohttp`. The optional `mcp` extra ships
  the `sportsbet-mcp` server.
- Build and tooling: PDM with SCM-derived versioning and a `src` layout. The `nox` sessions are `tests`, `checks`,
  `formatting`, `docs`, `changelog`, and `release`, with `pdm run` shortcuts. The gate runs in order, `pdm run
  formatting`, `pdm run checks`, `pdm run docs build`, then `pdm run tests`, covering `black`, `docformatter`, `ruff`
  at line length 120, `interrogate`, `bandit`, `pip-audit`, the executed documentation examples, `pytest` with
  `--doctest-modules`, and `mypy`. The docs build executes every gallery example, so a broken example fails the gate.
- Reading the gate: its verdict is the `nox` session summary line, such as `Session tests-3.13 was successful`. An exit
  code read from a piped command reports the last stage of the pipe, not the run, so it can read green over a red run.
  A reused `nox` environment can carry a stale toolchain, so a clean run can find a lint rule a cached one missed.
- Style: line length 120, Google docstrings, and `black` with skip-string-normalization.
- Package layering: `core`, then the domain packages `sources`, `dataloaders`, `evaluation`, and `execution`, then the
  surfaces `cli` and `mcp`. Builders live with what they build: `build_dataloader`, `build_bettor`, and `build_venue`.
- Named exceptions: `BuildError`, `SelectionError`, `ExecutionError`, and `CredentialError`.

**Version**: 2.3.0 | **Ratified**: 2026-07-08 | **Last Amended**: 2026-07-27
