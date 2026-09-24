# Feature Specification: Conventions conformance

**Feature Branch**: `007-conventions-conformance`

**Created**: 2026-07-20

**Status**: Draft

**Input**: Conform the whole codebase to Principle VI (`CONVENTIONS.md`): naming, docstrings, module structure, and test layout.

## Clarifications

### Session 2026-07-20

- Q: What is an allowed public-API rename versus a forbidden behaviour change? → A rename changes only the identifier of a public symbol and updates every caller, the `__init__` export and `__all__`, the tests, and the docs in the same commit. Inputs, outputs, side effects, and semantics stay identical. Anything that changes what a symbol does, returns, or accepts is a behaviour change and is out of scope.
- Q: How is a module docstring that says "Implements the X that ..." or "It provides ..." rewritten? → By stating, in one imperative line, what the module does, which is the same thing the old sentence described. The module's content does not change; only the sentence is reworded. `"""Implements the base classes of the data sources."""` becomes `"""Read the raw content a data source needs, and define the base a source implements."""`.
- Q: Do the scikit-learn estimator names conform to Principle VI, given Principle I fixes some of them? → Principle I wins on the estimator contract. The names the scikit-learn contract fixes — `fit`, `predict`, `predict_proba`, `bet`, `score`, `get_params`, `set_params`, the trailing-underscore fitted attributes, and the constructor parameter names — are NOT renamed, even where Principle VI's verb-first rule would otherwise apply. Every other name in `evaluation` and `dataloaders` that is not part of that contract conforms.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A contributor reads a module and finds it conforms (Priority: P1)

A contributor opens any module under `src/sportsbet` and reads it top to bottom. The module docstring is one imperative line. Functions begin with a verb and say what they do. A function docstring is a single line unless the function is a public entry point or a public class whose shape is not obvious. Helpers come first and the entry point last. There is no `Implements the ...` or `It provides ...`, no essay docstring, no explanatory inline comment, and no `market_outcomes`-style noun-first helper.

**Why this priority**: This is the point of the feature. The constitution now gates on Principle VI, so the code has to meet it. Reading conformance is the smallest slice that delivers the value, and every package builds on the same rules.

**Independent Test**: Pick a refactored package and check each module against the Principle VI checklist. Every public name is verb-first and exact, every docstring is imperative and sized to the rule, every base module imports no sibling, and `pdm run checks` is green.

**Acceptance Scenarios**:

1. **Given** a module whose docstring said `"""Implements the ..."""`, **When** the package is done, **Then** the docstring is one imperative line describing what the module does and the module's behaviour is unchanged.
2. **Given** a public function whose name was a noun or an empty verb, **When** the package is done, **Then** it is renamed to a verb phrase that says what it does, and every caller, the `__init__` export, `__all__`, the tests, and the docs use the new name.
3. **Given** a private helper with an Args/Returns block, **When** the package is done, **Then** its docstring is a single line.
4. **Given** a base module that imported a sibling, **When** the package is done, **Then** it imports no sibling and the shared definition lives with the base.

### User Story 2 - The tests mirror the source and name what they test (Priority: P2)

A contributor looks for the tests of a module and finds them at the mirrored path, named `test_<function>_<behavior>`, each with a single-line docstring, each using the public API, none touching the network.

**Why this priority**: The test layout is half of Principle VI and the thing that keeps the conformance from rotting, but it is only meaningful once the source it mirrors conforms.

**Independent Test**: For a refactored package, every source module has a test module at the mirrored path, every test name starts with the function under test, every test docstring is one line, and no test imports a private name from a private module.

**Acceptance Scenarios**:

1. **Given** a source module `sources/_utils.py`, **When** the package is done, **Then** its tests are at `tests/sources/test_utils.py` and named `test_<function>_<behavior>`.
2. **Given** a test that imported a private helper from a private module, **When** the package is done, **Then** it imports the helper from the package's public API, or the helper is made public because it is worth testing directly.
3. **Given** a test with a multi-line docstring, **When** the package is done, **Then** the docstring is a single line and the full sentence is not lost.

### User Story 3 - The public API and behaviour are unchanged (Priority: P1)

A user who depends on the library upgrades across this work and their code still runs. The public surface is the same, except for names that genuinely violated Principle VI, whose renames are listed in the changelog with a before and after.

**Why this priority**: A conventions refactor that changes behaviour is a bug. This is the guardrail that makes the whole feature safe to ship, so it is as critical as the conformance itself.

**Independent Test**: The full gate stays green at every package boundary. A public rename appears in `CHANGELOG.md` with before and after, and nothing else in the public API changed.

**Acceptance Scenarios**:

1. **Given** the suite passing on the green baseline, **When** a package is refactored, **Then** the suite still passes on 3.11, 3.12 and 3.13 with no test asserting different behaviour.
2. **Given** a public symbol is renamed, **When** the package is done, **Then** the changelog records the old and new name and every reference in the codebase and docs uses the new one.
3. **Given** a runnable example in `docs/examples`, **When** a public name it uses is renamed, **Then** the example uses the new name and still runs.

### Edge Cases

- A name the scikit-learn contract fixes (`fit`, `predict`, `predict_proba`, `bet`, `score`, a trailing-underscore attribute) looks noun-ish or non-verb. It is left alone: Principle I wins.
- A rename would collide with an existing public name. The rename is reconsidered rather than forced, and the collision is called out.
- A base module needs a sibling's code and merging would mix two genuinely separate concerns. The conflict is surfaced and the merge is not forced blindly.
- A docstring body carries a load-bearing why that a single line would lose. The why moves to the module docstring or is dropped only if it restated the code.
- A public helper is tested but not currently exported. It is exported (added to `__all__`) so the test uses the public API, following the resolver-helper precedent.

## Requirements *(mandatory)*

### Functional Requirements

#### Naming

- **FR-001**: Every public function name MUST begin with a verb and say exactly what the function does or returns. A noun-first name (`common_prefix_length`) or an empty verb (`process`, `handle`, `transform` with no object) MUST be renamed.
- **FR-002**: State learned during fitting MUST end in a trailing underscore. Implementation modules, classes and helpers MUST be private (`_name`), and the package `__init__` MUST re-export the public surface through an explicit `__all__`.
- **FR-003**: A rename MUST update, in the same commit, every caller, the `__init__` export and `__all__`, the tests, the docstrings and doctests, and the docs.
- **FR-004**: The names the scikit-learn contract fixes MUST NOT be renamed (see Clarifications). Principle I wins over Principle VI on the estimator contract surface.

#### Docstrings

- **FR-005**: Every module docstring MUST be one imperative line that says what the module does. `Implements the ...`, `It provides ...`, `Module that contains ...` and other meta narration MUST be reworded to the imperative without changing what the module is.
- **FR-006**: A function docstring MUST be a single line, except a public entry point or a public class whose shape is not obvious, which MAY carry an Args/Returns block or a short body. A private helper MUST NOT carry more than a line.
- **FR-007**: No docstring MUST describe what the code does not do, restate the code, editorialize, or run to an essay.

#### Module structure

- **FR-008**: A module MUST be one concern and read bottom-up: a name is defined before it is used, so helpers come first and the entry point last.
- **FR-009**: A definition MUST live in the module that owns it. A base module MUST import no sibling module; a base that needs a sibling's code MUST absorb it by merging rather than importing.
- **FR-010**: No import MUST appear inside a function body except to defer an optional extra, which carries a `# noqa: PLC0415` and a reason.
- **FR-011**: An error MUST raise a named exception with the message in a variable. Functions MUST be small with early returns and no explanatory inline comments.

#### Tests

- **FR-012**: The test tree MUST mirror the source tree: a source module has a test module at the mirrored path, and a source subpackage has a test subpackage with an `__init__`.
- **FR-013**: A test MUST be named `test_<function under test>_<behavior>` and carry a single-line docstring.
- **FR-014**: A test MUST use the public API and MUST NOT import a private name from a private module. A helper worth testing directly MUST be exported for that purpose.
- **FR-015**: No test MUST touch the network.

#### Process and safety

- **FR-016**: The work MUST be behaviour-preserving. Public behaviour and the public API stay the same except for the Principle VI renames, which are recorded in `CHANGELOG.md` with before and after.
- **FR-017**: The work MUST proceed one package at a time. Each package is its own commit and is done only when `pdm run formatting`, `pdm run checks` and `pdm run tests` are green on 3.11, 3.12 and 3.13.
- **FR-018**: No new runtime dependency MUST be added and the extras MUST install exactly what they install today.
- **FR-019**: Runnable examples under `docs/examples` and the user guide MUST follow any public rename and keep working. `docs/generated` MUST be regenerated, never hand-edited.

### Key Entities

- **Package**: A directory under `src/sportsbet` (`sources`, `dataloaders`, `evaluation`, `execution`, `cli`, `mcp`) plus the top-level modules (`_params`, `_selection`, `_artifacts`, `__init__`). The unit of work and of review.
- **Public symbol**: A name re-exported from a package `__init__` and listed in `__all__`. The thing a rename must propagate across.
- **Rename**: A change of a public symbol's identifier with its behaviour held constant, recorded in the changelog.
- **The gate**: `pdm run formatting`, `pdm run checks`, `pdm run tests` on 3.11, 3.12, 3.13. The definition of done for a package.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every module under `src/sportsbet` has a one-line imperative module docstring, with zero occurrences of `Implements the`, `It provides`, or `Module that contains` outside history.
- **SC-002**: Every public function name begins with a verb, except the scikit-learn contract names, which are unchanged.
- **SC-003**: Every private helper and every ordinary function has a single-line docstring; Args/Returns blocks appear only on public entry points and public classes.
- **SC-004**: No base module imports a sibling module, and no function body contains an import except a deferred optional extra.
- **SC-005**: Every source module has a test module at the mirrored path, and every test is named `test_<function>_<behavior>` with a single-line docstring.
- **SC-006**: The full gate is green on 3.11, 3.12 and 3.13 at every package commit.
- **SC-007**: The public API is unchanged except for the renames listed in `CHANGELOG.md`, each with a before and after, and every caller, test, example and doc uses the new name.
- **SC-008**: 100% of runnable examples still run after the sweep.

## Assumptions

- The green baseline (399 tests, full gate green on three versions) is the starting point, and every package commit returns to it.
- `CONVENTIONS.md` is the authoritative detail and Principle VI is its constitutional summary; where they could be read differently, `CONVENTIONS.md` governs.
- The scikit-learn contract surface is: `fit`, `predict`, `predict_proba`, `bet`, `score`, `get_params`, `set_params`, trailing-underscore fitted attributes, and constructor parameter names. These are fixed by Principle I.
- A public helper that a test exercises directly is worth exporting, so tests use the public API rather than reaching into a private module.
- `docs/generated` is produced by the docs build and is out of scope for hand edits; only `docs/examples`, `docs/overview`, `docs/practice` and the like are edited.
- Making the whole codebase conform is large, so it is sequenced by package rather than attempted in one sweep, and the order runs from the most foundational package to the surfaces that depend on it.
