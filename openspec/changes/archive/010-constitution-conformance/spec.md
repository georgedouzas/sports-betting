# Feature Specification: Constitution conformance

**Feature Branch**: `010-constitution-conformance`

**Created**: 2026-09-20

**Status**: Draft

**Input**: User description: "make project compatible to the constitution"

## Clarifications

### Session 2026-09-20

- Q: Which names count as the public API that must carry a runnable doctest example? → Every public function, class,
  and method, not only the names a package re-exports. A name that touches the network or needs a credential is the
  exception, since the rule itself says a network-touching class carries no runnable example, and such a capability is
  shown as reference code in the guide instead.
- Q: How should a private module obtain a name that exists only as a public name in another package? → It imports it
  from that package's surface, which is what a surface is for. The constitution was amended to say so, in version
  9.0.0. Privacy is about reaching past a surface, not about consuming one, so the rule a module breaks is importing
  another package's private name, and the rule a name breaks is staying public when its own package never re-exports
  it. A package and its subpackages count as one package.
- Q: Are the `TYPE_CHECKING` import guards kept? → No. Four exist in the tree and none prevents a cycle, since the
  package import graph is acyclic. They become plain imports, and version 9.0.0 forbids the guard, on the same ground
  that an import inside a function body papers over a cycle rather than fixing it.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A public name documents what it takes, returns, and raises (Priority: P1)

A contributor opens any public function or class in the package and reads its docstring without the source beside it.
The summary line says in one imperative line what the thing does. Below it, every parameter appears under `Args`, what
comes back appears under `Returns`, and every exception it raises appears under `Raises`. A private helper carries the
summary line and nothing else, so the reader can tell at a glance whether they are looking at a supported entry point
or an internal step.

**Why this priority**: This is the largest gap between the code and the constitution, and it is what a user of the
library actually reads. Of the 189 public functions and classes in the package, 111 carry no `Args`, `Returns`, or
`Raises` block, and 31 functions and classes carry no docstring at all.

**Independent Test**: Pick any package under the source tree and read every public function and class in it. Each one
has a one-line summary and the blocks for what it takes, returns, and raises. Each private one has the summary line
alone. The docstring coverage check and the doctest run both pass.

**Acceptance Scenarios**:

1. **Given** a public function that takes parameters and returns a value, **When** a contributor reads its docstring,
   **Then** every parameter is named under `Args` and the return is described under `Returns`.
2. **Given** a public function that raises a named exception, **When** a contributor reads its docstring, **Then** the
   exception appears under `Raises` with the condition that triggers it.
3. **Given** a private helper, **When** a contributor reads its docstring, **Then** it is a single imperative line with
   no blocks below it.
4. **Given** a public class, **When** a contributor reads its docstring, **Then** its initialization parameters appear
   under `Args` and the state it learns appears under `Attributes`.
5. **Given** a function or class with no docstring today, **When** the work is done, **Then** it has one.

---

### User Story 2 - What a package exports and what it keeps are the same two lists (Priority: P2)

A contributor reads a name and can tell from the name alone whether the package promises it. A name a package
re-exports is public and reachable through the surface. A name it keeps is private and named with an underscore. No
module reaches past another package's surface for a private name, and no import hides behind a type-checking guard.

**Why this priority**: The surface is what the library can be held to. 110 names look public today but no package
re-exports any of them, so the apparent surface is 192 names where the promised one is 82. A reader cannot tell which
is which, and nor can a contributor about to change one.

**Independent Test**: List every name defined in a private module. Each is either re-exported by its package or named
with an underscore, with nothing in between. No module imports another package's private name, and the tree contains
no type-checking import guard.

**Acceptance Scenarios**:

1. **Given** a name defined in a private module that no package re-exports, **When** a contributor reads it, **Then**
   it is named with a leading underscore.
2. **Given** a module that needs a name from another package, **When** a contributor reads its import, **Then** it
   comes from that package's surface and not from a module behind it.
3. **Given** a package `__init__`, **When** a contributor reads it, **Then** it holds only its docstring and the
   re-exports, with an explicit list of what it exports.
4. **Given** a name that is re-exported by a package, **When** a contributor searches the tree, **Then** it is
   re-exported once and no parent package re-exports it a second time.
5. **Given** a module that used a type-checking guard for an import, **When** the work is done, **Then** the import is
   plain and no cycle appears.
6. **Given** a test, **When** a contributor reads its imports, **Then** it imports from the public surface the way a
   user would, and imports no private name.

---

### User Story 3 - The source carries no comment that a name or a docstring should carry (Priority: P3)

A contributor reads a module and finds no inline comment explaining the next line. Where a comment used to explain
something, the name or the docstring now says it. The only comments left are the license header and the rare
suppression, and every suppression names its rule and its reason.

**Why this priority**: It is a smaller, mechanical sweep than the first two, and it depends on them. A comment often
disappears when the name it props up is fixed, so doing it after the naming and docstring pass avoids touching the same
lines twice. There are 75 such comment lines today.

**Independent Test**: Search the source tree for comment lines. What comes back is the license headers, the
suppressions, and nothing else.

**Acceptance Scenarios**:

1. **Given** an inline comment that explains the line below it, **When** the work is done, **Then** the comment is gone
   and the line or its names say what the comment said.
2. **Given** a suppression, **When** a contributor reads it, **Then** it carries the rule code and the reason.
3. **Given** a suppression that appeared in several modules for the same reason, **When** the work is done, **Then**
   the rule is configured once in the project configuration and the inline copies are gone.

---

### User Story 4 - Every example a reader can see has been run by the build (Priority: P4)

A contributor or a user reads an example, in the documentation or in a docstring, and knows it works, because the build
ran it and the output on the page is the output it produced. No example is a fragment, pseudo-code, or a demo that
cannot run.

**Why this priority**: The documentation pages already execute their blocks, so this story is about the docstring
examples on the public surface and about removing any example that cannot run. It is the largest story by volume, since
33 of 187 public names carry an example today, but it is worth less per unit of work than the docstrings and the
boundary, which a reader hits first.

**Independent Test**: Run the documentation build and the doctest run. Both are green, and no example in either place
is skipped, elided, or accompanied by output that was written by hand.

**Acceptance Scenarios**:

1. **Given** a public entry point that can run offline, **When** a contributor reads its docstring, **Then** it carries
   an example the doctest run executes.
2. **Given** a capability that cannot run without a credential or the network, **When** a contributor reads it,
   **Then** it carries no runnable example, and the capability is shown as reference code in the guide instead.
3. **Given** a documentation page, **When** the build runs, **Then** every code block on it is executed and its output
   is the build's output, not a paste.

---

### User Story 5 - The Project Profile describes the repository as it actually is (Priority: P5)

A contributor reads the Project Profile and finds that every concrete fact in it holds: the layers are the ones named,
the builders live with what they build, the exceptions raised are the four named ones, and the package front page
lists the submodules the package actually has.

**Why this priority**: It is the smallest slice and it closes the loop, since a profile that describes something else
makes every rule above it harder to trust. The package front page lists three submodules while the package ships more
than three public surfaces.

**Independent Test**: Read each Project Profile bullet and check it against the tree. Each one is true, and the package
front page names every public surface.

**Acceptance Scenarios**:

1. **Given** the layering bullet, **When** a contributor checks the tree, **Then** the packages and their import
   direction are the ones it names.
2. **Given** the named exceptions bullet, **When** a contributor searches the source, **Then** the exceptions raised
   are those four and no bare exception stands where a named one carries meaning.
3. **Given** the front page docstring, **When** a user reads it, **Then** it names every public surface the package
   ships.

---

### Edge Cases

- What happens to a method whose signature the scikit-learn contract fixes, such as `fit` or `predict`? It is public,
  so it carries the blocks, and its parameters are documented under the names the contract fixes rather than renamed.
- What happens to a dunder module such as the package entry point, which is named like a private module but is a
  public entry point? The import privacy rule has to say which side it falls on.
- What happens to a constructor, a property, or an inherited method that overrides a documented base? The rule has to
  say whether each carries its own blocks or inherits the base's docstring.
- What happens when a private module genuinely needs a name that only exists as a public name in another package? Two
  readings exist and User Story 2 cannot be finished until one is chosen.
- What happens to an example that needs a credential or the network? It is not a runnable example, so it moves to the
  guide as reference code or uses sample data.
- What happens to a public name that is part of the surface but is never meant to be called directly by a user, such
  as an abstract base? It still carries the blocks, since the rule turns on privacy, not on how often it is called.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Every module, class, and function in the source tree MUST carry a docstring.
- **FR-002**: Every public function and public class MUST document every parameter under `Args`, what it returns under
  `Returns`, and every exception it raises under `Raises`, omitting any block that would stand empty.
- **FR-003**: Every private function and private class MUST carry the one-line summary alone and MUST NOT carry those
  blocks.
- **FR-004**: Every docstring summary line MUST be one imperative line that says what the thing does, with no meta
  narration, no statement of virtues, and no rationale for its shape.
- **FR-005**: A class docstring MUST document its initialization parameters under `Args` and the state it learns under
  `Attributes`.
- **FR-006**: No module MUST import another package's private name, a package and its subpackages counting as one
  package.
- **FR-007**: Every name defined in a private module that its own package does not re-export MUST be renamed to carry
  a leading underscore, and every reference to it MUST move with it.
- **FR-022**: No module MUST guard an import with a type-checking condition, and every such guard MUST become a plain
  import without introducing a cycle.
- **FR-008**: Every package `__init__` MUST hold only its docstring and its re-exports, with an explicit export list,
  no logic, and no license header.
- **FR-009**: A public name MUST be re-exported once, by the package that owns it, and MUST NOT be re-exported a
  second time by a parent package.
- **FR-010**: Every test MUST import from the public surface and MUST NOT import a private name from a private module.
- **FR-011**: The source MUST carry no inline comment that explains the line below it, and the only comments left MUST
  be the license headers and the suppressions.
- **FR-012**: Every suppression MUST carry its rule code and its reason, and a suppression that recurs across the
  repository MUST be configured once in the project configuration instead of inline.
- **FR-013**: Every public entry point that can run offline MUST carry an example the doctest run executes.
- **FR-014**: A capability that cannot run without a credential or the network MUST NOT carry a runnable example, and
  MUST be shown as reference code in the guide instead.
- **FR-015**: Every code block in the documentation MUST be executed by the documentation build, and MUST NOT be
  accompanied by output written by hand.
- **FR-016**: Every fact in the Project Profile MUST hold against the repository, including the layers, the builders,
  and the named exceptions.
- **FR-017**: The package front page docstring MUST name every public surface the package ships.
- **FR-018**: The full gate MUST pass at the end of each user story, and no story MUST leave the release branch red.
- **FR-019**: The work MUST NOT change what any public name does, returns, or accepts. A rename, where one is needed,
  MUST update every caller, the re-export, the export list, the tests, and the documentation in the same commit.
- **FR-020**: Every public function, class, and method that can run offline MUST carry a runnable example checked by
  the doctest run, whether or not its package re-exports it.
- **FR-021**: A module that needs a public name from another package MUST import it from that package's surface, and
  MUST NOT reach past that surface for it.

### Key Entities

- **Public name**: a module, class, or function whose name does not begin with an underscore. It carries the full
  docstring, is re-exported by its owning package, and may be imported only by other public modules.
- **Private name**: a module, class, or function whose name begins with an underscore. It carries the summary line
  alone and may be imported only by other private modules.
- **Package surface**: the `__init__` of a package, holding its docstring and its re-exports, and the only place a
  public name leaves a private module.
- **Example**: a code block in a docstring or a documentation page that the build executes, whose printed output is
  the output the build produced.
- **Suppression**: an inline exemption from a lint or type rule, carrying the rule code and the reason, or a scoped
  entry in the project configuration where the same exemption recurs.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of public functions and classes document their parameters, their return, and the exceptions they
  raise, up from 41% today.
- **SC-002**: 100% of functions and classes carry a docstring, up from 91% today.
- **SC-003**: 100% of private functions and classes carry the summary line alone, which holds today and must still
  hold at the end.
- **SC-004**: Zero names defined in a private module are public without their package re-exporting them, down from
  110 today, so the apparent surface of 192 names becomes the promised surface of 82.
- **SC-011**: Zero modules import another package's private name, which holds today and must still hold at the end,
  and zero type-checking import guards remain, down from four today.
- **SC-005**: Zero comment lines remain beyond the license headers and the suppressions, down from 75 today.
- **SC-006**: Every example a reader can see has been executed by the build, with zero skipped, elided, or
  hand-written outputs.
- **SC-010**: 100% of public functions, classes, and methods that can run offline carry a runnable example, up from 18%
  today, and every one that cannot run offline carries none.
- **SC-007**: Every concrete statement in the Project Profile can be checked against the repository and holds.
- **SC-008**: The full gate passes at the end of every user story, and the public behaviour of the library is
  unchanged, which the existing test suite confirms without amendment.
- **SC-009**: A reviewer can check any single module against the rules in under five minutes, using only the rules and
  the module.

## Assumptions

- The constitution as of version 8.0.3 is the target. A later amendment changes the target, and this specification is
  read against the version current when the work is done.
- Conformance means the code obeys the rules. It does not mean the rules change to fit the code. Where a rule turns
  out to be wrong, that is an amendment, proposed separately, not a change made inside this work.
- No public behaviour changes. This is a documentation, naming, and boundary sweep, so the existing tests pass
  unchanged, and a test that has to change is a signal that behaviour moved.
- A constructor documents its parameters under the class `Args` rather than carrying a second docstring of its own, and
  a property carries the one-line summary, since it names a value rather than an action.
- A method that overrides a documented base carries its own docstring, since a reader in an editor sees the override
  rather than the base.
- The scikit-learn estimator contract wins over the naming rules where the two disagree, as it already does in the
  Project Profile, so the fixed method and parameter names are documented rather than renamed.
- Third-party imports are outside the privacy rule, which is about names inside the package.
- The constitution was amended to version 9.0.0 during this specification, which is what the import requirements above
  refer to. The amendment was the answer to a question this work raised, which is the intended order: a rule that does
  not fit is amended in the open, not worked around in a specification.
- Renaming an internal name to carry an underscore is not a public API change, since no package re-exports it and no
  user can reach it through a surface.
- The counts quoted in this specification were measured against the tree at the time of writing and are a baseline,
  not a target in themselves.
- The work is split so that each user story leaves the tree green, which lets any prefix of the stories ship on its
  own.
- This builds on the earlier conventions sweep, which conformed the tree to the naming and module rules as they stood
  then. Those rules still hold, so this work is the delta the amendments since have introduced, not a repeat.
