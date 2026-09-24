# Feature Specification: Plain Documentation

**Feature Branch**: `009-plain-documentation`

**Created**: 2026-07-28

**Status**: Draft

**Input**: User description: rewrite the documentation and examples in plain English, and update the constitution to
require that style and to record the lessons from the 0.15.0 release.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A new user reads the docs and gets going (Priority: P1)

A new user opens the README and the user guide. They want to install the library, get data, fit a model, and place a
bet. The text is plain and direct. Each page has clear headings. The user reads a short section, follows the steps, and
it works. They do not have to reread a sentence to understand it.

**Why this priority**: This is the point of the feature. The docs exist to be understood. If a new user can follow
them without confusion, the feature has delivered its value.

**Independent Test**: Give a new reader the README and the user guide. Ask them to install the library and run the
first example. They finish without asking what a sentence means.

**Acceptance Scenarios**:

1. **Given** the README, **When** a user reads the install section, **Then** they see separate steps for the basic
   install, the MCP extra, the execution extra, and the development install, each under its own heading.
2. **Given** any documentation page, **When** a reader reads a sentence, **Then** it is short, direct, and says who
   does what in the normal order.
3. **Given** a page that mentions a real risk, such as spending real money, **When** a reader reaches it, **Then** the
   risk is stated once, plainly, and the page moves on.

---

### User Story 2 - Every code example runs (Priority: P2)

A user copies a code example from the docs and runs it. It works. No example is a sketch or a fragment that cannot run.
The docs build and the doctest run execute every example, so a broken one fails the build.

**Why this priority**: A code example that does not run teaches the wrong thing and breaks trust. Runnable examples are
what make the docs a reliable reference.

**Independent Test**: Run the documentation build and the doctest run. Every example executes and passes. Copy one
example out of the docs and run it on its own. It works.

**Acceptance Scenarios**:

1. **Given** the gallery examples, **When** the documentation build runs, **Then** every example executes without
   error.
2. **Given** an example that would need a secret or the network, **When** it runs, **Then** it uses a placeholder or a
   fake so it runs offline, or it is removed. No example depends on a real key or a live request.
3. **Given** a code block in a docstring, **When** the doctest run executes it, **Then** it passes.

---

### User Story 3 - The rules keep the docs plain (Priority: P3)

A contributor writes new documentation later. The constitution tells them to write plainly and to make examples run. A
reviewer checks the writing against a short, concrete list of rules. The docs stay clear over time instead of drifting
back to the old style.

**Why this priority**: A one-time rewrite fades without a rule. The constitution is what keeps the standard, and it is
also where the release lessons belong.

**Independent Test**: Read the updated Writing Style rules. They forbid the clever, inverted, defensive style and
require the plain one, in terms a reviewer can check. The release lessons are recorded.

**Acceptance Scenarios**:

1. **Given** the constitution, **When** a reviewer reads the Writing Style rules, **Then** the rules name the plain
   style to use and the clever style to avoid, in checkable terms.
2. **Given** the constitution, **When** a contributor adds an example, **Then** a rule requires it to run, verified by
   the build.
3. **Given** the constitution, **When** someone plans a release, **Then** the recorded lessons cover the docs build in
   the gate, the masking by a cached toolchain or a local secret, the no-secret rule for examples, and branch and
   release hygiene.

---

### Edge Cases

- An existing example cannot be made to run offline, for example because it needs a paid live feed. It is rewritten to
  use sample data or a placeholder, or it is removed. It is not left as a fragment that does not run.
- A page states the same risk in several places. The rewrite states it once, in the right place, and drops the rest.
- A private docstring is not shown in the rendered docs. It is out of scope for this feature and is left as the code
  conventions require.
- A rewrite would change what a public function does or is called. That is out of scope. Only the wording changes.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: All user-facing documentation MUST be rewritten in plain English: short, direct sentences, one idea each,
  said in the normal subject-verb-object order.
- **FR-002**: The documentation MUST NOT use inverted or passive phrasing for effect, idioms, or literary flourish. It
  reads as a person wrote it for another person.
- **FR-003**: The documentation MUST state each real risk once, plainly, without hedging or repeated warnings.
- **FR-004**: Each documentation page MUST use headings and subsections so a reader can scan it. Long blocks are split.
  The install section has separate subsections for the basic install, the MCP extra, the execution extra, and the
  development install.
- **FR-005**: Every code example in the documentation MUST run. The documentation build and the doctest run execute
  them, and a broken example fails.
- **FR-006**: No code example MAY depend on a secret or the network to run. An example that would uses a placeholder or
  a fake so it runs offline, or it is removed.
- **FR-007**: The scope of the rewrite MUST be the README, the user guide pages, the gallery examples, and the
  user-facing docstrings that appear in the rendered docs.
- **FR-008**: The rewrite MUST NOT change library behavior, the public API, or any name. Only the wording of
  documentation and docstrings changes.
- **FR-009**: The constitution's Writing Style rules MUST require the plain, structured, human style and forbid the
  clever, idiomatic, inverted, and defensive style, in terms a reviewer can check without a tool.
- **FR-010**: The constitution MUST require that a code example runs and is verified by execution, with no un-runnable
  demo or pseudo examples.
- **FR-011**: The constitution MUST record the 0.15.0 release lessons: the gate includes the documentation build and it
  runs the examples, a cached toolchain or a locally set secret can hide a failure that a clean run finds, an example
  must not depend on a secret to build, and `development` stays a superset of the released `main` while the version
  tool must not collide with an existing tag.
- **FR-012**: `docs/generated` MUST be regenerated by the build, never hand-edited.

### Key Entities *(include if feature involves data)*

- **Documentation surface**: The set of pages in scope. The README, the user guide pages, the gallery examples, and the
  user-facing docstrings shown in the rendered docs.
- **Style rules**: The list of concrete writing rules in the constitution that a reviewer applies. What to do and what
  to avoid, stated so a person can check them.
- **Runnable example**: A code example that executes in the documentation build or the doctest run. It uses sample data,
  a placeholder, or a fake, never a secret or a live request.
- **Release lessons**: The guidance drawn from the 0.15.0 release, recorded in the constitution.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new reader installs the library and runs the first example by following the docs, without asking what a
  sentence means.
- **SC-002**: 100% of code examples in the documentation run and pass in the documentation build and the doctest run.
- **SC-003**: No code example depends on a secret or the network.
- **SC-004**: Every documentation page in scope has headings, and no page is a single undivided block. The install
  section has the four named subsections.
- **SC-005**: A reviewer can judge any page against the Writing Style rules and reach a clear pass or fail, without a
  tool.
- **SC-006**: The full gate passes in order, formatting then checks then the documentation build then tests, on the
  supported Python versions.

## Assumptions

- In scope is the README, the pages under the user guide, the gallery examples, and the docstrings of the public API
  that appear in the rendered docs. Out of scope is `docs/generated` (regenerated), the changelog (generated), private
  docstrings that are not rendered, and the body of the constitution apart from the Writing Style and example rules.
- The plain style is enforced by human review against the constitution's rules, not by a new automated tool, because a
  formatter cannot judge tone. The rules are written to be concrete enough to check by eye.
- An example is proven to run by the documentation build (which executes the gallery) and the doctest run. An example
  that needs a secret uses a placeholder, one that needs the network uses sample data or a fake, and one that cannot be
  made to run offline is removed.
- The existing sample sources and fixtures are enough to make the examples run offline.
