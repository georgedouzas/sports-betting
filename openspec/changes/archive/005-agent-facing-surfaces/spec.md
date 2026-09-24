# Feature Specification: Surfaces that reach the whole library

**Feature Branch**: `005-agent-facing-surfaces`

**Created**: 2026-07-12

**Status**: Draft

**Input**: User description: "do you think having a gui makes sense or remove it all together and replace it with a CLI that uses an agent like claude?"

## Summary

The library promises three ways in — code, a command line, and a graphical app — and promises they reach the same
capabilities. Two of the three no longer do.

While the data layer grew injectable sources, a second sport, three leagues and paid odds, the command line and the
graphical app stayed where they were. **Both can reach exactly one configuration: soccer, with the one free feed.** A
user who wants basketball, or the NBA, or a bookmaker's real prices, has to write Python. That is not a missing feature
in either surface — it is a single sentence in the way they are configured, and it silently locks out everything added
in the last three releases.

The graphical app is worse than merely behind. It is a quarter of the codebase, **nothing tests it** — the test runner
is explicitly told to skip it — it is frozen against an exact version of a framework it therefore cannot safely be
upgraded off, and it needs a whole second language toolchain to run. It is the largest and least examined thing in the
project, and it delivers the least.

So: **retire the graphical app, fix the configuration, and add a surface an agent can drive.** The result is that every
way into the library reaches all of it, for the first time since the data layer was rebuilt.

An agent is a *consumer* of this library, never a part of it. Nothing here puts a model, a key, or a call to one inside
the package.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The command line reaches the whole library (Priority: P1)

A user points the command line at any sport, any league, and any data source — including one that needs their own paid
credential — and it works, exactly as it would from Python.

**Why this priority**: This is the actual defect. Everything else in this feature follows from it. Today the command
line cannot express "use these sources", so it cannot reach basketball at all, and cannot buy odds for anything.

**Independent Test**: Configure the command line for a basketball league with a paid odds credential, and run the same
commands a soccer user runs today. They work.

**Acceptance Scenarios**:

1. **Given** a configuration naming a basketball league and a credentialled odds source, **When** the user prepares and
   extracts, **Then** they get the same data they would get from Python.
2. **Given** a configuration for soccer's free sources, **When** the user runs the same commands, **Then** it still
   works — the common case does not get harder in order to make the general case possible.
3. **Given** a configuration that has not yet chosen which seasons to select, **When** the user asks what is available,
   **Then** they are told — because you cannot choose before you know what exists.
4. **Given** a credential, **When** it is used, **Then** it comes from the environment and never has to be written into
   a file that could be committed.

---

### User Story 2 - An agent can drive the library (Priority: P1)

A user asks an assistant to find value bets in a league. The assistant discovers what data exists, prices the download
before spending anything, prepares it, trains a model, backtests it, and comes back with the bets — using the library
directly, without the user writing code.

**Why this priority**: It is what replaces the graphical app, and it serves the same user — the one who wants the
library's power without writing Python — while being testable, dependency-light, and reaching *all* of the library
rather than a tenth of it.

**Independent Test**: Drive the library through the agent-facing surface end to end, with no Python written by the
user.

**Acceptance Scenarios**:

1. **Given** an assistant connected to the library, **When** it is asked what data is available, **Then** it can find
   out without downloading anything.
2. **Given** a data source that charges per request, **When** the assistant is about to fetch, **Then** it learns the
   **exact cost first**, and spending never happens as a surprise.
3. **Given** prepared data, **When** the assistant is asked to evaluate a betting model, **Then** it can train,
   backtest and produce value bets.
4. **Given** any capability the command line has, **When** the assistant looks for it, **Then** it is there.

---

### User Story 3 - Nothing is left that cannot be trusted (Priority: P1)

Every surface the project ships is tested, and none of them requires a toolchain from another ecosystem.

**Why this priority**: The project's own rules require the automated gate to pass before anything merges. One surface
has been exempt from that gate for its entire life. Removing it is not a loss of capability — the capability it held is
being replaced by one that is tested — it is the removal of the only part of the project nobody could verify.

**Independent Test**: The test runner has no exclusions, and the package installs and runs without a second language
toolchain.

**Acceptance Scenarios**:

1. **Given** the project's test configuration, **When** it runs, **Then** it excludes nothing.
2. **Given** a fresh install, **When** every surface is exercised, **Then** none of them needs a toolchain outside the
   project's own language.
3. **Given** an existing user of the graphical app, **When** they upgrade, **Then** the removal is announced plainly
   and they are told what replaces it.

---

### Edge Cases

- **A configuration written for the old contract.** It must fail with a message that says what to change, not with an
  obscure error. This is a breaking change and it must behave like one.
- **A credential that is absent.** The user must be told which one is missing, and never have it echoed back.
- **An agent asked to fetch something expensive.** The cost must be surfaced before the spending, not reported after
  it.
- **An agent asked for a league whose odds nobody sells.** It must be told, rather than handed an empty dataset.
- **A user who wanted the graphical app.** They must be told where it went and what to use instead.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The command-line configuration MUST accept a **fully configured dataloader**, so that any sport, league
  and data source — including one requiring a credential — can be expressed.
- **FR-002**: The previous configuration contract MUST be removed rather than deprecated. This is one clean break, not
  two.
- **FR-003**: A configuration that does not yet select any data MUST still be able to ask what data exists.
- **FR-004**: A credential MUST be readable from the environment, so it never needs to be written into a file that
  could be committed.
- **FR-005**: Every existing command-line capability MUST keep working.
- **FR-006**: The library MUST offer a surface an assistant can drive, exposing at least: what data exists, what a
  preparation would cost, preparing it, extracting training data, extracting fixtures, backtesting a model, and
  producing value bets.
- **FR-007**: That surface MUST report the **exact cost** of a metered fetch **before** any spending occurs.
- **FR-008**: That surface MUST be an **optional** addition. A user who does not want it MUST NOT have to install
  anything extra for it.
- **FR-009**: That surface MUST be covered by the project's tests.
- **FR-010**: The graphical app MUST be removed, along with its dependencies, its entry point, and its exemption from
  the test run.
- **FR-011**: The removal MUST be announced as a breaking change, naming what is gone and what replaces it.
- **FR-012**: After this change, **every** shipped surface MUST reach every sport and every data source the library
  supports.
- **FR-013**: The test configuration MUST exclude no part of the shipped package.
- **FR-014**: No model, credential for a model, or call to a model may be added to the library. The assistant is a
  consumer of the library, never a component of it.
- **FR-015**: The data and evaluation layers MUST NOT change. This is a change to how the library is reached, not to
  what it does.

### Key Entities

- **Configuration**: What a user hands the command line. It changes from *a kind of dataloader plus a selection* to *a
  dataloader that has already been built* — which is the only representation that can carry a source, and therefore a
  credential.
- **Surface**: A way into the library. After this change there are three — code, the command line, and the
  assistant-facing one — and all three reach the same capabilities.
- **Cost estimate**: What a metered fetch would spend, known before it is spent. It already exists; this feature makes
  it reachable from every surface, because an assistant must never spend a user's money by surprise.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The command line reaches **every** sport, league and data source the Python API reaches — including every
  one that needs a paid credential. Today it reaches **one of them**.
- **SC-002**: An assistant can go from "what data exists?" to "here are the value bets" **without the user writing any
  code**.
- **SC-003**: A metered fetch is **always** priced before it is paid for, from every surface.
- **SC-004**: The test run **excludes nothing**. Today it skips a quarter of the codebase.
- **SC-005**: Installing and using the library requires **no toolchain from another language ecosystem**.
- **SC-006**: The package carries **no** model dependency, **no** model credential, and makes **no** model call.
- **SC-007**: The data and evaluation layers are **unchanged**; every one of their tests passes unmodified.
- **SC-008**: The assistant-facing surface is **optional**: a user who ignores it installs nothing extra.
- **SC-009**: The breaking changes are stated plainly, so an existing user knows exactly what to do.

## Assumptions

- **The graphical app's users are better served by an assistant.** Both serve someone who wants the library without
  writing Python. The assistant reaches all of the library, is tested, needs no second toolchain, and does not have to
  be redesigned every time a feature is added. This is a judgement, and it is the reason the app is being retired
  rather than repaired.
- **A pre-1.0 library may break its configuration contract once, cleanly.** Keeping the old one alive alongside the new
  one would mean two ways to say the same thing, one of which cannot express half the library.
- **The assistant is not shipped.** The library exposes a surface; the user brings their own assistant. This keeps
  every model concern — the key, the choice, the cost, the nondeterminism — outside a package whose value is
  predictable, composable estimators.
- **Cost visibility is a safety property, not a convenience.** An assistant acting on a user's behalf with a metered,
  paid credential must never be able to spend by accident. The library already computes the estimate exactly and for
  free; this feature simply refuses to let any surface skip it.

## Out of Scope

- **Putting a model inside the library.** Explicitly rejected: it would drag a credential, a model choice, a per-call
  cost and nondeterminism into a package built to be predictable, and a betting library that generates advice with a
  language model is a liability. The assistant stays outside and drives the library.
- **Reviving the graphical app in another framework.** If a graphical app is wanted later, it belongs in its own
  package, built on this library like any other consumer.
- **A hosted service or public web API.**
- **Changing what the library computes.** No new source, sport, market or model.
