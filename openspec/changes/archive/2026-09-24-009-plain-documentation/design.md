# Design

## Context

The documentation is part of the product under Principle V of the constitution (constitution v2.3.0 at planning time).
The pages used an inverted, literary voice, and the 0.15.0 release showed that a locally set secret could hide a broken
example. This change rewrites the user-facing documentation in plain English, keeps every example runnable offline, and
amends the constitution so the style holds. No library behaviour, type, schema or public name changes.

## Goals / Non-Goals

**Goals:**

- A new reader installs the library and runs the first example by following the docs, without asking what a sentence
  means (SC-001).
- Every code example in the docs runs and passes in the documentation build and the doctest run (SC-002).
- No code example depends on a secret or the network (SC-003).
- Every page in scope has headings, and the install section has the four named subsections (SC-004).
- A reviewer can judge any page against the Writing Style rules by eye, with a clear pass or fail (SC-005).
- The full gate passes in order: formatting, checks, the documentation build, tests, on 3.11, 3.12 and 3.13 (SC-006).

**Non-Goals:**

- Any change to library behaviour, the public API, or any name (FR-008).
- Private docstrings that are not rendered, `docs/generated`, and the generated changelog.
- A new automated style tool.

## Decisions

### D1. Scope

The README, the user guide pages under `docs/overview/user_guide`, the gallery examples under `docs/examples`, and the
docstrings of the public API that appear in the rendered docs. These are the pages a user reads, and they are where the
counter-examples came from.

### D2. The style is enforced by human review

A short, concrete list of rules in the constitution, checked by a reviewer. A formatter checks line length and
punctuation but cannot judge tone. The rules are written to be checkable by eye, for example "no sentence uses inverted
word order for effect" and "the install section has four subsections".

Alternative considered: a new automated tool. Rejected, since a tool cannot judge tone.

### D3. An example is proven to run by the build

The documentation build executes every gallery example, and the doctest run executes every docstring example. Both are
in the gate. An example that needs a secret uses a placeholder, one that needs the network uses sample data or a fake,
and one that cannot run offline is removed. The existing sample sources and fixtures are enough for this.

### D4. Docstrings change prose only

A docstring rewrite keeps the signature, the parameter and return names, and the `>>>` doctest blocks with their output.
The doctest run checks the output, so changing a doctest block risks the gate.

### D5. The plain style rules

These rules became the constitution's Writing Style rules and the review checklist.

Do:

- Write short, direct sentences. One idea per sentence.
- Say who does what in the normal order: subject, then verb, then object. "The library places the bet at the venue."
- Use plain words. Prefer the common word over the clever one.
- State a real risk once, in the right place, plainly. Then move on.
- Give each page headings and subsections, so a reader can scan it. Split long blocks.
- Make every code example run. Use sample data, a placeholder, or a fake.

Do not:

- Invert word order for effect. Not "A venue with an API is placed at by the library."
- Use the passive voice unless the doer is unknown or does not matter.
- Use idioms, metaphors, or literary flourish.
- Hedge or repeat a warning, or sound defensive.
- Write a page as one undivided block of text.
- Write a code example that cannot run. No fragments, no pseudo-code, no demo that needs a secret or the network.

Kept from before:

- A line is at most 120 characters.
- A sentence uses no semicolon and no dash as punctuation. Hyphenated words are fine.

Rewrite examples:

- Before: "A venue with an official betting API is placed at by the library." After: "The library places bets at a venue
  that has a betting API."
- Before: one block that mixes the basic install, the MCP extra, the execution extra and the development install.
  After: four subsections, one per install, each with its own heading and commands.
- Before: a single sentence describing the single-event unit that takes a fitted bettor, one upcoming match and a fixed
  stake, watches the event and places the model's bet once at a bookmaker. After: "A bettor finds value bets. The
  single-event unit places one of them. It takes a fitted bettor, one match, and a stake. It watches the match. At the
  moment the model was fitted for, it places the bet, once. You place the bet at a bookmaker where you hold an account.
  This spends real money."

### D6. Constitution amendments

The constitution's Writing Style rules were rewritten to require the plain style and forbid the clever, inverted,
passive-for-effect, idiomatic and defensive style. A rule was added that a code example must run, verified by the
build, with no un-runnable demo. The 0.15.0 release lessons were recorded:

- The gate includes the documentation build, and the build runs the examples.
- A cached toolchain or a locally set secret can hide a failure that a clean run finds.
- An example must not depend on a secret to build.
- `development` stays a superset of the released `main`, and the version tool must not collide with an existing tag.

The amendment bumped the constitution version and carries a Sync Impact Report.

### Requirement coverage

FR-001 to FR-008 and FR-012 govern the documentation rewrite. FR-009 to FR-011 govern the constitution amendment. None
of them changes observable library behaviour, which is why the change sets `skip_specs: true`.

## Risks / Trade-offs

- An example cannot be made to run offline, for example because it needs a paid live feed. It is rewritten to use
  sample data or a placeholder, or it is removed. It is not left as a fragment.
- A page states the same risk in several places. The rewrite states it once, in the right place.
- A rewrite would change what a public function does or is called. That is out of scope, and only the wording changes.
- A changed doctest block could break the gate. Docstring edits keep every `>>>` block and its output.
