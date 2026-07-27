# Phase 0 Research: Plain Documentation

## D1: What is in scope

**Decision**: The README, the user guide pages under `docs/overview/user_guide`, the gallery examples under
`docs/examples`, and the docstrings of the public API that appear in the rendered docs. Out of scope: `docs/generated`
(regenerated), the changelog (generated), private docstrings that are not rendered, and the rest of the constitution
body.

**Rationale**: These are the pages a user reads. They are also where the counter-examples the user gave come from.

## D2: How the plain style is enforced

**Decision**: Human review against a short, concrete list of rules in the constitution. No new automated tool.

**Rationale**: A formatter checks line length and punctuation, but it cannot judge tone. The rules are written to be
checkable by eye, for example "no sentence uses inverted word order for effect" and "the install section has four
subsections".

## D3: How an example is proven to run

**Decision**: The documentation build executes every gallery example, and the doctest run executes every docstring
example. Both are in the gate. An example that would need a secret uses a placeholder, one that would need the network
uses sample data or a fake, and one that cannot run offline is removed.

**Rationale**: The 0.15.0 release showed that a locally set secret hid a broken example. Running every example in the
build, offline, is the proof.

## D4: How docstrings are rewritten without breaking anything

**Decision**: Change only the prose of a docstring. Keep the signature, the parameter and return names, and the `>>>`
doctest blocks with their output unchanged.

**Rationale**: The doctest run checks the output. Changing prose is safe. Changing a doctest block risks the gate.

## D5: The plain style, stated concretely

See `contracts/style-rules.md`. That list becomes the new Writing Style rules in the constitution and the checklist a
reviewer uses.
