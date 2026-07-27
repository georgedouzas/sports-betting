---

description: "Task list for plain documentation"
---

# Tasks: Plain Documentation

**Input**: design documents in `specs/009-plain-documentation/`. Style rules in `contracts/style-rules.md`.

**Tests**: the documentation build and the doctest run are the proof. Both are in the gate.

## Phase 1: Foundational (the standard)

- [x] T001 Update `.specify/memory/constitution.md`: rewrite the Writing Style rules to require the plain style and
  forbid the clever, inverted, passive-for-effect, idiomatic, and defensive style, per `contracts/style-rules.md`. Add
  the rule that a code example must run, verified by the build, with no un-runnable demo. Record the 0.15.0 release
  lessons (docs build in the gate runs the examples, a cached toolchain or a local secret can mask a failure, an
  example must not need a secret, `development` stays a superset of the released `main`, the version tool must not
  collide with an existing tag). Bump the version and write the Sync Impact Report. Keep the file within its own rules.

## Phase 2: US1 - the docs read plainly (P1)

- [x] T002 [P] Rewrite `README.md` in plain English. Split the install into four subsections: basic, MCP extra,
  execution extra, development. Keep the badges and links.
- [x] T003 [P] Rewrite `docs/overview/user_guide/index.md` in plain English with headings.
- [x] T004 [P] Rewrite `docs/overview/user_guide/sources.md`.
- [x] T005 [P] Rewrite `docs/overview/user_guide/dataloader.md`.
- [x] T006 [P] Rewrite `docs/overview/user_guide/bettor.md`.
- [x] T007 [P] Rewrite `docs/overview/user_guide/execution.md`.

## Phase 3: US2 - every example runs (P2)

- [x] T008 [P] Rewrite the prose in the gallery examples under `docs/examples/sources/*.py`. Keep every example
  runnable offline, no secret, no network.
- [x] T009 [P] Rewrite the prose in `docs/examples/dataloaders/*.py` and `docs/examples/modelling/*.py`.
- [x] T010 [P] Rewrite the prose in `docs/examples/execution/*.py`.
- [x] T011 [P] Rewrite the prose of the public front-page docstrings: each package `__init__` and the main public
  classes and functions in `src/sportsbet`. Change prose only. Keep signatures, names, and the `>>>` doctest blocks and
  their output unchanged.

## Phase 4: US3 - the rules hold (P3)

- [x] T012 Review every rewritten page against `contracts/style-rules.md`. Fix anything that still reads clever,
  inverted, or defensive, or any page that is one undivided block.

## Phase 5: Polish

- [x] T013 Run `pdm run docs build` to regenerate `docs/generated` and confirm every example executes. Never hand-edit
  `docs/generated`.
- [x] T014 Run the full gate in order on 3.11, 3.12, and 3.13: formatting, checks, docs build, tests. Confirm green
  from the nox session summaries.

## Notes

- Docstring rewrites change prose only. A changed `>>>` block or output risks the gate.
- No example depends on a secret or the network. Use sample data, a placeholder, or a fake.
- Commit after logical groups. No Claude co-author trailer.
