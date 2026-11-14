# Tasks

## 1. Foundational: the standard

- [x] 1.1 Update `openspec/constitution.md`: rewrite the Writing Style rules to require the plain style and forbid the
  clever, inverted, passive-for-effect, idiomatic and defensive style, per the style rules in the design. Add the rule
  that a code example must run, verified by the build, with no un-runnable demo. Record the 0.15.0 release lessons (the
  docs build in the gate runs the examples, a cached toolchain or a local secret can mask a failure, an example must not
  need a secret, `development` stays a superset of the released `main`, the version tool must not collide with an
  existing tag). Bump the version and write the Sync Impact Report. Keep the file within its own rules. (T001)

## 2. US1: the docs read plainly

- [x] 2.1 Rewrite `README.md` in plain English. Split the install into four subsections: basic, MCP extra, execution
  extra, development. Keep the badges and links. (T002)
- [x] 2.2 Rewrite `docs/overview/user_guide/index.md` in plain English with headings. (T003)
- [x] 2.3 Rewrite `docs/overview/user_guide/sources.md`. (T004)
- [x] 2.4 Rewrite `docs/overview/user_guide/dataloader.md`. (T005)
- [x] 2.5 Rewrite `docs/overview/user_guide/bettor.md`. (T006)
- [x] 2.6 Rewrite `docs/overview/user_guide/execution.md`. (T007)

## 3. US2: every example runs

- [x] 3.1 Rewrite the prose in the gallery examples under `docs/examples/sources/*.py`. Keep every example runnable
  offline, with no secret and no network. (T008)
- [x] 3.2 Rewrite the prose in `docs/examples/dataloaders/*.py` and `docs/examples/modelling/*.py`. (T009)
- [x] 3.3 Rewrite the prose in `docs/examples/execution/*.py`. (T010)
- [x] 3.4 Rewrite the prose of the public front page docstrings: each package `__init__` and the main public classes and
  functions in `src/sportsbet`. Change prose only. Keep signatures, names, and the `>>>` doctest blocks and their
  output unchanged. (T011)

## 4. US3: the rules hold

- [x] 4.1 Review every rewritten page against the style rules. Fix anything that still reads clever, inverted or
  defensive, and any page that is one undivided block. (T012)

## 5. Polish

- [x] 5.1 Run `pdm run docs build` to regenerate `docs/generated` and confirm every example executes. Never hand-edit
  `docs/generated`. (T013)
- [x] 5.2 Run the full gate in order on 3.11, 3.12 and 3.13: formatting, checks, docs build, tests. Confirm green from
  the nox session summaries. (T014)
