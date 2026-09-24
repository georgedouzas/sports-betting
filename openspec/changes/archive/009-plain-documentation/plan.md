# Implementation Plan: Plain Documentation

**Branch**: `009-plain-documentation` | **Date**: 2026-07-28 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/009-plain-documentation/spec.md`

## Summary

Rewrite the user-facing documentation and every example in plain English, and update the constitution to require that
style and to record the lessons from the 0.15.0 release. The scope is the README, the user guide pages, the gallery
examples, and the public docstrings shown in the rendered docs. No library behavior or public API changes. The proof is
the full gate, run in order: formatting, checks, the documentation build that executes the examples, then tests.

## Technical Context

**Language/Version**: Python `>=3.11, <3.14`. Documentation is Markdown and Python gallery scripts.

**Primary Dependencies**: no new dependency. The docs build (mkdocs plus mkdocs-gallery) executes the gallery examples.
The doctest run executes the docstring examples.

**Storage**: N/A.

**Testing**: the documentation build runs every gallery example. The doctest run (`--doctest-modules`) runs every
docstring example. Both are part of the gate.

**Target Platform**: the rendered documentation site and the README on the repository page.

**Project Type**: a library with prose documentation and runnable examples.

**Constraints**: documentation and constitution only. Every example runs offline, with no secret and no network. The
gate stays green in order: formatting, checks, docs build, tests, on 3.11, 3.12, and 3.13. `docs/generated` is
regenerated, never hand-edited.

**Scope/Scope**: the README, five user guide pages, four gallery directories, the public docstrings, and the
constitution's Writing Style and example rules.

## Constitution Check

*GATE: checked before and after design. Constitution v2.3.0.*

- **I to IV**: unaffected. No code behavior, types, schemas, or gate mechanics change beyond adding no example that
  needs a secret.
- **V. Documentation as a First-Class Artifact**: this feature is that principle applied. Every public thing keeps its
  docstring, every example runs, and the docs move with the code.
- **VI. A Library, Not an Application**: unaffected. No surface or public name changes.
- **Writing Style**: this feature strengthens the Writing Style rules. That is a constitution amendment, recorded with
  a version bump and a Sync Impact Report.

No violations. Complexity Tracking stays empty.

## Project Structure

```text
README.md                              # rewrite, with install subsections
docs/overview/user_guide/*.md          # rewrite index, sources, dataloader, bettor, execution
docs/examples/**/plot_*.py             # rewrite prose, keep every example runnable offline
src/sportsbet/**/*.py                   # rewrite public docstring prose, keep signatures and doctests
.specify/memory/constitution.md        # strengthen Writing Style, add the example-runs rule, record release lessons
```

**Structure Decision**: documentation and constitution only. No source layout change. Docstring edits change prose, not
signatures, names, or the `>>>` doctest blocks and their output.

## Complexity Tracking

No violations, so this table is empty.
