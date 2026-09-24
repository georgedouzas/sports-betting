# Quickstart: validating constitution conformance

**Feature**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Date**: 2026-09-21

How to check that a pass of this feature landed, and that it landed without moving anything a user depends on. Run
these from the repository root.

## Prerequisites

- The toolchain the Project Profile names: PDM, `nox`, and the `pdm run` shortcuts.
- The optional extras the tests need, which the `tests` session installs itself.
- No credential set in the environment. The build runs the examples with no secret, and a secret that is set can hide
  an example that needs one.

## The gate, in order

The constitution's Gates section fixes the order. Run the whole sequence before opening a PR, and read the verdict
from the session summary line rather than from a piped exit code.

```console
pdm run formatting
pdm run checks
pdm run docs build
pdm run tests
```

Green looks like `Session tests-3.13 was successful.` A reused environment can carry a stale toolchain, so settle any
doubt about a lint finding with a clean run.

## Per story

### User Story 1, docstrings

```console
pdm run checks
pdm run tests
```

- `interrogate` reports full docstring coverage, so the 31 names that carry none now carry one.
- The conformance check reports no public name missing a block its signature calls for, down from 111.
- The conformance check reports no private name carrying a block, which already holds and must keep holding.
- The doctest run passes, since `--doctest-modules` makes every example a test.

### User Story 2, the surface

```console
pdm run checks
pdm run tests
```

- The conformance check reports every package declaring `__all__`, up from one of seven.
- It reports no name defined in a private module that is public without its package re-exporting it, down from 110.
- It reports no type-checking import guard, down from four.
- Every module still imports, which the doctest collection proves by importing the tree.

The contract in [contracts/public-surface.md](./contracts/public-surface.md) is what tells you a rename went too far:
if a test or a documentation page needed editing to keep passing, a name on that list moved.

### User Story 3, comments

```console
pdm run checks
```

The conformance check reports no comment outside a licence header or a suppression, down from 75.

### User Story 4, examples

```console
pdm run docs build
pdm run tests
```

- The documentation build executes every page block, so a page that stopped working fails here.
- The doctest run executes every docstring example.
- The conformance check reports every offline public name carrying an example, and no name that needs the network or a
  credential carrying one.

To see an example fail the way a reader would, break one on purpose and rerun. The build must go red.

### User Story 5, the Project Profile

There is no command for this one. Read each bullet of the Project Profile in `openspec/constitution.md` against
the tree and confirm it holds: the layers, the builders, the four named exceptions, and the front page naming every
public surface the package ships.

## What a regression looks like

| Symptom | What it means |
| --- | --- |
| A test needed an edit | A public name moved. Check it against the contract |
| The documentation build went red | An example stopped running, or a page referenced a renamed name |
| The doctest run went red but the suite passed | An example is wrong, which is the point of running them |
| `interrogate` fell | A docstring was removed rather than written |
| The check passes but a reviewer disagrees | One of the three rules a check cannot decide, listed in the rule model |

## Bounds

- The public behaviour of the library does not change, so the existing test suite passes unchanged. That is the single
  strongest signal, and it is worth running first after any pass.
- Coverage does not fall. New logic is not added by this feature, so any change in coverage is a sign that something
  other than a docstring moved.
