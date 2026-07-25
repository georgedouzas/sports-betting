# Phase 1 Quickstart: the per-package gate recipe

**Feature**: [spec.md](./spec.md) | **Date**: 2026-07-20

How one package is brought to conformance and proven. Every package follows the same recipe, and a package
is done only when it ends green.

## Prerequisites

The green baseline: `pdm run formatting`, `pdm run checks`, `pdm run tests` all pass on 3.11/3.12/3.13, and
the browser is installed for the execution tests (`pdm run python -m playwright install --only-shell chromium`).

## The recipe for one package

1. Re-scan the package against the violation categories in [data-model.md](./data-model.md): module
   docstrings, function names, private-helper docstrings, base-module sibling imports, in-function imports,
   inline comments.
2. Reword every module docstring to one imperative line (FR-005), keeping the module's meaning.
3. Rename every non-conforming function to a verb phrase (FR-001), skipping the D1 contract surface. Record
   each public rename in [contracts/rename-ledger.md](./contracts/rename-ledger.md) and update every caller,
   the `__init__` export and `__all__`, and the docs, in this commit.
4. Collapse private-helper docstrings to one line (FR-006); drop essay bodies and inline comments (FR-007,
   FR-011).
5. Make any base module self-contained (FR-009) and move any in-function import out unless it defers an extra
   (FR-010).
6. Bring the package's tests to the mirror: rename the test module to the mirrored path, rename each test to
   `test_<function>_<behavior>`, collapse test docstrings to one line, and switch private-module imports to the
   public API, exporting a helper if it is worth testing directly (FR-012 to FR-015).
7. Add a `CHANGELOG.md` entry for any public rename, with before and after.
8. Run the gate:

   ```bash
   pdm run formatting
   pdm run checks
   pdm run tests
   ```

   All three green on 3.11/3.12/3.13. If anything is red, fix it before the commit.
9. Commit the package as one reviewable unit.

## Definition of done for a package

- Zero non-conforming module docstrings in the package (SC-001).
- Every public function verb-first, the D1 surface untouched (SC-002).
- Every private helper and ordinary function one-line-docstringed (SC-003).
- The package's base module imports no sibling; no function-body import except a deferred extra (SC-004).
- The package's tests mirror its source and are named `test_<function>_<behavior>` (SC-005).
- The full gate green on three versions (SC-006), the public API unchanged but for the ledgered renames
  (SC-007), and every runnable example still runs (SC-008).
