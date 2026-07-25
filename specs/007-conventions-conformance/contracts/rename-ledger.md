# Rename ledger: Conventions conformance

**Feature**: [spec.md](../spec.md)

Every public-symbol rename, recorded as it is made, so the changelog and the callers stay in step (FR-003,
FR-016, SC-007). A rename changes the identifier only; behaviour is held constant. Filled per package as the
sweep proceeds; empty rows are packages not yet done.

| Package | Old name | New name | Callers / exports / tests / docs updated | In CHANGELOG |
| --- | --- | --- | --- | --- |
| sources | _to be filled by the sources task_ | | | |
| dataloaders | | | | |
| evaluation | | | | |
| execution | | | | |
| selection & artifacts | | | | |
| cli | | | | |
| mcp | | | | |

## Rules

- A name the scikit-learn contract fixes (research D1) never appears here: it is not renamed.
- A rename lands in the same commit as its package, with every caller, the `__init__` export and `__all__`,
  the tests, the docstrings and doctests, and `docs/` updated, and a `CHANGELOG.md` line with before and after.
- If a rename would collide with an existing public name, it is reconsidered rather than forced, and the
  collision is noted in the package's task.
