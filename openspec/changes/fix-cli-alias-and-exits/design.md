# Design

## Decisions

### D1. Fix the words, not the code, for aliases

The code, the shipped gallery example and the error message all read an alias as odds name first. Only the help and
the docstring say otherwise. Changing the words fixes the defect without breaking a working configuration. The parsing
variables in `_parse_aliases` are renamed to `odds_name` and `stats_name` in the order they are read, so the code says
what it does.

### D2. Builders, not context managers

`_build_selected` and `_build_modelled` become plain functions that return the dataloader and the bettor and let
`BuildError` propagate. Each command already runs inside `_report_errors`, which prints the red panel and exits with
status 1. The `if loader is None: return` branches go. A generator context manager that catches and yields again was
the cause of both the exit status 0 and the `RuntimeError`, so removing it removes both.
