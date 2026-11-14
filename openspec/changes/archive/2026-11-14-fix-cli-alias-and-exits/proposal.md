# Proposal

## Why

Two defects in the command line surface.

The `--alias` help says the form is `stats name=odds name`, and so does the `aliases` docstring of `build_dataloader`
that the MCP tools point to. The code reads it the other way: `_parse_aliases` maps the first name to the second, and
`resolve_odds` takes that mapping as odds name to statistics name. The example in the error message,
`Olimpia Milano=EA7 Emporio Armani Milan`, is in the order the code reads. A user who follows the help gets an alias
that pairs nothing, and the club stays without odds with no error.

A selection that fails to build does not fail the command. `_build_selected` and `_build_modelled` catch `BuildError`,
print it in a red panel and yield `None`, and the command then returns, so the exit status is 0. A script that chains
commands carries on after the failure. Worse, a `BuildError` raised inside the `with` block, such as a malformed
`--target-event-time` read by `_build_extraction`, is thrown back into the generator, which yields a second time, and
Python raises `RuntimeError: generator didn't stop after throw()` with a traceback.

## What Changes

- The `--alias` help, the `build_dataloader` docstring and the parsing variable names state the form the code reads,
  `odds name=stats name`. The behaviour of an alias does not change, so aliases that work today keep working.
- A selection or a model that fails to build makes the command print the message in a red panel and exit with status 1.
  The two context managers become plain builders, and every library error reaches `_report_errors`.

## Non-goals

- Catching `ExecutionError` in the execution commands. That is task 6.1 of `rebuild-single-event-execution`, which
  rewrites those commands.
- Swapping the alias direction to match the old help. That would break every alias that works today.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `interfaces/cli`: the stated form of `--alias`, and the exit status of a build error.
- `interfaces/mcp-server`: the stated form of `aliases`.

## Impact

- Code: `src/sportsbet/cli/_options.py`, `src/sportsbet/cli/_building.py`, `src/sportsbet/cli/_data.py`,
  `src/sportsbet/cli/_betting.py`, `src/sportsbet/dataloaders/_factory.py`.
- Tests: `tests/cli/` gains exit status tests for a build error before and inside the block, and a test that an alias in
  the documented form pairs a club.
- Docs: `docs/overview/user_guide/` pages that show `--alias`.
