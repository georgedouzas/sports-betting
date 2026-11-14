# Design

## Context

The library has three delivery surfaces. The Python API reached 2 sports, 3 leagues and 4 sources. The command line
and the graphical app reached soccer through football-data and nothing else. Principle I of the constitution says the
surfaces MUST expose the same capabilities. This change makes that true by deleting the surface that reached least and
repairing the one that can reach everything.

Measured state before the change:

| | GUI | CLI | Core (`datasets` + `evaluation`) |
| --- | --- | --- | --- |
| Lines | 1,951 | 504 | 5,892 |
| Tests | none | yes | yes |
| Sports reached | 1 | 1 | 2 |

The configuration contract before and after:

| | Before | After |
| --- | --- | --- |
| What is handed over | `DATALOADER_CLASS`, a class | `DATALOADER`, a built dataloader |
| The selection | `PARAM_GRID`, a separate variable | `param_grid=...`, a constructor argument |
| Can it carry a source | No | Yes |
| Can it carry a credential | No | Yes, from `os.environ` |
| Sports reachable | 1 of 2 | all |
| Sources reachable | 1 of 4 | all |

The surfaces after the change:

| Surface | Reaches | Tested |
| --- | --- | --- |
| Python API | everything | yes |
| CLI | everything | yes |
| GUI | deleted | was never tested |
| MCP server | everything | yes |

The cost estimate already existed. `prepare(dry_run=True)` fetches the free statistics, derives the schedule and prices
the odds items exactly, and spends nothing. This change makes the estimate a precondition for any surface that can
spend. A single NBA season quotes at 17,722 credits.

## Goals / Non-Goals

**Goals:**

- Every shipped surface reaches every sport and every data source the library supports.
- The command line takes a configured dataloader, so a source and its credential can be expressed.
- An assistant can go from "what data exists" to "here are the value bets" with no code written by the user.
- A metered fetch is always priced before it is paid for.
- The test run excludes nothing, and no surface needs a toolchain from another language.

**Non-Goals:**

- Putting a model inside the library. It would bring a credential, a model choice, a per-call cost and nondeterminism
  into a package built to be predictable.
- Reviving the graphical app in another framework. A future graphical app belongs in its own package.
- A hosted service or a public web API.
- Changing what the library computes. No new source, sport, market or model. `datasets/**` and `evaluation/**` show a
  zero diff, and every one of their tests passes unmodified.

## Decisions

### D1. Delete the graphical app

The app was a quarter of the codebase, excluded by name from pytest, coverage and `--doctest-modules`. It pinned
`reflex==0.7.0` exactly and could not safely move, pulled in `reflex-ag-grid` and `nest-asyncio`, and needed Node.
Its launcher ran `node`, which is the only reason `bandit` skipped `B404`, `B603` and `B607`. Its footprint outside its
own directory was `pyproject.toml`, `README.md`, `docs/generate_api.py`, and historical `CHANGELOG.md` entries, which
stay as history. Nothing in `datasets` or `evaluation` imported it.

A search for `gui` matches the word "user guide" in many docstrings and pages. The removal matches on word boundaries.

Alternatives considered: repair the app to parity with sources, keys and two sports, which would keep the largest
untested surface and put paid credentials in a local web UI. Spin it out to its own package, which stays open to anyone
who wants it but is not this library's job.

### D2. The command line takes a configured dataloader

The command line asked the configuration for `DATALOADER_CLASS` and `PARAM_GRID` and built `dataloader_cls(param_grid)`
with no sources. `BasketballDataLoader` has `DEFAULT_ODDS = None` and raises when built bare. `OddsApi` needs a key
and a class cannot hold one. The fix is one name in the contract:

```python
DATALOADER = BasketballDataLoader(
    param_grid={'league': ['NBA'], 'year': [2026]},
    stats=NBAStats(),
    odds=OddsApi(key=os.environ['ODDS_API_KEY']),
)
```

`get_dataloader_cls` and `get_param_grid` become one `get_dataloader`, validated with `isinstance(..., BaseDataLoader)`.
`get_bettor` also checked `DATALOADER_CLASS` and changes with it. The configuration is Python, so a key comes from
`os.environ` and is never written into a file that could be committed. This is also why the configuration is a Python
module and not YAML: a YAML file would have to hold the key or invent a syntax for not holding it.

`params` must work before a `param_grid` exists, since it tells the user what to select. `param_grid=None` stays valid,
and `params` asks the configured sources for `available_params`, which is free.

`PARAM_GRID` is removed, not deprecated. A configuration with `DATALOADER_CLASS` fails with a message that names the
replacement, not with an `AttributeError` and not silently.

`tests/conftest.py` holds a `CONFIG` string in the old contract, and it changes. This change exists to change the CLI
contract, so the tests of that contract change with it. No test of the core changes.

Configuration variables after the change:

| Variable | Required | Meaning |
| --- | --- | --- |
| `DATALOADER` | yes | A configured `BaseDataLoader`. Rejected if it is a class, or is not one. |
| `BETTOR` | for `backtest` and `bet` | A configured `BaseBettor`. |
| `ODDS_TYPE`, `DROP_NA_THRES`, `TARGET_EVENT_STATUS`, `TARGET_EVENT_TIME`, `CV`, `N_JOBS`, `VERBOSE` | no | Unchanged |

### D3. The MCP tools take a configuration path

Each tool takes the path of the same Python configuration module the command line reads, and reads it through the same
`get_dataloader`. The two surfaces cannot drift. The credential never enters a tool argument, where it would be logged
and shown in a transcript. An assistant can write the configuration file, and the user can read, keep and rerun it.
Every source is reachable, including one added later, because the configuration is Python.

Alternative rejected: tools that take JSON describing the sport, sources and `param_grid`. That needs a schema extended
for every new source, puts the credential in the tool call, and duplicates the command line contract in a second
dialect.

### D4. An assistant cannot spend a user's money by accident

`prepare` on the server refuses a metered preparation unless the caller passes `confirm_cost` equal to the total that
`estimate_preparation` quotes. When the confirmation is missing or wrong, the refusal names the real cost. A preparation
whose total cost is zero, because every source is free or everything is already held, needs no confirmation.

```text
estimate_preparation(config) -> {to_fetch: 898, held: 0, cost: {'odds_api': 17722}}
prepare(config, confirm_cost=17722)        runs
prepare(config)                            refused, the cost was never confirmed
prepare(config, confirm_cost=100)          refused, the real cost is 17722
```

Alternative rejected: a tool description asking the model to check first. A sentence in a docstring is not a guard.

### D5. The assistant stays outside the package

The library exposes a surface. The user brings the assistant. No model, model credential or model call enters the
package. The library is already legible to an assistant: 37 public names, complete type annotations, real docstrings
and a working command line. The MCP server is a thin adapter over the public API. It sits in `src/sportsbet/mcp/` as a
peer of `src/sportsbet/cli/` and imports only the public API.

Alternative rejected: embedding an agent. It would bring a model credential, a model choice, a per-call cost and
nondeterminism into a library of estimators meant to behave predictably inside a scikit-learn `Pipeline`.

### D6. Two breaking changes in one release

`sportsbet-gui` and the `gui` extra go, and `DATALOADER_CLASS` with `PARAM_GRID` becomes `DATALOADER`. Both ship in
one release, each with a before and after in the changelog. A pre-1.0 library takes one clean break rather than a
deprecation cycle that keeps a configuration dialect unable to express half the library.

### D7. Dataframes cross the wire as records

An assistant cannot consume a pandas object, so the server returns extracted data, backtest results and value bets as
JSON records that round-trip (task T024).

### Requirement coverage

FR-001, FR-002, FR-003, FR-004, FR-005 and FR-012 land in `interfaces/cli`. FR-010 and FR-011 land in the removal of
the graphical app requirement. FR-006, FR-007, FR-008, FR-009 and FR-014 land in `interfaces/mcp-server`. FR-013 (the
test configuration excludes nothing) and FR-015 (the data and evaluation layers do not change) are constraints on the
work, recorded here and verified by tasks T003 and T032.

## Risks / Trade-offs

- A user who wanted the graphical app loses it. The replacement reaches more of the library. A future graphical app
  belongs in its own package, built on this library.
- The configuration break could strand existing users. An old configuration fails with a message that says what to
  change, and the changelog carries a before and after snippet.
- An assistant could spend a user's odds credits. The `confirm_cost` rule is enforced in code, not in a tool
  description.
- The MCP server could drift from the command line and recreate the parity bug. The server reuses the command line
  configuration contract and `get_dataloader`.
- Scope could creep into the core. `datasets/**` and `evaluation/**` must show a zero diff. If they need to change, the
  work stops and reports.

## Migration Plan

1. Phase 1 deletes the GUI. After it, `addopts` has no `--ignore`, `pdm run checks` passes with the stricter `bandit`
   configuration, and the package installs with no Node.
2. Phase 2 gives the command line a configured dataloader. The checkpoint is that the command line prepares an NBA
   dataset with a paid odds key, verified with the free `--dry-run`.
3. Phase 3 adds the MCP server. It reuses `get_dataloader`, so Phase 2 must land first.
4. Phase 4 updates the docs and records both breaking changes in `CHANGELOG.md` with a before and after.

A user migrates a configuration by replacing:

```python
DATALOADER_CLASS = SoccerDataLoader
PARAM_GRID = {'league': ['England'], 'year': [2025]}
```

with:

```python
DATALOADER = SoccerDataLoader(param_grid={'league': ['England'], 'year': [2025]})
```

A user of `sportsbet-gui` installs `sportsbet[mcp]` and connects an assistant to `sportsbet-mcp`, or uses the
`sportsbet` command line.
