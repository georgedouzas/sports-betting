# Proposal

## Why

The user asked: "do you think having a gui makes sense or remove it all together and replace it with a CLI that uses an
agent like claude?"

The library promised three ways in: code, a command line, and a graphical app. It promised that all three reach the same
capabilities. Two of the three did not. While the data layer gained injectable sources, a second sport, three leagues
and paid odds, the command line and the graphical app could reach exactly one configuration: soccer with the free
football-data feed.

The root cause was the command line configuration. It handed over `DATALOADER_CLASS`, a class, plus `PARAM_GRID`. A
class cannot carry a source, and a source is what carries a credential. So `BasketballDataLoader`, which has no free
odds default and raises when built bare, was unreachable, and `OddsApi` was unreachable for every sport.

The graphical app was 1,951 lines, a quarter of the codebase. The test run skipped it by name
(`--ignore=src/sportsbet/gui`), so pytest, coverage and doctests never ran on it. It pinned `reflex==0.7.0` exactly,
pulled in `reflex-ag-grid` and `nest-asyncio`, needed a Node toolchain, and forced three `bandit` suppressions
(`B404`, `B603`, `B607`) because its launcher ran `node`. It reached one sport.

This change retires the graphical app, fixes the configuration, and adds a surface an assistant can drive. The assistant
is a consumer of the library. The package gains no model, no model credential and no model call.

## What Changes

- **BREAKING**: Remove the graphical app `sportsbet-gui`, its `gui` extra (`reflex`, `reflex-ag-grid`,
  `nest-asyncio`) and the Node toolchain it needed.
- **BREAKING**: The command line configuration reads `DATALOADER`, a configured dataloader instance, in place of
  `DATALOADER_CLASS` and `PARAM_GRID`. A configuration that still uses `DATALOADER_CLASS` fails with a message that
  names the change.
- A credential in the configuration comes from the environment through `os.environ`, so the file never holds it.
- `sportsbet dataloader params` works with a dataloader that has no `param_grid` yet.
- Add the `sportsbet-mcp` server as the optional `mcp` extra. Its tools are `available_params`,
  `estimate_preparation`, `prepare`, `extract_train_data`, `extract_fixtures_data`, `backtest` and `bet`. Each tool
  takes the path of the same configuration module the command line reads.
- `prepare` on the server refuses a metered preparation unless the caller passes `confirm_cost` equal to the quoted
  cost. A free preparation needs no confirmation.
- The test run excludes nothing, and `bandit` no longer skips `B404`, `B603` and `B607`.

## Capabilities

### New Capabilities

- `interfaces/mcp-server`: the `sportsbet-mcp` server, its tools, the configuration path they take, and the rule that a
  metered preparation needs a confirmed cost.

### Modified Capabilities

- `interfaces/cli`: the configuration contract changes from a dataloader class to a configured dataloader, the command
  line reaches every sport and source, and the graphical app is removed.

## Impact

- Code removed: `src/sportsbet/gui/`.
- Code changed: `src/sportsbet/cli/_utils.py` (`get_dataloader_cls` and `get_param_grid` become `get_dataloader`,
  and `get_bettor` stops checking `DATALOADER_CLASS`), `src/sportsbet/cli/_data.py` (`params`, `prepare`,
  `odds_types`, `training`, `fixtures`), `src/sportsbet/cli/_betting.py` (`backtest`, `bet`), `tests/conftest.py` (the
  `CONFIG` string).
- Code added: `src/sportsbet/mcp/` and `tests/test_mcp.py`.
- Not touched: `src/sportsbet/datasets/**` and `src/sportsbet/evaluation/**` show a zero diff.
- Dependencies: `mcp` becomes an optional extra in the slot `gui` held. No new required dependency. Removed:
  `reflex==0.7.0`, `reflex-ag-grid`, `nest-asyncio`.
- `pyproject.toml`: remove the `gui` extra, the `sportsbet-gui` entry point, the pytest `--ignore`, and the three
  `bandit` skips. Add the `mcp` extra and the `sportsbet-mcp` entry point.
- Docs: `README.md` drops the GUI screenshots and gains the agent story, `docs/generate_api.py` drops the GUI, the CLI
  user guide shows a credentialled source read from `os.environ`, a new page `docs/overview/user_guide/agent.md` covers
  driving the library with an assistant, and `CHANGELOG.md` records two breaking changes with before and after.
- The public API docs audit must still report a runnable example for every public name (37 of 37 before the change).
- Specification quality checklist: every item passed. The first draft failed "no implementation details" and was
  rewritten. Two items were kept on purpose: the success criterion that the test run excludes nothing, and the
  requirement that no model enters the library, which is a scope boundary rather than a technical detail.
