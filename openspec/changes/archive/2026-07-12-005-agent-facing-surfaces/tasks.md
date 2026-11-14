# Tasks

## 1. Delete the GUI (US3)

- [x] 1.1 Remove `src/sportsbet/gui/` with `git rm -r`: 1,951 lines, zero tests, one sport. (T001)
- [x] 1.2 In `pyproject.toml`, remove the `gui` optional dependency group (`nest-asyncio`, `reflex==0.7.0`,
  `reflex-ag-grid`) and the `sportsbet-gui = "sportsbet.gui.main:run"` entry point. The Node toolchain goes with them.
  (T002)
- [x] 1.3 In `pyproject.toml`, remove `--ignore=src/sportsbet/gui` from `addopts`, so the test run excludes nothing
  (SC-004). (T003)
- [x] 1.4 In `pyproject.toml`, remove the `bandit` skips `B404`, `B603`, `B607` and the comment above them. Keep
  `B101`. If `bandit` then flags something real, stop and report, and do not put the skip back. (T004)
- [x] 1.5 Remove the GUI from `docs/generate_api.py`. (T005)
- [x] 1.6 Remove the GUI screenshots section from `README.md`. The agent story arrives in group 4. (T006)

## 2. The CLI takes a configured dataloader (US1)

- [x] 2.1 In `src/sportsbet/cli/_utils.py`, replace `get_dataloader_cls` and `get_param_grid` with a single
  `get_dataloader(mod)` that requires `DATALOADER`, validates it with `isinstance(..., BaseDataLoader)`, and returns the
  instance. (T007)
- [x] 2.2 In `src/sportsbet/cli/_utils.py`, make a configuration that still carries `DATALOADER_CLASS` fail with a
  message that names the change, not an `AttributeError` and not silence (FR-002). (T008)
- [x] 2.3 In `src/sportsbet/cli/_utils.py`, update `get_bettor`, which also checks for `DATALOADER_CLASS`. (T009)
- [x] 2.4 In `src/sportsbet/cli/_data.py`, update `params`, `prepare`, `odds_types`, `training`, `fixtures` to take the
  configured dataloader. (T010)
- [x] 2.5 In `src/sportsbet/cli/_betting.py`, update `backtest` and `bet`. (T011)
- [x] 2.6 Keep `params` working when nothing is selected: `param_grid=None` stays a valid configuration. It reads
  `DATALOADER.sources` and asks each source's `available_params`, which is free. (T012)
- [x] 2.7 In `tests/conftest.py`, rewrite the `CONFIG` string from `DATALOADER_CLASS` and `PARAM_GRID` to a configured
  `DATALOADER`. This edit is expected, since the contract it tests is the one this change changes. (T013)
- [x] 2.8 Test that a configuration using the old contract fails with the message that names the change. (T014)
- [x] 2.9 Test that a configuration whose dataloader carries a credentialled source works. (T015)
- [x] 2.10 Test that `params` works with `param_grid=None`. (T016)

## 3. The MCP server (US2)

- [x] 3.1 In `pyproject.toml`, add an `mcp` optional dependency group (the `mcp` package) and a `sportsbet-mcp` entry
  point. Add `mcp` to the lockfile with `pdm` and make sure the test environment has it. (T017)
- [x] 3.2 Create `src/sportsbet/mcp/` as a peer of `src/sportsbet/cli/`. It imports only the public API. (T018)
- [x] 3.3 Make every tool take a configuration path, the same Python configuration module the CLI reads, so one
  `get_dataloader` serves both surfaces and the credential never enters a tool argument. (T019)
- [x] 3.4 Implement the free tools `available_params` (what leagues and seasons exist, downloads nothing) and
  `estimate_preparation` (what a preparation would fetch and exactly what it would cost). (T020)
- [x] 3.5 Make `prepare` refuse unless the caller passes `confirm_cost` equal to the total the estimate quotes
  (FR-007, SC-003). (T021)
- [x] 3.6 Let a preparation whose total cost is zero run with no confirmation. (T022)
- [x] 3.7 Implement the remaining tools: `extract_train_data`, `extract_fixtures_data`, `backtest`, `bet`. (T023)
- [x] 3.8 Decide and record how a dataframe crosses the wire: records as JSON, and it must round-trip. (T024)
- [x] 3.9 In `tests/test_mcp.py`, test that the tool set exists, that a free preparation needs no confirmation, and
  that the tools reach every sport through the CLI configuration. (T025)
- [x] 3.10 In `tests/test_mcp.py`, test the refusals: a metered preparation is refused with no `confirm_cost`, refused
  with the wrong `confirm_cost` with a message that names the real cost, and allowed with the right one. No test
  touches the network. (T026)

## 4. Docs and the breaking changes

- [x] 4.1 In `README.md`, add the agent story (`pip install 'sportsbet[mcp]'`) where the screenshots were. (T027)
- [x] 4.2 In `docs/overview/user_guide/`, update the CLI page for the new configuration, showing a credentialled
  source that reads the key from `os.environ` (FR-004). (T028)
- [x] 4.3 In `docs/overview/user_guide/`, add a page for driving the library with an assistant and register it in
  `properdocs.yml`. (T029)
- [x] 4.4 In `CHANGELOG.md`, record two breaking changes, each with a before and after snippet: `sportsbet-gui` and the
  `gui` extra are gone, and `DATALOADER_CLASS` with `PARAM_GRID` is replaced by a configured `DATALOADER`. (T030)
- [x] 4.5 Rerun the public API docs audit. Every public name must still have a runnable example (37 of 37 before).
  (T031)
- [x] 4.6 Verify SC-007: `git diff --stat -- src/sportsbet/datasets src/sportsbet/evaluation` is empty. Verify SC-004:
  `addopts` excludes nothing. Verify SC-006: no model dependency anywhere. (T032)
