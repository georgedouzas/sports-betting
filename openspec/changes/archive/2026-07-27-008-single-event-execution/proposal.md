# Proposal

## Why

The batch quote-and-confirm path placed a slate of value bets across many upcoming matches at once. That is the
behaviour a bookmaker recognises as automation and blocks, and it did not do the one thing a bettor needs: watch one
event and act on it at the right moment. The user asked to redesign the execution layer around a single betting event.
A single execution unit monitors one event preplay, inplay and postplay, logs the event to the terminal, and uses a
fitted bettor to decide automatically to bet at the right time. Extending to many events is the user's responsibility,
and the user pays the price if the bookmaker blocks that behaviour.

Browser monitoring is not mature, so monitoring is the terminal log, fed by a polled source rather than by the
bookmaker's page. The browser stays, headless, to navigate, explore and place.

## What Changes

- Add `execute_event(event, bettor, dataloader, session, *, stake, urls, live=False, placer=None, poll=30s,
  bound=None, clock=None, wait=None)` to `sportsbet.execution`. It explores the candidate URLs to match the event,
  ensures the session is logged in, polls the dataloader and logs the event to its betting moment, applies the fitted
  bettor at the moment, and places one bet of the configured stake when the model finds value and the run is armed.
- A run is a no-stakes dry run unless armed with `live=True` or `--live`. Armed, it places with no second confirmation.
- The bettor decides whether to bet and the selection. The stake is an execution parameter.
- Add a default placer that enters the stake into the pinned `stake` control and clicks the pinned `confirm` control of
  the matched `FixedSession`. A custom `placer(intent, session)` overrides it from Python.
- Repurpose `sportsbet execution run` and the MCP tool `execution_run` for the single-event unit, with `--venue`,
  `--dataloader`, `--bettor`, `--event`, `--stake`, repeatable `--url`, `--live`, `--poll` and `--output`.
- The browser runs headless with no live-preview window. All monitoring information goes to the terminal.
- **BREAKING**: remove `quote`, `place` and `build_value_bet_intents` from `sportsbet.execution`, the CLI commands
  `execution quote` and `execution place`, and the MCP tools `execution_quote` and `execution_place`.
- **BREAKING**: remove the batch `execute` and `select_feasible` from `sportsbet.execution`, and the batch behaviour of
  `execution run` and `execution_run`. These batch names were added after feature 006 by commits that no archived
  feature specifies, so this change records their removal here and in the changelog rather than as spec deltas.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `execution/event-run`: the batch quote, confirm, ceilings, kill switch and sequential placement requirements are
  removed. The single-event unit is added. The minimum price applies to the one bet.
- `execution/venue-contract`: the browser session runs headless with no live-preview window.
- `interfaces/cli`: the `execution quote` and `execution place` commands are removed and the single-event
  `execution run` command is added.
- `interfaces/mcp-server`: the `execution_quote` and `execution_place` tools are removed and the single-event
  `execution_run` tool is added.

## Impact

- Code: new `src/sportsbet/execution/_event.py` with `execute_event` and the default placer. `_place.py` is deleted.
  `_schedule.py` keeps `find_betting_moment`, the `Clock`, `Wait`, `Placer`, `Staking` and `Scheduled` aliases and
  `TOLERANCE`. `execution/__init__.py`, `cli/_execution.py` and `mcp/_server.py` drop the batch names and gain the
  single-event command and tool.
- API: `execute_event` is new. `execute`, `select_feasible`, `quote`, `place` and `build_value_bet_intents` are
  removed. `BrowserSession`, `FixedSession`, `PageSnapshot`, `BetIdentity`, `PlacementIntent`, `PlacementReceipt`,
  `PlacementReceiptSchema`, `PlacementStatus`, `ExposureLimits`, `build_receipts_frame`, `find_betting_moment`,
  `build_venue`, `CredentialRef`, `resolve`, `Placer` and the error family stay.
- Dependencies: none new. The browser stays behind the optional `execution` extra.
- Tests: `tests/execution/test_event.py` and fakes in `tests/execution/conftest.py`. The batch tests are deleted. The
  parity test in `tests/mcp/test_parity_surfaces.py` loses the quote and place pairs and keeps the run pair with the
  new parameters.
- Docs: `docs/examples/execution/plot_place_value_bets.py` and `docs/overview/user_guide/execution.md` are rewritten for
  the single-event flow, and `docs/generated` is regenerated.
- Changelog: the removed public names are listed under a `BREAKING CHANGE` footer of the conventional commit.
- Requirements checklist: every item passed. The two open questions, the real-money gate and the live-data supply,
  were resolved with the user on 2026-07-27.
