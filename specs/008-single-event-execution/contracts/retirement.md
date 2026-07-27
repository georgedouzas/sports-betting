# Contract: Retirement Ledger

The batch quote-and-confirm path is removed. Each removal is a public-API change recorded here and in the changelog.

## Removed

| Symbol | Where | Replaced by |
|--------|-------|-------------|
| `execute` | `execution._schedule` | `execute_event` |
| `select_feasible` | `execution._schedule` | folded into the single-event monitor |
| `quote` | `execution._place` | none, the quote-and-confirm model is gone |
| `place` | `execution._place` | `execute_event` places directly at the moment |
| `build_value_bet_intents` | `execution._place` | the runner builds the one intent it needs |
| `execution run` (batch) | `cli._execution` | `execution run` (single-event) |
| `execution quote`, `execution place` | `cli._execution` | none |
| `execution_run` (batch) | `mcp._server` | `execution_run` (single-event) |
| `execution_quote`, `execution_place` | `mcp._server` | none |

## Kept and reused

| Symbol | Where | Role in the feature |
|--------|-------|---------------------|
| `BrowserSession`, `FixedSession`, `PageSnapshot` | `execution._browser` | the headless browser, matching, placing |
| `find_betting_moment` | moved to live with the runner | the moment to wait for |
| `BetIdentity`, `PlacementIntent`, `PlacementReceipt` | `execution._base` | the one bet and its receipt |
| `PlacementStatus`, `PlacementReceiptSchema`, `build_receipts_frame` | `execution._base` | the run record |
| `ExposureLimits` | `execution._base` | still available, the single stake is the primary control |
| `Placer` | moved to live with the runner | how a bet is placed on a site |
| `build_venue`, `CredentialRef`, `resolve` | `execution._factory`, `._credentials` | build and authenticate the venue |
| `page` commands, `browser_*` tools | `cli._execution`, `mcp._server` | explore the URLs and pin the controls |

## Migration notes

- `docs/examples/execution/plot_place_value_bets.py` and `docs/overview/user_guide/execution.md` are rewritten for the
  single-event flow. `docs/generated` is regenerated.
- The changelog records the removed public names under a breaking-change footer, generated from the conventional
  commit that lands the retirement.
- Tests for the removed functions are deleted, and `tests/execution/test_event.py` covers the runner.
