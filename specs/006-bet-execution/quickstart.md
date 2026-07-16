# Phase 1 Quickstart: Bet execution

**Feature**: [spec.md](./spec.md) | **Contracts**: [venue.md](./contracts/venue.md),
[surfaces.md](./contracts/surfaces.md) | **Date**: 2026-07-16

How to prove this feature works without staking money. No venue anywhere offers a placement
sandbox (research D2), so every scenario below runs against a fake venue, recorded payloads, or a
page served over loopback. That is not a limitation of the test setup, it is the only safe option
that exists.

## Prerequisites

```bash
pdm install -dG maintenance -dG tests -G mcp -G execution
pdm run python -m playwright install --only-shell chromium   # browser scenarios only
```

The browser binary is a separate step, so installing the extra alone does not fetch 93.5 MiB.

## Validation scenarios

Each maps to a success criterion. All are runnable without a bookmaker account.

### 1. The default refuses (SC-002, FR-009)

```bash
sportsbet execution place --venue tests/fake_venue.py:VENUE --quote quote.json
```

Expect the full itemised quote printed, `DRY_RUN` on every receipt, zero stakes, non-zero exit.
Then repeat with `--confirm-stake 999 --confirm-exposure 999` against a quote whose real total is
different. Expect `REFUSED_UNCONFIRMED` and a message naming the real figures. Only exact figures
place anything, and only against the fake.

### 2. Nothing is hidden before money moves (SC-008)

`quote` writes every intent, its stake, its price, and the batch total. Diff the quote against the
receipts: every placed row traces to a quoted row, and there are no unquoted rows.

### 3. Limits refuse and name themselves (SC-009)

Quote a batch above `--max-stake`, then one above `--max-exposure`. Expect `REFUSED_LIMIT` and
`detail` naming which ceiling was hit. Trip the kill switch mid-batch and expect
`REFUSED_KILLED` on the remainder, with the already-placed prefix intact.

### 4. Once and only once, under faults (SC-003, FR-014)

The load-bearing scenario. Against the fake venue, inject a timeout after the venue accepted but
before the response landed, then retry. Inject a crash between accept and receipt, then re-run
from scratch with no state carried over. Both must end with exactly one stake per identity, the
second attempt reporting `ALREADY_PLACED` (FR-014a).

This works because the identity hash is derived, not remembered ([data-model](./data-model.md)):
a fresh process recomputes the same 32-hex ref from the same four fields and finds the bet at the
venue. Assert that no file, cache, or directory holds placement state, since FR-015 forbids a
local record that could drift.

For the Betfair path, assert against recorded payloads that `placeOrders` carries the ref as both
`customerRef` and `customerOrderRef`, and that recovery calls
`listCurrentOrders(customerOrderRefs=[ref])`.

**This scenario covers the API path only, and that is the honest scope.** `BrowserSession` offers
no `place`, so there is nothing here to test: the agent places, and the once-only guarantee is the
agent's ([contracts/venue.md](./contracts/venue.md)). What is testable is that the library claims
nothing it cannot deliver. Assert that `BrowserSession` exposes no `place`, no `read_status` and
no `cancel`, and is not a `BaseVenue`, so a caller cannot reach a guarantee that is not there.

### 5. Credentials stay named (SC-004, FR-017)

Set a credential variable to a sentinel value. Run every command and tool. Grep the full output,
logs, exceptions, written files, and any pickle for the sentinel. Expect zero hits. Assert a
fitted bettor pickle contains no credential and no venue (FR-002). Unset the variable and expect
the run to name the variable it wanted and stop (FR-019).

### 6. The browser refuses what it cannot click (research D4)

Serve a mock bet slip on loopback with a disabled confirm button. `click` on its ref must raise
rather than report success. This is the exact behaviour that decided Playwright over injected
JavaScript, and it is worth a permanent regression test: a false success on a confirm button is a
receipt that lies about money.

Then re-enable the button and confirm the returned snapshot reflects the new state.

### 6b. A fixed session survives a re-render

Serve a page whose odds widget re-renders on a timer, changing every snapshot ref. Explore it,
`fix` the stake and confirm locators, then act after a re-render. The pinned locators must still
resolve, which is the point of pinning roles and accessible names rather than refs. Assert that
`fix` rejects a raw ref, and that no price is stored on the `FixedSession`.

### 7. A blocked venue reports and stops (FR-023)

Serve a page returning a block response. Expect `BLOCKED` on the receipt and placement to stop.
Expect no retry with different timing, headers, or presentation.

### 8. Parity across the three surfaces (SC-006)

`tests/cli/test_parity.py` gains the execution group. Every capability in
[surfaces.md](./contracts/surfaces.md) is reachable from the Python API, the CLI and the MCP
server.

### 9. Without the extra there is nothing to place with (SC-010)

In a clean environment, `pip install sports-betting`. Assert no `playwright` in the tree and that
`sportsbet execution` reports the extra to install rather than raising `ImportError`. Install
`[mcp]` alone and assert the execution tools degrade with a message naming `[execution]`.

### 10. The suite touches no venue (SC-007, FR-026)

The socket guard in `tests/conftest.py` already fails any test reaching the network, permitting
loopback. Run the full suite and confirm zero venue requests and zero real bets. A test that could
stake money is a bug, and there is no sandbox that would make it otherwise.

## The gate

```bash
pdm run formatting
pdm run checks
pdm run tests
```

All three green, every phase. Watch item: the bandit skips `B404`/`B603`/`B607` were removed with
the GUI and the gate is stricter now. Playwright launches its driver internally rather than
through our `subprocess`, so nothing should re-trip them. If something does, report it rather than
re-adding the skip.

## Zero diff on the core (FR-025)

```bash
git diff --stat -- src/sportsbet/dataloaders src/sportsbet/evaluation
```

Must be empty. This is an additive feature. Execution imports the public API and has no reason to
reach inside the core. If the core must change, stop and report why.
