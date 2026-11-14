# Spec Delta

## MODIFIED Requirements

### Requirement: execution_run
`execution_run` SHALL take `venue`, `dataloader`, `bettor`, `event`, `stake`, `ledger`, `lead` (default `5min`), `poll`
(default `5min`), `live` (default false) and an optional `output` directory. It SHALL return `outcome`, `moment`,
`steps` and `receipts`, and with `output` write `run_steps.csv` and `receipts.csv` under
`<output>/sports-betting-data/`.

#### Scenario: A dry run returns the bet it would make
- **WHEN** a client calls `execution_run` with `live=False` and the bettor backs a market
- **THEN** the result's `outcome` is `dry_run` and its `receipts` hold one record with `status` `dry_run`, staking
  nothing

### Requirement: Browser tools
`browser_navigate(venue, url)`, `browser_snapshot(venue, selector, depth)`, `browser_click(venue, ref)`,
`browser_type(venue, ref, text)` and `browser_select(venue, ref, value)` SHALL return an object with the page snapshot
`yaml` and the page `url`. The server MUST NOT offer a tool that pins controls. The browser session is specified in
execution/venue-contract.

#### Scenario: A ref from a snapshot
- **WHEN** a client calls `browser_navigate` and then `browser_click` with a ref from the returned snapshot
- **THEN** `browser_click` returns the snapshot of the page the click produced

#### Scenario: No pinning tool
- **WHEN** a client lists the server's tools
- **THEN** there is no `browser_fix` tool

### Requirement: Venue kind matches the tool
`execution_authenticate`, `execution_read_balance`, `execution_list_markets`, `execution_read_status`,
`execution_cancel` and `execution_run` MUST be given a venue, a browser venue included. The browser tools MUST be given
a browser session or a browser venue, using its session. A reference of another kind SHALL fail with an
`ExecutionError` message.

#### Scenario: A browser session given to a venue tool
- **WHEN** a client calls `execution_read_balance` with a reference to a bare `BrowserSession`
- **THEN** the call fails with a message that says it is a browser session and points to the browser tools
