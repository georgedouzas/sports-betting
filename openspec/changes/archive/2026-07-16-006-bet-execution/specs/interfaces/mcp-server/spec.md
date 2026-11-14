## ADDED Requirements

### Requirement: Execution venue tools
The MCP server SHALL expose `execution_venue_info`, `execution_authenticate`, `execution_list_markets`,
`execution_read_balance`, `execution_read_status` and `execution_cancel`, each taking a `venue` reference resolved by
`build_venue`. `execution_venue_info` SHALL return the venue's `key`, `url`, `notes` and `can_cancel` verbatim. No tool
SHALL take a credential. DataFrames SHALL cross as records.

#### Scenario: Venue notes reach the agent
- **WHEN** an agent calls `execution_venue_info` with a venue reference
- **THEN** the result holds the venue's key, URL, notes and `can_cancel` exactly as configured

### Requirement: Execution quote and place tools
The MCP server SHALL expose `execution_quote` and `execution_place`. `execution_place` MUST enforce the confirmation in
code. Without `confirm_stake` and `confirm_exposure` it SHALL return refusals stating the real figures. With figures
that differ from the quote it SHALL refuse and state the real figures. Only exact figures SHALL place.

#### Scenario: An agent places without confirmation
- **WHEN** an agent calls `execution_place` without confirmations
- **THEN** the tool stakes nothing and returns the real stake and exposure

#### Scenario: An agent places with exact figures
- **WHEN** an agent calls `execution_place` with the exact quoted stake and exposure
- **THEN** the tool places each quoted bet once against the venue

### Requirement: Browser tools
The MCP server SHALL expose `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_select`
and `browser_fix`, acting on one browser context held for the server's lifetime. The tools SHALL carry no site
knowledge.

#### Scenario: A login survives between tool calls
- **GIVEN** an agent logged in to a bookmaker through the browser tools
- **WHEN** it calls `browser_navigate` in a later tool call
- **THEN** the page is reached in the same authenticated session

### Requirement: Execution tools without the extras
The execution and browser tools need both the `mcp` and `execution` extras. When `execution` is missing, each tool
SHALL return a message naming the extra to install and MUST NOT raise an `ImportError`.

#### Scenario: Calling an execution tool with only the mcp extra
- **WHEN** an agent calls `execution_venue_info` in an environment with `mcp` but not `execution`
- **THEN** the result names the `execution` extra to install

### Requirement: Execution tool parity
Every `sportsbet execution` command SHALL have a matching MCP tool that can be told what the command can be told, and
every execution tool SHALL map to a Python API capability.

#### Scenario: Each command has a tool
- **WHEN** the parity test compares the execution commands with the MCP tools
- **THEN** each command maps to a tool with the same capability
