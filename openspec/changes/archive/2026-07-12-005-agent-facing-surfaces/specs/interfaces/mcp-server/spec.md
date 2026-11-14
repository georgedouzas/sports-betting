## Purpose

The `sportsbet-mcp` server lets an assistant drive the library through MCP tools that read the same configuration
module as the command line, and it never spends on metered data without a confirmed cost.

## ADDED Requirements

### Requirement: Offer the library as MCP tools
The `sportsbet-mcp` server SHALL expose the tools `available_params`, `estimate_preparation`, `prepare`,
`extract_train_data`, `extract_fixtures_data`, `backtest` and `bet`. An assistant SHALL be able to go from discovering
data to value bets with these tools and no code written by the user.

#### Scenario: From discovery to value bets
- **GIVEN** an assistant connected to `sportsbet-mcp` and a configuration path
- **WHEN** it calls `available_params`, `estimate_preparation`, `prepare`, `extract_train_data`, `backtest` and `bet`
- **THEN** it receives the available parameters, the cost, the prepared data, the training data, the backtest results
  and the value bets

### Requirement: Take a configuration path
Every tool SHALL take the path of a Python configuration module of the same form the `sportsbet` command line reads,
with `DATALOADER` and optionally `BETTOR`. A credential SHALL come from the environment through the configuration and
MUST NOT be a tool argument.

#### Scenario: A credentialled source through a configuration
- **GIVEN** a configuration that builds `OddsApi(key=os.environ['ODDS_API_KEY'])`
- **WHEN** the assistant calls a tool with the path of that configuration
- **THEN** the tool uses the key and no tool argument carries it

#### Scenario: Every sport is reachable
- **GIVEN** a configuration for a basketball league
- **WHEN** the assistant calls the tools with its path
- **THEN** the tools work as they do for soccer

### Requirement: Discover data without downloading
`available_params` SHALL return the leagues and seasons the configured sources publish, and it SHALL download no data.

#### Scenario: Ask what exists
- **WHEN** the assistant calls `available_params` with a configuration path
- **THEN** it receives the leagues and seasons and no data is downloaded

### Requirement: Estimate the cost of a preparation
`estimate_preparation` SHALL return what a preparation would fetch, what is already held, and the exact cost per
metered source. It SHALL spend nothing.

#### Scenario: Price an NBA season
- **GIVEN** a configuration for one NBA season with `OddsApi`
- **WHEN** the assistant calls `estimate_preparation`
- **THEN** it receives the items to fetch, the items held and the cost, for example `{'odds_api': 17722}`, and no
  credits are spent

#### Scenario: A league whose odds nobody sells
- **GIVEN** a configuration whose selection has no odds from the configured sources
- **WHEN** the assistant asks for the data
- **THEN** it is told that no odds exist for the selection rather than handed an empty dataset

### Requirement: Confirm the cost before a metered preparation
`prepare` SHALL refuse a preparation whose total cost is above zero unless the caller passes `confirm_cost` equal to the
total cost `estimate_preparation` quotes. A refusal for a wrong value MUST name the real cost. A preparation whose total
cost is zero SHALL run with no confirmation.

#### Scenario: No confirmation
- **WHEN** the assistant calls `prepare` on a metered configuration without `confirm_cost`
- **THEN** the tool refuses and spends nothing

#### Scenario: Wrong confirmation
- **WHEN** the assistant calls `prepare` with `confirm_cost=100` and the real cost is 17722
- **THEN** the tool refuses, spends nothing, and the message names 17722

#### Scenario: Right confirmation
- **WHEN** the assistant calls `prepare` with `confirm_cost=17722` and the real cost is 17722
- **THEN** the preparation runs

#### Scenario: Free preparation
- **GIVEN** every source is free or every item is already held
- **WHEN** the assistant calls `prepare` without `confirm_cost`
- **THEN** the preparation runs

### Requirement: Train, backtest and bet
`extract_train_data` and `extract_fixtures_data` SHALL return the extracted data, `backtest` SHALL backtest the
configuration's `BETTOR`, and `bet` SHALL return the value bets for the fixtures. Each result SHALL be JSON records that
round-trip to the data the Python API returns.

#### Scenario: Backtest and bet on prepared data
- **GIVEN** prepared data and a configuration with a `BETTOR`
- **WHEN** the assistant calls `backtest` and then `bet`
- **THEN** it receives the backtest results and the value bets as JSON records

### Requirement: Reach every command line capability
The MCP server SHALL offer every capability the `sportsbet` command line offers, through the same configuration
contract, so the two surfaces reach the same sports and sources.

#### Scenario: Look for a command line capability
- **GIVEN** a capability the command line offers
- **WHEN** the assistant looks for it among the tools
- **THEN** a tool offers it

### Requirement: Install the server as an optional extra
The server SHALL ship as the optional `mcp` extra with the `sportsbet-mcp` entry point. A user who does not install the
extra MUST NOT need any MCP dependency. The server MUST be covered by the project's tests.

#### Scenario: Install with the extra
- **WHEN** the user runs `pip install 'sportsbet[mcp]'`
- **THEN** the `sportsbet-mcp` command is available

#### Scenario: Install without the extra
- **WHEN** the user runs `pip install sportsbet`
- **THEN** no MCP dependency is installed and the library works

### Requirement: Keep models outside the package
The library MUST NOT add a model, a model credential, or a call to a model. The assistant is a consumer of the library
and lives outside the package.

#### Scenario: Inspect the dependencies
- **WHEN** a user inspects the package dependencies and source
- **THEN** they find no model dependency, no model credential and no model call
