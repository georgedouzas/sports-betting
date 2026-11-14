# Spec Delta

## MODIFIED Requirements

### Requirement: Event must be a fixture

A run SHALL act on exactly one event, named as `Home vs Away`, with one fitted bettor, one venue and one fixed stake.
The event MUST be among the dataloader's fixtures when the run starts. The run MUST NOT place a bet on any other event.

#### Scenario: Unknown event

- **WHEN** a run starts for an event that is not among the dataloader's fixtures
- **THEN** it raises an execution error that names the event, and contacts no venue

#### Scenario: Other fixtures are ignored

- **WHEN** the dataloader's fixtures hold several matches and the run names one of them
- **THEN** the run decides and places only on the named match

### Requirement: Opening the site

Before it waits for the moment, a run SHALL authenticate at the venue once. If authentication fails or the venue
blocks access, the run MUST stop, log the reason, and place nothing.

#### Scenario: Authentication fails

- **WHEN** the venue refuses authentication at the start of the run
- **THEN** the run stops before waiting, logs the failure, places nothing, and returns a run record whose outcome says
  so

### Requirement: Betting moment

The betting moment SHALL be the event's kickoff minus a lead time. The lead time is a run parameter with a default of
five minutes, and it MUST NOT be negative. A dataloader fitted for an in-play moment MUST be refused before any venue
is contacted, since a run supports preplay models only.

#### Scenario: Preplay model

- **WHEN** a run starts for an event with kickoff at 15:00 and no lead time is given
- **THEN** the betting moment is 14:55

#### Scenario: In-play model

- **WHEN** a run starts with a dataloader fitted with `target_event_status='inplay'`
- **THEN** it raises an execution error saying in-play execution is not supported, and contacts no venue

#### Scenario: Negative lead time

- **WHEN** a run is given a lead time of minus one minute
- **THEN** it raises an execution error and contacts no venue

### Requirement: Moment already passed

When the current time is after the betting moment at the start of the run, the run SHALL place nothing, log that the
moment has passed, and return a run record whose outcome says so.

#### Scenario: Started too late

- **WHEN** the run starts after its betting moment
- **THEN** it places nothing and its run record's outcome is `moment_passed`

### Requirement: Logging while waiting

While it waits, a run SHALL re-read the event's data from the dataloader on each poll and log the event, its status,
the time left to the moment, and the current price of each betting market where one is available.

#### Scenario: Poll log line

- **WHEN** the run waits with the default five minute poll and twelve minutes to the moment
- **THEN** it logs one line per poll with the event, its status, the time left and the prices, and waits five minutes
  between polls, then the two minutes left

#### Scenario: Price changes between polls

- **WHEN** the source price for a betting market changes between two polls
- **THEN** each poll's log line shows the price read at that poll

#### Scenario: Event leaves the fixtures

- **WHEN** a poll finds the event is no longer among the fixtures, for example because it was postponed
- **THEN** the run stops, logs that the event is no longer listed, and places nothing

### Requirement: Decision

At the betting moment a run SHALL read the event's data again and apply the bettor to that data. If the read fails,
the run MUST place nothing and log why. When the bettor backs no market, the run MUST place nothing and say so.

#### Scenario: Value bet found

- **WHEN** the bettor backs `home_win` on `Arsenal vs Chelsea` in the data read at the moment
- **THEN** the intended bet backs `Arsenal` on `home_win` with the configured stake

#### Scenario: Decision on fresh data

- **WHEN** the moment arrives and the source now carries different odds from those at the start of the run
- **THEN** the bettor is applied to the odds read at the moment

#### Scenario: Source unreachable at the moment

- **WHEN** reading the event's data at the moment fails
- **THEN** the run places nothing, logs the failure, and its run record's outcome is `data_failed`

#### Scenario: No value

- **WHEN** the bettor backs no market on the event at the moment
- **THEN** the run places nothing, logs that no value bet was found, and its run record holds no receipt

### Requirement: Dry run by default

A run SHALL be a dry run unless it is armed up front. A dry run reads the venue's current price for the intended
selection and records the bet it would place with that price and the configured stake. It MUST NOT enter a stake or
confirm a bet.

#### Scenario: Dry run with value

- **WHEN** an unarmed run reaches the moment, the bettor backs a market, and the venue offers a price for it
- **THEN** the run records one `dry_run` receipt with the venue's price and the configured stake, and nothing is staked

#### Scenario: Dry run with a worse venue price

- **WHEN** an unarmed run finds the venue's price below the minimum acceptable price
- **THEN** the run records one `refused_price` receipt and logs both prices

### Requirement: Placing an armed bet

An armed run SHALL place the intended bet at the moment through the venue, once, with no further confirmation and for
exactly the configured stake. Before it places, it MUST log the event, market, selection, stake, venue and minimum
price.

#### Scenario: Armed run with value

- **WHEN** an armed run reaches the moment and the bettor backs a market
- **THEN** the run logs the intended bet, then asks the venue to place it once with the configured stake, and returns
  the venue's receipt in its run record

#### Scenario: Custom placer

- **WHEN** a caller needs a site-specific way to place a bet
- **THEN** they implement it in their venue, since the run takes no placer and places only through the venue

### Requirement: Surfaces

The run SHALL be reachable from the Python API, the `sportsbet execution run` command and the `execution_run` MCP
tool with the same parameters. The command and the tool MUST write the run record's steps and receipt to an output
directory when given one.

#### Scenario: Parity of parameters

- **WHEN** the parity tests compare the Python run, the command and the tool
- **THEN** each takes the venue, dataloader, bettor, event, stake, lead time, poll interval, ledger path, the arming
  flag and the output directory

#### Scenario: Output written

- **WHEN** `sportsbet execution run` finishes with `--output out`
- **THEN** the steps are written to `out/sports-betting-data/run_steps.csv` and the receipt to
  `out/sports-betting-data/receipts.csv`

## ADDED Requirements

### Requirement: Market maps to its selection

The decision SHALL map the backed market to the side it backs. `home_win` backs the home team. `away_win` backs the
away team. `draw` backs `Draw`. An over or under market backs that line, as in `Over 2.5`.

#### Scenario: Away win backed

- **WHEN** the bettor backs `away_win` on `Arsenal vs Chelsea`
- **THEN** the intended bet's selection is `Chelsea`

#### Scenario: Over market backed

- **WHEN** the bettor backs `over_2.5`
- **THEN** the intended bet's selection is `Over 2.5`

### Requirement: One market when several are backed

When the bettor backs several markets on the event, the decision SHALL take the first of them in the bettor's market
order and log the others as not placed.

#### Scenario: Two markets backed

- **WHEN** the bettor's market order is `home_win, draw, away_win, over_2.5` and it backs `away_win` and `over_2.5`
- **THEN** the intended bet is on `away_win` and the log names `over_2.5` as not placed

### Requirement: Minimum price is the price the model decided on

The intended bet's minimum acceptable price SHALL be the source's latest price for the backed market in the data the
bettor was applied to. When the source has no price for that market, the run MUST place nothing and log why.

#### Scenario: No source price

- **WHEN** the bettor backs a market for which the source carries no price
- **THEN** the run places nothing and logs that no price was available

### Requirement: At most one bet across processes

A run SHALL keep a ledger file at a path the caller gives. It records the bet as attempted before asking the venue to
place it, and records the receipt after. A run that finds its bet in the ledger MUST NOT place it again.

#### Scenario: Rerun after a placed bet

- **WHEN** a second armed run starts for an event whose bet the ledger holds as placed
- **THEN** it places nothing and records an `already_placed` receipt

#### Scenario: Rerun after a crash mid-placement

- **WHEN** an armed run starts and the ledger holds its bet as attempted with no receipt
- **THEN** it places nothing, records a `refused_unconfirmed` receipt, and tells the user to check the venue for the
  bet

#### Scenario: Dry runs do not block a live run

- **WHEN** the ledger holds only `dry_run` receipts for a bet and an armed run starts for it
- **THEN** the armed run places normally

### Requirement: Run record

A run SHALL return a run record holding the event, the betting moment, every step it went through with its time and
detail, the outcome, and the receipt when there is one. The steps and the receipt MUST be validated frames.

#### Scenario: Record of an armed run

- **WHEN** an armed run places a bet
- **THEN** its run record lists the start, the venue check, each poll, the decision, the placement and the outcome
  in order, and holds the venue's receipt

### Requirement: The browser is not held open while waiting

A run SHALL release any venue resources it opened for the up-front check before it waits, and open them again only
for the decision at the moment. It MUST release them when the run ends, whatever the outcome.

#### Scenario: Run ends with an error

- **WHEN** placing raises an error
- **THEN** the run releases the venue's resources before the error propagates

## REMOVED Requirements

### Requirement: Finding the event page

**Reason**: Matching the event by searching candidate pages for the literal `Home vs Away` text fails on real sites,
which format and translate team names. Finding the event belongs to the venue, which knows its own site.

**Migration**: Drop `urls` and `--url`. A browser venue implements finding the event page in its `BaseBrowserVenue`
subclass. An API venue looks the event up through its API inside `read_price` and `place`.
