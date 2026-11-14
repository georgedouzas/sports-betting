# execution/event-run Specification

## Purpose

Runs one betting event with a fitted bettor through a browser session: it finds the event's page, waits for the
betting moment while logging, decides, and places at most one bet or logs the bet it would place.

## Requirements

### Requirement: Event must be a fixture

`execute_event` SHALL act on one event named `Home vs Away`. When the event is not among the dataloader's fixtures, it
MUST raise `ExecutionError` naming the event.

#### Scenario: Unknown event

- **WHEN** `execute_event` is called for an event that is not among the fixtures
- **THEN** it raises `ExecutionError` naming the event

### Requirement: Finding the event page

The run SHALL navigate each candidate URL in order and keep the first page whose snapshot contains the event name
verbatim. When none does, it MUST log that and return an empty receipts frame. Locators pinned on the session are
re-pinned to the event.

#### Scenario: No URL carries the event

- **WHEN** no candidate URL's page contains the event name
- **THEN** the run places nothing and returns an empty receipts frame

### Requirement: Opening the site

After finding the event page, the run SHALL authenticate the session by navigating to the session's site URL. When
that raises `ExecutionError`, including `VenueBlockedError`, it MUST log the failure and return an empty receipts frame.

#### Scenario: Authentication fails

- **WHEN** authentication raises `ExecutionError`
- **THEN** the run places nothing and returns an empty receipts frame

### Requirement: Betting moment

`find_betting_moment` SHALL return the kickoff plus the fitted target event time for a dataloader fitted for an
in-play moment, and the kickoff for any other dataloader.

#### Scenario: Preplay model

- **WHEN** the dataloader was fitted for a preplay moment and the kickoff is 15:00
- **THEN** the betting moment is 15:00

#### Scenario: In-play model

- **WHEN** the dataloader was fitted with `target_event_status='inplay'` and a target event time of 45 minutes
- **THEN** the betting moment is 45 minutes after the kickoff

### Requirement: Moment already passed

When the current time is after the betting moment at the start of the run, the run SHALL log that and return an empty
receipts frame.

#### Scenario: Started too late

- **WHEN** the run starts after its betting moment
- **THEN** it places nothing and returns an empty receipts frame

### Requirement: Logging while waiting

Until the betting moment, the run SHALL log on each poll the event, a status of preplay before kickoff and in-play
after, and the price of the bettor's first betting market in the data read at the start. It waits the poll interval,
or the time left when that is shorter.

#### Scenario: Poll log line

- **WHEN** the run waits with a 30 second poll and five minutes to the moment
- **THEN** it logs one line per poll with the event, its status and the price, and waits 30 seconds between them

### Requirement: Decision

At the moment, the run SHALL apply the bettor to the event's features and odds read at the start of the run and take
the first backed market in the bettor's market order. The selection MUST be the home team. The minimum price MUST be
the source's latest price for the market, or 1.01 when it has none.

#### Scenario: Value bet found

- **WHEN** the bettor backs `home_win` on `Arsenal vs Chelsea`
- **THEN** the intended bet backs `Arsenal` on `home_win` with the configured stake

#### Scenario: No value

- **WHEN** the bettor backs no market
- **THEN** the run logs that no value bet was found and returns an empty receipts frame

### Requirement: Dry run by default

A run SHALL place nothing unless `live` is set. A dry run MUST log the bet it would place and return one `dry_run`
receipt with no stake.

#### Scenario: Dry run with value

- **WHEN** an unarmed run finds a value bet
- **THEN** it returns one `dry_run` receipt and nothing is placed

### Requirement: Placing an armed bet

An armed run SHALL log the stake, selection, market, venue and minimum price, then call the placer once. The default
placer types the stake into the pinned `stake` control, clicks the pinned `confirm` control, and returns a
`matched_full` receipt at the minimum price.

#### Scenario: Armed run with value

- **WHEN** an armed run with a recording placer finds a value bet
- **THEN** the placer is called once and the run returns its one receipt

#### Scenario: Custom placer

- **WHEN** a caller passes a `placer` coroutine
- **THEN** the run calls it with the intent and the session instead of the default placer

### Requirement: Injectable time

The run SHALL take a clock and a wait function, defaulting to the real UTC time and sleeping, so a test can run the
whole wait without real time passing.

#### Scenario: Fake clock

- **WHEN** a test passes a clock and a wait that advances it
- **THEN** the run reaches the moment without sleeping

### Requirement: Surfaces

The run SHALL be reachable as `sportsbet execution run` and the `execution_run` MCP tool, taking the venue reference,
dataloader, bettor, event, stake, candidate URLs, live flag, poll interval and an output directory. The receipts are
written to `receipts.csv` under the output directory.

#### Scenario: Output written

- **WHEN** `sportsbet execution run` places or dry-runs a bet with `--output out`
- **THEN** the receipt is written to `out/sports-betting-data/receipts.csv`
