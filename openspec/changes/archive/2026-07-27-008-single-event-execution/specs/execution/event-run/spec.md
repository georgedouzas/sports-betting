## ADDED Requirements

### Requirement: One event per unit
`execute_event` SHALL act on exactly one betting event, named by a single match as `"Home vs Away"`, with exactly one
fitted bettor, and MUST NOT place a bet on any other event. It MUST NOT orchestrate more than one event. Extending to
many events SHALL be left to the user.

#### Scenario: The unit takes one event and one bettor
- **WHEN** a unit is given its inputs
- **THEN** it accepts exactly one event and one fitted bettor

#### Scenario: The event is not a fixture
- **WHEN** the named event is not among the dataloader's fixtures
- **THEN** `execute_event` raises `ExecutionError`

### Requirement: The fitted bettor decides
The unit SHALL use the fitted bettor as the sole decider of whether to bet and which selection to back, by applying
`bettor.bet` to the event's one-row features and odds at the betting moment. The bettor MUST NOT decide the stake.

#### Scenario: A value bet names the selection
- **WHEN** `bettor.bet` returns a value bet for the event at the moment
- **THEN** the unit backs the selection the value bet names

### Requirement: Configured stake
The stake SHALL be a configured execution parameter of the unit, `stake` in Python and `--stake` on the CLI, and the
unit MUST place exactly that amount when it bets.

#### Scenario: The placed stake equals the configured stake
- **WHEN** an armed unit with a stake of 10 places a bet
- **THEN** the receipt shows a stake of 10

### Requirement: Betting moment from the fitted bettor
The unit SHALL derive the betting moment from the bettor's fitted moment through `find_betting_moment`: kickoff for a
preplay model, or kickoff plus the fitted in-play offset for a live model. The unit MUST wait until the moment and only
then apply the bettor to the event's data as of that moment.

#### Scenario: A preplay model bets at the preplay moment
- **GIVEN** a model fitted preplay and an event that finds value
- **WHEN** the preplay moment arrives
- **THEN** the unit places then, within seconds of the moment

#### Scenario: An in-play model waits for its offset
- **GIVEN** a model fitted for a set number of minutes into the match
- **WHEN** the unit runs
- **THEN** it waits until kickoff plus that offset and places then, not before

### Requirement: Fixed operating sequence
The run SHALL follow a fixed order: explore the candidate URLs to match the event, ensure the login, begin the terminal
log, then place at the betting moment. No monitoring or placing SHALL happen before the setup steps succeed.

#### Scenario: Setup precedes monitoring
- **GIVEN** the model, the event, the candidate URLs and the stake
- **WHEN** the unit starts
- **THEN** it explores the URLs, matches the event's page, and prompts the login before any monitoring or placing

#### Scenario: Logging begins after login
- **WHEN** the user has logged in and setup completes
- **THEN** the unit begins logging the event to the terminal

### Requirement: Candidate URLs are matched to the event
The unit SHALL take one or more candidate bookmaker URLs and, during setup, navigate each headless to find the page
that names the event and the selection, then pin the site's controls as a `FixedSession`. When no URL matches, the
unit MUST stop, log it, and place nothing.

#### Scenario: No URL matches
- **WHEN** none of the candidate URLs carries the event
- **THEN** the unit stops before monitoring, logs it, and returns an empty receipts frame

### Requirement: Login before monitoring
The unit SHALL ensure the browser session is logged in to the bookmaker before monitoring and placing, and SHALL prompt
the user to log in during setup when the session is not authenticated. When login is not in place within the bound or
authentication fails, the unit MUST stop, log the failure, and place nothing.

#### Scenario: Login not completed
- **WHEN** the user does not complete the login within the bound
- **THEN** the unit stops before monitoring, logs it, and stakes nothing

#### Scenario: Authentication fails
- **WHEN** authentication to the venue fails
- **THEN** the unit stops before monitoring, logs the failure, and stakes nothing

### Requirement: Self-polling monitor
The unit SHALL obtain the event's evolving data by polling the configured dataloader every `poll` interval from the
start of the run until postplay, so the caller writes no monitoring loop. It SHALL derive the status from the event's
timing and status field: preplay before kickoff, inplay until the end, postplay after. The bookmaker's page MUST NOT be
the source of the decision data.

#### Scenario: The data supply drops
- **WHEN** the live data supply drops mid-monitor
- **THEN** the unit logs the gap and does not place a bet on stale data

#### Scenario: The event never advances
- **WHEN** the event is postponed or abandoned and its status never advances
- **THEN** the unit waits up to the bound, then stops and logs why

### Requirement: Terminal log of the event
The unit SHALL log to the terminal, on the `sportsbet.execution` logger, the event identity, the current status, the
current price where available, and its decision at each step: waiting, moment reached, bet placed or declined, and the
outcome. Monitoring SHALL be this terminal log, fed by the polled source.

#### Scenario: Status transitions are logged
- **WHEN** the event's status changes from preplay to inplay to postplay
- **THEN** each transition is logged with the time and the current price

#### Scenario: The end of the run is logged
- **WHEN** the event reaches postplay and monitoring ends
- **THEN** the unit logs the final status and the outcome of its bet if the outcome is known

### Requirement: Announce before placing
Before placing, the unit MUST log the selection, the stake and the price it is about to stake, so the action is legible
before money moves.

#### Scenario: The bet is announced first
- **WHEN** the betting moment arrives and the unit is about to place
- **THEN** it logs the selection, stake and price before it places

### Requirement: One bet when the model finds value
When the fitted bettor returns a value bet at the moment, an armed unit SHALL place exactly one bet, for that selection
at the configured stake, by calling the placer once. The default placer SHALL enter the stake into the pinned `stake`
control and click the pinned `confirm` control. A custom `placer(intent, session)` SHALL override it from Python.

#### Scenario: An armed run places one bet
- **GIVEN** a fitted preplay model that finds value on the event
- **WHEN** the preplay moment arrives in an armed run
- **THEN** the unit places the configured stake on the model's selection once and places no other bet

#### Scenario: A custom placer is called once
- **WHEN** an armed `execute_event` call is given a custom placer and the model finds value
- **THEN** the custom placer is invoked once at the moment

### Requirement: No bet without value
When the fitted bettor returns no value bet at the moment, or the venue offers no market or price for the selection,
the unit MUST place nothing and MUST log it, without failing.

#### Scenario: The model finds no value
- **WHEN** the moment arrives and the model finds no value on the event
- **THEN** the unit places nothing and logs that no bet was found

#### Scenario: No market for the selection
- **WHEN** the venue offers no market or price for the model's selection at the moment
- **THEN** the unit places nothing and logs it

### Requirement: At most one bet per run
The unit MUST place at most one bet on the event over its entire run, regardless of later price movement or later
polls.

#### Scenario: No second bet after acting
- **GIVEN** the unit has placed or declined its one bet
- **WHEN** the event continues and the price moves
- **THEN** the unit places no further bet

### Requirement: Dry run unless armed
A run SHALL be a no-stakes dry run unless the user arms it up front with `live=True` or `--live`. A dry run MUST
monitor, decide and log the bet it would have placed, and MUST NOT call the placer. An armed run SHALL place at the
moment with no further confirmation.

#### Scenario: A dry run stakes nothing
- **WHEN** the unit runs without `live`
- **THEN** it logs the bet it would have placed and stakes nothing

#### Scenario: Re-running armed
- **GIVEN** the user observed a full dry run
- **WHEN** the user re-runs the same unit with stakes enabled
- **THEN** the unit places for real at the moment

### Requirement: The moment has already passed
When the betting moment has already passed as the unit starts, because the event is in-play past the fitted moment or
finished, the unit MUST stop, log that the moment is unreachable, and place nothing.

#### Scenario: Starting after the moment
- **WHEN** the unit starts after the fitted moment
- **THEN** it logs that the moment is unreachable and returns an empty receipts frame

### Requirement: Run record
`execute_event` SHALL return a receipts frame built by `build_receipts_frame` and validated by
`PlacementReceiptSchema`, holding the placed bet's receipt or the dry-run receipt if a bet was decided, and no rows
otherwise. The unit MUST reuse the existing receipt shape, bettor, data supply and browser session.

#### Scenario: A placed bet appears in the record
- **WHEN** an armed run places a bet
- **THEN** the returned frame holds exactly one receipt for the model's selection at the configured stake

## MODIFIED Requirements

### Requirement: Minimum acceptable price
The one intent the unit builds SHALL carry a minimum acceptable price, the price of the selection in the event's odds
at the moment. When the price at placing is worse than that minimum, the unit MUST decline, log it, and place nothing.

#### Scenario: The price moved below the floor
- **WHEN** the price at placing is worse than the model's minimum acceptable price
- **THEN** the unit declines, logs it, and places nothing

#### Scenario: The default floor
- **WHEN** the unit builds its one intent at the moment
- **THEN** the minimum price is the selection's price in the event's odds at that moment

## REMOVED Requirements

### Requirement: Value bets become placement intents
**Reason**: The multi-match batch path is retired. `build_value_bet_intents` is removed, and the single-event unit
builds the one intent it needs from the bettor's decision at the moment.
**Migration**: Call `execute_event` once per event with the fitted bettor, the dataloader and the configured stake.

### Requirement: Quote before placing
**Reason**: The quote-and-confirm model is gone. `quote` is removed. A unit places one configured stake and announces
it in the terminal log before placing.
**Migration**: Run `execute_event` without `live`, or `sportsbet execution run` without `--live`, to see the bet the
unit would place without staking.

### Requirement: Placement requires the exact quoted figures
**Reason**: `place` and its `confirm_stake` and `confirm_exposure` gate are removed. A run is armed once up front and
then places at the moment with no second confirmation.
**Migration**: Pass `live=True` to `execute_event`, or `--live` to `sportsbet execution run`, to enable stakes.

### Requirement: Stake and exposure ceilings
**Reason**: The batch `place` entry point that enforced `max_stake_per_bet` and `max_total_exposure` is removed. A
unit places at most one bet of the configured stake, which bounds its exposure.
**Migration**: Set `stake` to the amount to risk on the event. `ExposureLimits` stays importable but no runner reads it.

### Requirement: Kill switch
**Reason**: `ExposureLimits.killed` stopped a batch in flight. The unit places at most one bet, so there is no batch to
stop.
**Migration**: Leave the run unarmed, or stop the process before the betting moment.

### Requirement: Sequential batch placement
**Reason**: There is no batch. A unit acts on one event and places at most one bet.
**Migration**: Run one unit per event. Running many units at once is the user's responsibility.
