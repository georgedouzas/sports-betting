## Purpose

Turn the value bets a fitted bettor produces into bets placed at a venue, refusing by default and placing only what the
caller confirmed, within the limits the user set.

## ADDED Requirements

### Requirement: Value bets become placement intents
Execution SHALL consume the value bets a bettor produces, together with the identity of the matches they refer to, and
SHALL build one `PlacementIntent` per value bet the caller acts on. The stake SHALL be an input from the caller and
never a model output. The user SHALL NOT need to re-enter any bet by hand.

#### Scenario: Value bets to intents
- **GIVEN** a fitted bettor and the fixtures of an upcoming round
- **WHEN** the caller hands the value bets to execution
- **THEN** each value bet becomes an intent carrying its identity, stake, minimum price and value bet reference

### Requirement: Quote before placing
`quote(venue, intents, limits)` SHALL return a `PlacementQuote` that itemises every bet with its stake and price, and
states the total stake of the batch and the total exposure, which is the batch stake plus the exposure already open at
the venue. The system MUST report this before any money moves.

#### Scenario: The batch is visible before money moves
- **WHEN** the caller quotes a batch
- **THEN** the quote lists every intended bet with its stake and price and the batch totals
- **AND** every later receipt traces to a quoted row

### Requirement: Placement requires the exact quoted figures
`place(venue, quote, limits, confirm_stake=None, confirm_exposure=None)` MUST NOT place a real bet unless the caller
passes back the exact total stake and total exposure of the quote. Without confirmation every receipt SHALL be
`DRY_RUN`. With figures that differ every receipt SHALL be `REFUSED_UNCONFIRMED` and the message MUST state the real
figures. Both cases stake zero.

#### Scenario: The default places nothing
- **GIVEN** a set of value bets and a configured venue
- **WHEN** the caller places without confirmation
- **THEN** every receipt is `DRY_RUN`, nothing is staked, and the venue records no placement call

#### Scenario: Exact figures place each bet once
- **GIVEN** the quote just returned
- **WHEN** the caller passes back its exact total stake and total exposure
- **THEN** each intended bet is placed exactly once and a receipt is returned for each

#### Scenario: Mismatched figures refuse
- **WHEN** the confirmed stake, the confirmed exposure, or both differ from the quote
- **THEN** every receipt is `REFUSED_UNCONFIRMED`, nothing is placed, and the message states the real figures

### Requirement: Stake and exposure ceilings
The system SHALL enforce `max_stake_per_bet` and `max_total_exposure` from `ExposureLimits` and MUST refuse a
placement that would exceed either with `REFUSED_LIMIT`, naming the ceiling reached in `detail`.

#### Scenario: A stake above the ceiling
- **WHEN** an intent's stake exceeds `max_stake_per_bet` and the caller confirms
- **THEN** that receipt is `REFUSED_LIMIT` and its `detail` names the stake ceiling

#### Scenario: Exposure reached partway through a batch
- **WHEN** the running exposure reaches `max_total_exposure` partway through a confirmed batch
- **THEN** the bets placed before it stand and the remainder are `REFUSED_LIMIT` naming the exposure ceiling

### Requirement: Kill switch
`ExposureLimits.killed` SHALL stop further placement. It MUST be checked before the batch and before every stake.

#### Scenario: Engaged before a batch
- **WHEN** the kill switch is set before placing
- **THEN** every receipt is `REFUSED_KILLED`

#### Scenario: Engaged mid-batch
- **WHEN** the kill switch is set while a batch is in flight
- **THEN** the bets already placed stand and the remainder are `REFUSED_KILLED`

### Requirement: Minimum acceptable price
Every intent SHALL carry a minimum acceptable price, defaulting to the price the value bet was computed at, and the
caller SHALL be able to set a different minimum. The system MUST refuse to place below it with `REFUSED_PRICE` and a
stake of zero.

#### Scenario: The price moved below the floor
- **WHEN** the venue price at placing is below the intent's `min_price`
- **THEN** the receipt is `REFUSED_PRICE` and nothing is staked

#### Scenario: The default floor
- **WHEN** the caller sets no minimum price
- **THEN** the minimum is the price the value bet was computed at

### Requirement: Sequential batch placement
A batch SHALL place one bet at a time, checking the kill switch, the ceilings against the running exposure, and the
minimum price before each stake, in that order. A configurable minimum interval in seconds SHALL separate venue calls.

#### Scenario: Calls arrive in order
- **WHEN** a confirmed batch is placed
- **THEN** the venue receives the placements strictly in order, one at a time, with the minimum interval between them
