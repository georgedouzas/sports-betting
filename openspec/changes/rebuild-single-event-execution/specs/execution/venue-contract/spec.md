# Spec Delta

## MODIFIED Requirements

### Requirement: Venue contract

A venue SHALL be a `BaseVenue` subclass with a `key` that implements `authenticate`, `read_price` and `place` as
coroutines, and MAY override `close` to release resources. It MUST place a bet once and only once for each bet
identity. An API venue and a browser venue MUST meet the same contract, and a run MUST NOT treat them differently.

#### Scenario: Incomplete venue

- **WHEN** a user instantiates a `BaseVenue` subclass that does not implement `read_price`
- **THEN** Python raises `TypeError` naming the missing method

#### Scenario: Same run, two venues

- **WHEN** the same run is given a conforming API venue and then a conforming browser venue for the same event
- **THEN** both runs go through the same steps and return run records of the same shape

### Requirement: Secrets are named, not passed

A secret SHALL be referenced by the name of the environment variable that holds it, as a `CredentialRef`, and read with
`resolve` where it is used. When the variable is unset or empty, `resolve` MUST raise `CredentialError` naming it. A
secret MUST NOT appear in a receipt, a run record, a ledger or a log line.

#### Scenario: Variable unset

- **WHEN** `resolve(CredentialRef('VENUE_API_KEY'))` runs with `VENUE_API_KEY` unset
- **THEN** it raises `CredentialError` whose message names `VENUE_API_KEY`

#### Scenario: Ledger holds no secret

- **WHEN** an armed run with an API venue that reads a token from a named variable writes its ledger and run record
- **THEN** neither file contains the token's value

### Requirement: Placement statuses

A receipt status SHALL be one of `dry_run`, `accepted`, `matched_full`, `matched_partial`, `already_placed`,
`refused_limit`, `refused_price`, `refused_unconfirmed` or `rejected`.

#### Scenario: Unknown status

- **WHEN** a caller builds a placement status from the string `pending`
- **THEN** Python raises `ValueError`

#### Scenario: Statuses nothing produces are gone

- **WHEN** a caller builds a placement status from `refused_killed` or `blocked`
- **THEN** Python raises `ValueError`, since no run or venue produces either

## ADDED Requirements

### Requirement: Optional venue operations

Listing markets, reading the balance, reading the status of bets and cancelling a bet SHALL be optional for a venue.
A venue that does not provide one MUST raise a named execution error that says the venue does not support it.

#### Scenario: Balance not supported

- **WHEN** a user asks a venue that does not read balances for its balance, from the API, the command or the tool
- **THEN** a named execution error says the venue does not read its balance

#### Scenario: Cancel not supported

- **WHEN** a user asks a venue that cannot cancel to cancel a bet
- **THEN** the cancellation-unsupported error is raised

### Requirement: Price is read without staking

Reading the price for an intended bet SHALL return the price the venue offers for that event, market and selection,
or nothing when the venue does not offer it. It MUST NOT enter a stake or confirm a bet.

#### Scenario: Selection not offered

- **WHEN** the venue does not offer the intended market or selection for the event
- **THEN** reading the price returns nothing, and a run records a `rejected` receipt that says the selection was not
  offered

### Requirement: Placing checks the price first

Placing SHALL read the venue's price for the intended bet before staking. When that price is below the intended
minimum, the venue MUST NOT stake and MUST return a `refused_price` receipt carrying the venue's price.

#### Scenario: Price drifted below the minimum

- **WHEN** an armed run places a bet with minimum price 2.10 and the venue now offers 2.00
- **THEN** nothing is staked and the receipt is `refused_price` with price 2.00

### Requirement: Receipts report only what the venue confirmed

A receipt from placing SHALL carry a status, price and venue bet identifier read from the venue's own response or
confirmation. When the venue gives no confirmation within its timeout, the receipt MUST be `refused_unconfirmed`.

#### Scenario: Confirmation read from the venue

- **WHEN** the venue confirms the bet with its own bet identifier and an accepted price
- **THEN** the receipt carries that identifier and that price, and a placed status

#### Scenario: No confirmation

- **WHEN** the confirm action completes but the venue shows no confirmation within the timeout
- **THEN** the receipt is `refused_unconfirmed` and its detail tells the user to check the venue for the bet

#### Scenario: Venue rejects the bet

- **WHEN** the venue answers that the stake exceeds the account's limit
- **THEN** the receipt is `refused_limit` and carries the venue's message as its detail

### Requirement: Browser venue drives one bet in a fixed order

A browser venue SHALL place a bet in this order: open the event, add the selection to the bet slip, read the slip's
price, check it against the minimum, enter the stake, confirm, and read the confirmation. A step that fails MUST stop
the sequence before the stake is confirmed.

#### Scenario: Event page not found

- **WHEN** the browser venue cannot find the event on the bookmaker's site
- **THEN** it enters no stake and returns a `rejected` receipt that says the event was not found

#### Scenario: Slip shows a different selection

- **WHEN** the bet slip holds a selection other than the intended one after the selection was added
- **THEN** the browser venue enters no stake and returns a `rejected` receipt that says the slip did not match

### Requirement: Browser venue is written per bookmaker

The library SHALL supply a browser venue base that a user completes per bookmaker by writing how to find the event,
add a selection, read the slip, enter the stake, confirm, read the confirmation and tell that the session is logged in.
The library MUST ship no concrete bookmaker.

#### Scenario: Incomplete browser venue

- **WHEN** a user builds a browser venue that does not provide how to add a selection
- **THEN** building it fails with an error that names the missing step

### Requirement: Logged-in check

A browser venue SHALL authenticate by opening the bookmaker's site from the saved browser profile and checking that
the session is logged in. When it is not, authentication MUST fail with an error that tells the user to log in once
with a visible browser using the same profile.

#### Scenario: Profile not logged in

- **WHEN** a browser venue authenticates with a profile that holds no login
- **THEN** authentication fails with an error naming the profile directory and how to log in

#### Scenario: Site blocks automation

- **WHEN** the bookmaker's site answers with a blocking status
- **THEN** authentication raises the venue-blocked error and the venue tries nothing further

## REMOVED Requirements

### Requirement: Pinning controls for a match

**Reason**: The pinned locators lived only in memory, so a pin made by one command was gone before the run that needed
it. A bookmaker also needs logic that a set of locators cannot hold.

**Migration**: Write the locators and the logic in the hooks of a `BaseBrowserVenue` subclass. Use `execution page
read` and `execution page act`, or the `browser_*` read and act tools, to find them.

### Requirement: Resolving a pinned control

**Reason**: It only served the pinned locators, which are removed.

**Migration**: A `BaseBrowserVenue` hook reaches the page through the session and locates its controls directly.
