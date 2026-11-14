## Purpose

Define the venues where a user holds an account, the identity and receipt of each bet, the credentials named by
variable, and the browser session an agent drives at a bookmaker with no API.

## ADDED Requirements

### Requirement: Placement lives outside the bettor
Bet placement SHALL live in the `sportsbet.execution` module and MUST NOT be a method or attribute of a bettor. Bettors
MUST keep the scikit-learn estimator contract. A fitted bettor MUST NOT hold a credential or a venue, so a saved and
reloaded bettor cannot spend money. Execution MUST leave the dataloaders and evaluation code unchanged.

#### Scenario: A pickled bettor carries no venue
- **WHEN** a fitted bettor is saved and its pickle is searched for a sentinel credential value and a venue
- **THEN** neither is found

#### Scenario: The core is unchanged
- **WHEN** the change is complete
- **THEN** `git diff --stat -- src/sportsbet/dataloaders src/sportsbet/evaluation` is empty

### Requirement: Execution is an optional extra
The execution capability SHALL live behind the optional `execution` extra, which holds `playwright` alone. The default
install MUST NOT pull in any execution dependency and MUST NOT offer a way to place a bet.

#### Scenario: Default install has no execution dependency
- **WHEN** the user runs `pip install sports-betting`
- **THEN** `playwright` is absent from the installed tree

### Requirement: One venue contract
The system SHALL define one venue contract, `BaseVenue`, that every venue with an official API implements. It MUST
declare `key` and `can_cancel`, and MUST cover `authenticate`, `list_markets` with current prices, `read_balance`
returning the balance and open exposure, `place`, `read_status` of placed and open bets, and `cancel`.

#### Scenario: A fake venue satisfies the contract
- **WHEN** a `FakeVenue` implements every `BaseVenue` method in memory
- **THEN** the execution entry points accept it as a venue

### Requirement: No bookmaker ships with the library
The system MUST NOT ship an implementation of the venue contract for any specific bookmaker or exchange. A venue SHALL
be supplied by the user and named the way a model is named, as `venue.py:VENUE`. The contract SHALL be proved in tests
against a fake.

#### Scenario: A user venue is resolved from a file
- **WHEN** the caller names a venue as `venue.py:VENUE`
- **THEN** `build_venue` loads the `VENUE` object from that file

#### Scenario: The package holds no site knowledge
- **WHEN** the package is searched for a site table, a selector pack, or a default naming a bookmaker
- **THEN** none is found

### Requirement: Honest cancellation
A venue that cannot cancel SHALL declare `can_cancel` as false and MUST raise `CancellationUnsupported` when asked to
cancel, rather than return a receipt that implies it cancelled.

#### Scenario: Cancelling at a venue that cannot cancel
- **WHEN** `cancel` is called on a venue with `can_cancel=False`
- **THEN** `CancellationUnsupported` is raised and no receipt is returned

### Requirement: Bet identity
Every intended bet SHALL carry a `BetIdentity` derived from the venue, the match, the market and the selection, and
from nothing else. Its `ref` MUST be exactly 32 hexadecimal characters and MUST be the same for the same four fields in
any process or run. Two intents that share those four fields SHALL be the same bet.

#### Scenario: The reference is deterministic
- **WHEN** two processes build a `BetIdentity` from the same venue, match, market and selection
- **THEN** both `ref` values are equal and are 32 hexadecimal characters

#### Scenario: Any field changes the reference
- **WHEN** one of the four fields differs between two identities
- **THEN** their `ref` values differ

### Requirement: Once-only placement per identity
A venue's `place` MUST be idempotent on `identity.ref`. A retry, a reconnection, a restart, or a fresh run that lost
its state MUST NOT result in more than one stake for one identity. When a bet already exists at the venue for that
identity, the system SHALL report `ALREADY_PLACED` and place nothing further.

#### Scenario: A retry after a timeout
- **GIVEN** the venue accepted a bet but the response was lost to a timeout
- **WHEN** the placement is retried
- **THEN** the venue holds exactly one stake and the retry reports `ALREADY_PLACED`

#### Scenario: A fresh run after a crash
- **GIVEN** a crash between the venue accepting a bet and the receipt being written
- **WHEN** a new process runs the same placement from scratch
- **THEN** the venue holds exactly one stake and the new run reports `ALREADY_PLACED`

### Requirement: The venue is the record
The system SHALL attach each bet's identity to the bet at the venue and MUST decide whether a bet exists by reading the
identity back from the venue. The system MUST NOT keep a local or hidden store of placements. Where a venue cannot carry
a caller reference, the system MUST recognise an already placed bet from what the venue reports.

#### Scenario: No placement state on disk
- **WHEN** a batch has been placed
- **THEN** no file, cache or directory holds placement state

#### Scenario: Tracing from the venue alone
- **WHEN** the venue is queried for the identities of placed bets with no local state available
- **THEN** each bet traces back to its match, market and selection

### Requirement: Placement receipts
Every placement attempt SHALL produce a `PlacementReceipt` identifying the venue, the match, the market, the selection,
the stake, the price obtained, the `PlacementStatus` outcome, the venue's bet identifier when given, and the value bet
that caused it. Receipts SHALL cross public boundaries as a DataFrame validated by a `pandera` schema. A refused
receipt MUST carry a stake of zero.

#### Scenario: A receipt identifies its value bet
- **WHEN** the user inspects a placement receipt
- **THEN** it identifies the value bet, the market, the selection, the stake and the price it was placed at

#### Scenario: The venue declines a bet
- **WHEN** the venue declines a placement because the market is suspended or closed or the balance is short
- **THEN** the receipt is `REJECTED` with a stake of zero and a `detail` naming the reason

### Requirement: Status and balance from the venue
A venue SHALL report the balance and the exposure currently open, and SHALL report each placed bet as open, matched in
full or in part, rejected, or settled.

#### Scenario: Reading status
- **GIVEN** bets placed earlier, one of them matched in part
- **WHEN** the user asks for their status
- **THEN** each is reported as open, matched, rejected or settled, and the partial match is reported as partial

#### Scenario: Reading the balance
- **WHEN** the user asks a venue for the balance
- **THEN** both the balance and the current exposure are reported

### Requirement: Credentials named by variable
A credential SHALL be read from the environment or a secret store at use. The caller MUST name the variable that holds
it and MUST NOT pass the secret as a function argument, a command line option, or a tool argument. When the variable is
missing, the system MUST name the variable it expected and stop.

#### Scenario: Missing credential
- **GIVEN** no credential in the environment
- **WHEN** the user attempts to authenticate
- **THEN** the system names the missing variable and places nothing

### Requirement: Credentials never leak
A credential value MUST NOT appear in a return value, a `repr`, a log line, an error message, a receipt `detail`, a
saved file, a pickle, or a transcript.

#### Scenario: A sentinel secret is never echoed
- **GIVEN** a credential variable set to a sentinel value
- **WHEN** every execution command and tool runs
- **THEN** the sentinel appears in no output, log, exception, written file or pickle

### Requirement: Browser session primitives
For a bookmaker with no API, the system SHALL provide `BrowserSession`, which exposes generic primitives: `navigate`,
`snapshot`, `click`, `type` and `select`. Reads SHALL return a `PageSnapshot` holding an AI-mode ARIA snapshot with
element refs. Each act SHALL take a ref and return the resulting snapshot. The system MUST NOT claim that any site
works.

#### Scenario: Reading a market page
- **GIVEN** a bookmaker site and an authenticated session
- **WHEN** the agent navigates to a market page
- **THEN** it receives a snapshot carrying roles, names and refs it can act on

### Requirement: Actions refuse controls that cannot be acted on
A browser action MUST fail on a hidden or disabled control rather than report success.

#### Scenario: Disabled confirm button
- **GIVEN** a bet slip whose confirm button the site has disabled
- **WHEN** the agent clicks its ref
- **THEN** the action fails and reports no placement

### Requirement: The browser session is not a venue
`BrowserSession` MUST NOT implement `BaseVenue` and MUST NOT expose `place`, `read_status` or `cancel`. On the
site-driven path the agent places, so once-only placement and the stake and exposure ceilings are the agent's to keep.

#### Scenario: Looking for placement on the site path
- **WHEN** a caller looks for `place`, `read_status` or `cancel` on a `BrowserSession`
- **THEN** none exists and the session is not a `BaseVenue`

### Requirement: Pinned sessions
`fix(match, locators)` SHALL pin role and accessible name locators the agent discovered into a `FixedSession`, so later
actions resolve after the page re-renders. It MUST reject a raw snapshot ref and MUST NOT store a price.

#### Scenario: Acting after a re-render
- **GIVEN** a page whose odds widget re-renders and changes every ref
- **WHEN** the agent pins the stake and confirm locators and acts after a re-render
- **THEN** the pinned locators still resolve

#### Scenario: A raw ref is rejected
- **WHEN** `fix` is given a locator that is a snapshot ref such as `e5`
- **THEN** it refuses the locator

### Requirement: Venue notes returned verbatim
A browser session SHALL store its `key`, `url`, `notes`, `credential_env`, `user_data_dir` and `min_interval`
unmodified. `notes` SHALL be one free-text value that the system returns verbatim and never parses.

#### Scenario: Notes reach the agent unchanged
- **WHEN** the venue information is read
- **THEN** the `notes` text equals the text the user configured

### Requirement: Blocked venues stop
The system MUST NOT include any means of evading a venue's automation controls, including stealth browsing,
fingerprint spoofing, captcha solving, proxy rotation, geographic circumvention, or human-timing mimicry. When a venue
blocks automation, the system SHALL report `BLOCKED` and stop. Pacing SHALL be a minimum interval in seconds.

#### Scenario: A block response
- **GIVEN** a venue that blocks automated access
- **WHEN** that block is met
- **THEN** the system reports `BLOCKED` and stops
- **AND** it makes no retry with different timing, headers or presentation

### Requirement: No agent loop in the package
The package MUST NOT contain an agent loop, a model call, a model key, or a model choice. Monitoring and deciding SHALL
belong to the outside agent that drives the tools.

#### Scenario: Browser tools carry no site knowledge
- **WHEN** the agent drives a site through the browser primitives
- **THEN** the agent supplies which site, which control and which market, and the package makes no model call

### Requirement: Risk stated ahead of use
The documentation MUST state, prominently and ahead of any instruction to use it, that driving a bookmaker's website
breaches essentially every bookmaker's terms of service and risks account closure and loss of the balance. It MUST
state that no exchange offers a placement sandbox.

#### Scenario: Reading the execution guide
- **WHEN** the user opens the execution user guide
- **THEN** the terms-of-service and money risks appear before any instruction

### Requirement: Tests never touch a venue
No test SHALL contact a real venue or place a real bet. Tests MUST use fakes, recorded responses, a page served over
loopback, or a venue sandbox.

#### Scenario: Running the suite
- **WHEN** the full test suite runs
- **THEN** it makes zero requests to a live venue and places zero real bets
