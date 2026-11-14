# execution/venue-contract Specification

## Purpose

Defines the venues where the library places bets, the identity and receipt of a bet, how secrets are read, and the
browser session that drives a bookmaker's website for a venue with no API.

## Requirements

### Requirement: Venue contract

A venue SHALL be a `BaseVenue` subclass with a `key` and a `can_cancel` flag that implements `authenticate`,
`list_markets`, `read_balance`, `place`, `read_status` and `cancel` as coroutines. It MUST place a bet once and only
once for each bet identity.

#### Scenario: Incomplete venue

- **WHEN** a user instantiates a `BaseVenue` subclass that does not implement `read_status`
- **THEN** Python raises `TypeError` naming the missing method

### Requirement: Bet identity

A bet identity SHALL be its venue, match, market and selection. Its `ref_` MUST be a 32 character hexadecimal digest
of those four fields, so two identities with the same fields have the same reference.

#### Scenario: Same fields, same reference

- **WHEN** two `BetIdentity('exchange', 'Arsenal vs Chelsea', 'home_win', 'Arsenal')` objects are built
- **THEN** their `ref_` values are equal and are `2f3152b1a4f9f6333904f3624320d921`

### Requirement: Placement intent and receipt

A placement intent SHALL carry an identity, a stake, a minimum price and the value bet it came from. A receipt SHALL
carry the identity, a status, the stake, the price, the venue bet id, the value bet, the time placed and a detail.

#### Scenario: Receipt defaults

- **WHEN** a receipt is built with only an identity and a status
- **THEN** its stake is 0.0 and its price, venue bet id and time placed are empty

### Requirement: Placement statuses

A receipt status SHALL be one of `dry_run`, `accepted`, `matched_full`, `matched_partial`, `already_placed`,
`refused_limit`, `refused_price`, `refused_unconfirmed`, `refused_killed`, `blocked` or `rejected`.

#### Scenario: Unknown status

- **WHEN** a caller builds a placement status from the string `pending`
- **THEN** Python raises `ValueError`

### Requirement: Receipts frame

`build_receipts_frame` SHALL return receipts as a frame with exactly the columns `ref`, `venue`, `match`, `market`,
`selection`, `status`, `stake`, `price`, `venue_bet_id`, `value_bet`, `placed_at` and `detail`, validated against a
strict schema. The stake MUST be zero or more and `placed_at` MUST be a UTC timestamp.

#### Scenario: No receipts

- **WHEN** `build_receipts_frame` is called with an empty list
- **THEN** it returns an empty frame with the twelve columns

#### Scenario: Negative stake

- **WHEN** a receipt with a negative stake is framed
- **THEN** schema validation fails

### Requirement: Secrets are named, not passed

A secret SHALL be referenced by the name of the environment variable that holds it, as a `CredentialRef`, and read with
`resolve` where it is used. When the variable is unset or empty, `resolve` MUST raise `CredentialError` naming it.

#### Scenario: Variable unset

- **WHEN** `resolve(CredentialRef('VENUE_API_KEY'))` runs with `VENUE_API_KEY` unset
- **THEN** it raises `CredentialError` whose message names `VENUE_API_KEY`

### Requirement: Building a venue from a reference

`build_venue` SHALL load a venue from a reference of the form `file.py:NAME` and return a `BaseVenue` or a
`BrowserSession`. It MUST raise `BuildError` for a reference without a colon, a missing execution extra, or an object
that is neither.

#### Scenario: Reference without a file

- **WHEN** `build_venue('betfair')` is called
- **THEN** it raises `BuildError` saying the venue must be named in a Python file and that the library ships none

#### Scenario: Object of the wrong type

- **WHEN** the reference names an object that is not a venue or a browser session
- **THEN** it raises `BuildError` saying so

### Requirement: Browser session configuration

A `BrowserSession` SHALL store its key, site URL, notes, credential variable names, profile directory, minimum interval
between actions, element timeout and headless flag unmodified. It MUST run headless by default and MUST NOT offer a
`place` method.

#### Scenario: Default session

- **WHEN** a session is built with only a key and a URL
- **THEN** it is headless, waits one second between actions, waits 30 seconds for an element, and has no `place`

### Requirement: Browser opens on first navigation

A browser session SHALL open the browser on its first navigation, from the profile directory when one is given so a
saved login is reused. `stop` MUST close the browser.

#### Scenario: Navigate before start

- **WHEN** `navigate` is called on a session that has not been started
- **THEN** the session starts the browser and goes to the page

### Requirement: Blocked sites are reported

When a navigation answers with status 403 or 429, the session SHALL raise `VenueBlockedError` naming the venue and the
status.

#### Scenario: Site answers 403

- **WHEN** the session navigates to a page that answers 403
- **THEN** it raises `VenueBlockedError`

### Requirement: Pages as accessibility snapshots

`navigate`, `read_snapshot`, `click`, `type` and `select` SHALL each return the page as a `PageSnapshot` holding an
accessibility tree in YAML and the page URL. Every actionable element MUST carry a ref, and `click`, `type` and
`select` act on an element by its ref.

#### Scenario: Typing shows in the result

- **WHEN** a user types into a textbox by its ref
- **THEN** the returned snapshot shows the typed text in that textbox

#### Scenario: Reading part of a page

- **WHEN** `read_snapshot` is called with a selector and a depth
- **THEN** it returns only that part of the page, to that depth

### Requirement: Actions wait between each other

The session SHALL wait its minimum interval before each navigation and each action.

#### Scenario: Two clicks

- **WHEN** a user clicks twice with a minimum interval of one second
- **THEN** at least one second passes before each click

### Requirement: Session without a page

An action on a session with no open browser or no page SHALL raise `ExecutionError` telling the user to start or
navigate first.

#### Scenario: Click before navigating

- **WHEN** `click` is called before any navigation
- **THEN** it raises `ExecutionError` telling the user to call `start` first

### Requirement: Pinning controls for a match

`fix` SHALL pin named locators for a match at the current page URL and keep them on the session in memory. It MUST
refuse a locator that is a ref and a name that contains `price` or `odds`, raising `ExecutionError`.

#### Scenario: Ref refused

- **WHEN** `fix` is given `{'stake': 'e3'}`
- **THEN** it raises `ExecutionError` telling the user to pin a role and an accessible name instead

#### Scenario: Price refused

- **WHEN** `fix` is given a locator named `price`
- **THEN** it raises `ExecutionError` saying a price is read when the bet is placed

### Requirement: Resolving a pinned control

`resolve` SHALL return a snapshot of what a pinned locator points at now. It MUST raise `ExecutionError` when nothing
is pinned or the name was not pinned, listing the pinned names.

#### Scenario: Name not pinned

- **WHEN** `resolve('confirm')` is called on a session that pinned only `stake`
- **THEN** it raises `ExecutionError` listing `stake` as pinned
