# Feature Specification: Bet execution

**Feature Branch**: `006-bet-execution`

**Created**: 2026-07-16

**Status**: Draft

**Input**: User description: "Real betting: authenticate to a bookmaker, read its markets, and place bets, driven by an outside agent. The library finds value bets and stops. Add a `sportsbet.execution` module behind an optional extra, with adapters for official betting APIs and for driving a bookmaker's website, exposed through the CLI and MCP so an outside agent can place and watch bets. Placement must not live on the bettor. Dry run by default, stake and exposure limits, a kill switch, idempotent placement, credentials from the environment. No agent loop, no anti-detection machinery."

## Clarifications

### Session 2026-07-16

- Q: For a bookmaker with no API, where does the site-specific knowledge live? → A: The library exposes generic browser and page capabilities and the agent supplies the site knowledge. No per-site adapters ship.
- Q: Who may authorise real money, and how? → A: Any caller including an agent, but only by passing back the exact stake and total exposure just quoted, as feature 005 does with confirm_cost. Human-set limits still cap it.
- Q: Where does the durable record of what was placed live? → A: The venue. The system attaches its own reference to each bet and reads it back to dedupe and reconcile. No local or hidden store.
- Q: What makes two bets the same bet for deduplication? → A: The venue, the match, the market and the selection. One stake per selection per venue, independent of which run produced it.
- Q: How does the caller express the worst acceptable price? → A: An absolute minimum price per bet, defaulting to the price the value bet was computed at.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Place the value bets at a venue with an official betting API (Priority: P1)

A user has fitted a bettor and extracted the upcoming fixtures. `bettor.bet(X_fix, O_fix)` tells them which markets are worth backing. Today they open a browser and type each one in by hand. Instead, they hand those value bets to execution, point it at a venue where they hold an account, and see exactly what would be staked and what the total exposure would be. Nothing moves. When they are satisfied, they opt in explicitly and the bets are placed.

**Why this priority**: This is the point of the feature and the smallest slice that closes the loop from model to money. It uses the venue's own sanctioned betting API, so it is stable, testable, and leaves the user's account in good standing. Everything else builds on the contract this story defines.

**Independent Test**: Fully testable against a fake venue and recorded payloads: take a set of value bets, run execution in its default mode and assert nothing was staked, then run with the explicit opt-in and assert exactly the intended bets were placed once each. Planning established that no exchange offers a placement sandbox, so a fake is the only safe test facility rather than the fallback.

**Acceptance Scenarios**:

1. **Given** a set of value bets and a configured venue, **When** the user runs execution without opting in to real money, **Then** the system reports each bet it would place and the total exposure, and places nothing.
2. **Given** the same value bets and the quote just returned, **When** the caller passes back the exact stake and total exposure quoted, **Then** each intended bet is placed exactly once and a receipt is returned for each.
3. **Given** a confirmation whose figures do not match the quote, **When** placement is attempted, **Then** nothing is placed and the system states the real figures.
4. **Given** a stake that would exceed the configured exposure limit, **When** the caller confirms, **Then** the system refuses that placement and names the limit reached.
5. **Given** no credential in the environment, **When** the user attempts to authenticate, **Then** the system names the missing variable and places nothing.
6. **Given** a placement that is interrupted and retried, **When** the retry runs, **Then** the venue holds exactly one stake for that bet.

---

### User Story 2 - Follow the money after placement (Priority: P2)

Having placed bets, the user needs to know what became of them: what the balance is, which bets are still open, which matched, which were rejected, and which value bet each one came from. Without this, placement is a write-only operation and the user cannot tell whether the model's edge survived contact with the venue.

**Why this priority**: Placement without reconciliation is unusable in practice, but it is only meaningful once P1 exists. It is also what lets a backtested edge be compared with what was actually obtained.

**Independent Test**: Against a fake venue holding known bets, read balance, open bets and statuses, and assert each placement traces back to the value bet, the match, the market and the selection that caused it.

**Acceptance Scenarios**:

1. **Given** bets placed earlier, **When** the user asks for their status, **Then** each is reported as open, matched, rejected or settled.
2. **Given** a placement receipt, **When** the user inspects it, **Then** it identifies the value bet, the market, the selection, the stake and the price it was placed at.
3. **Given** a venue account, **When** the user asks for the balance and the current exposure, **Then** both are reported.

---

### User Story 3 - Place at a bookmaker that publishes no API, driven by an agent (Priority: P3)

Most bookmakers publish no betting API. For those, the user points an agent at the bookmaker's website through tools the library exposes: the agent navigates, reads the page, finds the market and the bet slip, and places the bet. The library supplies the browser and the page, and the agent supplies the knowledge of that particular site.

**Why this priority**: It is the least safe and least stable path. It breaches essentially every bookmaker's terms of service, risks account closure and loss of the balance, and breaks whenever the site's markup changes. It is worth doing last, behind the sanctioned path, and only with that risk stated plainly.

**Independent Test**: Against a fake bookmaker page served over loopback, an agent can navigate, read the markets, pin a session, fill a bet slip and confirm a placement.

**What this path does not promise**: Planning found FR-005 and FR-007 in contradiction, because a `place` on a website has to find the market, click the price, fill the stake and work the confirm flow, and every one of those is site knowledge. That makes it either a per-site adapter, which FR-007 forbids, or a model call inside the package, which FR-022 forbids. The site-driven path therefore exposes primitives and the agent places. Two guarantees consequently hold on the API path and are the agent's here: placing once and only once, which needs the venue's record read back, and the stake and exposure ceilings, which bind whoever calls the venue. The documentation states this rather than letting a user infer that the rails are the same.

**Acceptance Scenarios**:

1. **Given** a bookmaker site and an authenticated session, **When** the agent reads a market page, **Then** it receives the page content in a form it can reason about and act on.
2. **Given** a market page the agent has explored, **When** it pins the session, **Then** later actions resolve against the pinned locators after the page re-renders.
3. **Given** a confirm control the site has disabled, **When** the agent acts on it, **Then** the action fails rather than reporting a placement that did not happen.
4. **Given** a venue that blocks automated access, **When** that block is met, **Then** the system reports it and stops, and offers no means of getting around it.
5. **Given** the site-driven path, **When** a caller looks for placing, status or cancelling on it, **Then** those are absent, so no caller can reach a guarantee that is not there.

---

### Edge Cases

- A placement is retried after a timeout, a crash, or a lost connection. The user must never be staked twice for one intended bet.
- The price moves between the moment the value bet was computed and the moment the bet is placed. The bet must not go on at a price worse than the user allowed.
- The market is suspended, closed, or already settled when placement is attempted.
- The account holds less than the intended stake.
- An exchange matches only part of the stake.
- The exposure limit is reached partway through a batch of bets.
- The kill switch is engaged while a batch is in flight.
- A credential is absent, malformed, or expires mid-session.
- A site-driven venue changes its markup, so the page the agent expects is no longer there.
- A venue detects and blocks automation. The system reports it and stops.
- The same value bets are submitted twice by a user or an agent that lost track of state.

## Requirements *(mandatory)*

### Functional Requirements

#### Placement lives outside the estimator

- **FR-001**: Bet placement MUST live in a new `sportsbet.execution` module and MUST NOT be a method or attribute of a bettor.
- **FR-002**: Bettors MUST remain unchanged and MUST continue to satisfy the scikit-learn estimator contract. A fitted bettor MUST NOT hold a credential and MUST NOT be able to cause a real-world side effect, so that a saved and reloaded bettor cannot spend money.
- **FR-003**: Execution MUST consume the value bets a bettor produces, together with the identity of the matches they refer to.
- **FR-004**: The execution capability MUST live behind an optional extra and MUST NOT add a required runtime dependency to the default install.

#### The venue contract

- **FR-005**: The system MUST define one venue contract that every venue implements, covering: authenticate, list markets and their current prices, read balance and current exposure, place a bet, read the status of placed and open bets, and cancel a bet where the venue supports cancelling.
- **FR-006**: The system MUST ship a reference implementation of that contract against a venue with an official, sanctioned betting API.
- **FR-007**: The system MUST support a venue that publishes no API by exposing generic browser and page capabilities to an agent: navigate, read the page in a form the agent can reason about, and act on it. The agent supplies the knowledge of that particular site. The system MUST NOT ship or maintain per-site adapters, and MUST NOT claim that any given site works.
- **FR-008**: A venue that cannot cancel MUST say so rather than appear to cancel.

#### Refusal is the default

- **FR-009**: Execution MUST NOT place a real bet unless the caller passes back the exact stake and total exposure the system quoted for that batch. A caller that omits the confirmation, or passes figures that do not match the quote, MUST result in zero stakes, and the system MUST state the real figures. This rule applies to every caller and every surface, including an agent calling a tool, so that spending requires having read the quote.
- **FR-010**: Before any money moves, the system MUST report every bet it is about to place, its stake, its price, and the total exposure of the batch.
- **FR-011**: The system MUST enforce a maximum stake per bet and a maximum total exposure, and MUST refuse a placement that would exceed either, naming the limit reached.
- **FR-012**: The system MUST provide a kill switch that stops further placement.
- **FR-013**: Every intended bet MUST carry a minimum acceptable price, and the system MUST refuse to place it below that price. The minimum MUST default to the price the value bet was computed at, since below that price the bet is no longer a value bet. A caller MAY set a different minimum.

#### Placing once and only once

- **FR-014**: Every intended bet MUST carry an identity derived from the venue, the match, the market and the selection, and from nothing else. Two intents that share those four ARE the same bet, whichever run produced them, so one selection at one venue carries at most one stake. A retry, a reconnection, a restart, or a fresh run by a caller that lost track of its state MUST NOT result in more than one stake for that selection.
- **FR-014a**: When a bet already exists at the venue for that identity, the system MUST report it as already placed and MUST place nothing further, rather than failing.
- **FR-015**: The venue MUST be the record of what was placed. The system MUST attach the bet's identity to the bet at the venue, and MUST decide whether that bet already exists by reading the identity back from the venue rather than by keeping state of its own. The system MUST NOT create a local or hidden store of placements, so there is nothing that can drift out of step with the venue. Where a venue cannot carry a caller reference, the system MUST recognise an already placed bet from what the venue reports.
- **FR-016**: Every placement MUST produce a receipt identifying the venue, the market, the selection, the stake, the price, the outcome of the attempt, and the value bet that caused it.

#### Credentials

- **FR-017**: A credential MUST be read from the environment or a secret store. The caller MUST name the variable holding it and MUST NOT pass the secret itself as a function argument, a command line option, or a tool argument.
- **FR-018**: A credential MUST never reach a log, an error message, a saved file, or a transcript.
- **FR-019**: When a credential is missing, the system MUST name the variable it expected and stop.

#### Surfaces

- **FR-020**: Every execution capability MUST be reachable from the Python API, the CLI, and the MCP server, so no surface holds a capability the others cannot reach.
- **FR-021**: The MCP server MUST expose execution as tools an outside agent can call: authenticate, read markets, place, read status, and, for a site-driven venue, read and act on the page.
- **FR-022**: The package MUST NOT contain an agent loop, a model call, a model key, or a model choice. The monitoring and the deciding belong to the agent that drives the tools.

#### Boundaries

- **FR-023**: The system MUST NOT include any means of evading a venue's automation controls, including stealth browsing, fingerprint spoofing, and captcha solving. When a venue blocks automation, the system reports it and stops.
- **FR-024**: The documentation MUST state, prominently and ahead of any instruction to use it, that driving a bookmaker's website breaches essentially every bookmaker's terms of service and risks account closure and loss of the balance.
- **FR-025**: The existing dataloaders and evaluation code MUST be unchanged by this feature.

#### Testing

- **FR-026**: No test may contact a real venue or place a real bet under any circumstance. Tests MUST use fakes, recorded responses, a locally served page, or a venue sandbox.

### Key Entities

- **Venue**: A place where a user holds an account and can back a selection. It is reached either through an official betting API or by driving its website. It authenticates, quotes, places, reports, and sometimes cancels.
- **Credential reference**: The name of the environment variable or secret entry holding a secret. The secret itself never travels through the system's interfaces.
- **Market**: An event and an outcome that can be backed at a venue, with a current price. It corresponds to a betting market the library already models, such as a home win.
- **Placement intent**: What the user means to do. A venue, a match, a market, a selection, a stake, and a minimum acceptable price defaulting to the price the value bet was computed at. Its identity is the venue, match, market and selection together, which is what makes it recognisable if it arrives again.
- **Placement receipt**: What happened. Accepted, rejected, or matched in part or in full, at what price, when, at which venue, and which value bet it came from.
- **Exposure**: The total amount currently at stake, against which the limits are enforced.
- **Value bet**: The existing output of a bettor. It is the input to execution and the thing every receipt traces back to.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can take the value bets of an upcoming fixture set and place them at a supported venue without re-entering any of them by hand.
- **SC-002**: Of runs that do not explicitly opt in to real money, 100% place zero real bets.
- **SC-003**: Under fault injection covering timeouts, crashes and retries during placement, the number of stakes never exceeds one per intended bet.
- **SC-004**: Zero credential values appear in any command argument, tool argument, log line, error message, saved file, or recorded transcript.
- **SC-005**: 100% of placements can be traced back, from the venue alone, to the value bet that caused them: the match, the market and the selection.
- **SC-006**: Every execution capability is reachable from all three surfaces: the Python API, the CLI, and the MCP server.
- **SC-007**: The test suite places zero real bets and makes zero requests to a live venue.
- **SC-008**: Before any money moves, the user can see every intended stake and the total exposure of the batch.
- **SC-009**: A placement that would exceed a stake or an exposure limit is refused 100% of the time, with the limit named.
- **SC-010**: Installing the package without the optional extra pulls in no execution dependency and offers no way to place a bet.

## Assumptions

- Users hold their own accounts and their own funds at the venue. The system places and cancels bets and reads state. It never opens accounts, deposits, or withdraws.
- Users are responsible for the legality of automated betting where they live and for their agreement with the venue. The documentation states the terms-of-service risk of the site-driven path rather than deciding it for them.
- A venue is reachable from some jurisdictions and not others, and the two adapter families answer to different users. Planning established that all four surveyed exchanges are closed to Greece, where the maintainer is: three sit on the Hellenic Gaming Commission blacklist and Betfair blocks Greece itself, holding no Greek licence since it withdrew in 2012. The Betfair adapter therefore serves users elsewhere and fixes the shape of the venue contract, while the site-driven path serves the venues a Greek user can actually reach. Both ship. The system leaves a venue's geographic controls alone, so a user outside a venue's footprint uses a different venue.
- The reference venue is a betting exchange that publishes an official API intended for programmatic betting. Planning settled this on Betfair. Planning also established that no such exchange offers a placement sandbox: Betfair's delayed application key operates on the live exchange and places real bets, so it is a test facility for prices rather than for placement. The sanctioned path is therefore proved against fakes, recorded responses and contract tests, which is what FR-026 already requires.
- Stake sizing is an input, not a model. It comes from the caller or from the bettor's existing stake parameter. Bankroll strategy is out of scope.
- The venue can carry a caller-supplied reference on a bet, which is what lets the venue be the record. A venue that cannot MUST still let an already placed bet be recognised from what it reports. Planning confirmed this for Betfair and found that it takes two references rather than one: a request reference that deduplicates within a sixty second window and is not readable afterwards, and an order reference that persists and can be filtered on but that the venue leaves unpoliced. Placing once and only once needs both, so the venue contract carries both. Planning also found that Betfair is the only exchange of the four surveyed that carries a caller reference at all.
- A bet is placed at the price on offer when it lands, which may be worse than the odds the model was backtested against, down to the minimum acceptable price. Reconciliation records what was actually obtained.
- Credentials are provisioned by the user before use, following the pattern the library already uses for the odds source, where a tool names the variable rather than carrying the secret.
- A model registry, automated promotion, or any MLOps lifecycle is out of scope and belongs to a different project.
- The agent that watches markets and decides when to place lives outside the package, consistent with the constitution and with feature 005.
