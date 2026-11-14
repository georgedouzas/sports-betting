## ADDED Requirements

### Requirement: Headless browser session
The browser session that serves the single-event unit MUST run headless, with no live-preview window. It SHALL be used
to navigate, explore and place, not to present a monitoring view. Login SHALL be interactive at setup and SHALL be able
to reuse a persisted browser profile, so a later headless run starts authenticated.

#### Scenario: No window opens
- **WHEN** a single-event run drives the bookmaker's site
- **THEN** no browser window is shown and all information appears in the terminal log

#### Scenario: A persisted login is reused
- **GIVEN** the user logged in once with a persisted browser profile
- **WHEN** a later run starts with the same profile
- **THEN** the headless session is already authenticated
