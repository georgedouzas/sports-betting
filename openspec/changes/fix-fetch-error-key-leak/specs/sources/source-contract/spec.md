## MODIFIED Requirements

### Requirement: HTTP errors raise

`fetch_payloads` SHALL raise `aiohttp.ClientResponseError` when a server answers a request with an error status, and
MUST NOT return a payload for an error response. The error SHALL name the item's own URL and MUST NOT hold the URL
`authorize` returned, in its message or in a chained error, so a credential added at request time never reaches a
traceback or a log.

#### Scenario: Missing remote file

- **WHEN** a caller fetches an item whose server answers with status 404
- **THEN** `fetch_payloads` raises `aiohttp.ClientResponseError`

#### Scenario: Key kept out of the error

- **GIVEN** an `authorize` that adds `apiKey=secret` to the item URL
- **WHEN** the server answers with status 401
- **THEN** `fetch_payloads` raises `aiohttp.ClientResponseError` with status 401 whose message names the item URL
- **AND** `secret` appears nowhere in the formatted traceback
