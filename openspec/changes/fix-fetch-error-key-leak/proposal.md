# Proposal

## Why

`OddsApi.request_url` adds the key to the URL at request time, so that no declared item carries it. When the vendor
answers with an error status, `fetch_payloads` raises the `aiohttp.ClientResponseError` that aiohttp built, and its
message holds the full URL it requested, key included. Any traceback, log line or report of that error leaks the key.
The `verify-basketball-sources` change met it on 2026-11-15, when the free key was refused by the historical endpoint.

The library already promises the opposite. The user guide says credentials go in `request_url` so that a key stays out
of what is seen and saved, and the agents page says a key passed as an argument is a key written into a transcript.

## What Changes

- When a server answers with an error status, `fetch_payloads` raises an `aiohttp.ClientResponseError` that names the
  item's own URL, never the URL `authorize` returned, and that does not chain the error aiohttp raised.
- The type and the status of the error are kept, so a caller that catches it keeps working.

## Non-goals

- Moving the key out of the URL. The vendor reads it from the query.
- Redacting errors that carry no URL, such as a refused connection, which names only the host.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `sources/source-contract`: the HTTP error a fetch raises names the item and never the authorized URL.

## Impact

- Code: `src/sportsbet/sources/_base.py`.
- Tests: `tests/sources/test_base.py` gains a test against a local server that answers `401`.
