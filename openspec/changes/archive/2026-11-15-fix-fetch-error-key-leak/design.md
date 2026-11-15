# Design

## Decisions

### D1. Name the item, keep the type

`fetch_payloads` pairs each authorized URL with its item's URL. When a request fails with an error status, the fetch
raises a new `aiohttp.ClientResponseError` with the same status and message, whose request names the item's URL. The
item's URL is what the source declared, so it is the address a user can act on, and by contract it holds no credential.
Keeping the type keeps the `HTTP errors raise` requirement and every caller that catches it.

### D2. Do not chain the original

The new error is raised `from None`, so a printed traceback leaves out the error aiohttp built. The original stays on
`__context__`, as Python keeps it, but tracebacks and the `logging` module honour the suppression.

### D3. The fix lives in the fetch, not in the source

Every source that adds a credential in `request_url` has the same exposure, so the fetch removes it for all of them,
rather than `OddsApi` alone.
