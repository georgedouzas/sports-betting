# Tasks

The group ends green on `pdm run formatting`, `pdm run checks`, `pdm run docs build` and `pdm run tests`, read from the
nox session summary line, with the docs build run with no secret set.

## 1. Fetch errors

- [x] 1.1 Write failing tests in `tests/sources/test_base.py` against a local server: a `404` raises
  `aiohttp.ClientResponseError`, and a `401` on a URL that `authorize` gave `apiKey=secret` raises an error with status
  401 whose message names the item URL and whose formatted traceback holds no `secret`.
- [x] 1.2 In `src/sportsbet/sources/_base.py`, pair each authorized URL with its item URL and raise the error named
  by the item, `from None`. Verify the tests of 1.1.
- [x] 1.3 Commit as `fix(sources): ...`.
