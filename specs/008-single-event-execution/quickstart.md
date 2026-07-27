# Quickstart: Single-Event Execution Unit

How to validate the feature end to end. Signatures are in [contracts/](./contracts/), entities in
[data-model.md](./data-model.md).

## Prerequisites

- The `execution` extra installed, with a headless browser: `python -m playwright install chromium`.
- A saved dataloader configured for the event's league and season, with a source that carries in-play snapshots.
- A model fitted and saved with `sportsbet evaluation fit`.
- A bookmaker account, logged in once so the browser profile persists the session.

## Gate

A package is done only when all three are green on 3.11, 3.12, and 3.13:

```sh
pdm run formatting
pdm run checks
pdm run tests
```

Read the verdict from the nox session summary line, not a piped exit code.

## Scenario 1: Dry run, preplay model (US1, US2, offline)

Watch one event with stakes off and confirm the decision without money moving.

```sh
sportsbet execution page fix --venue venue.py:VENUE --url "$URL" \
  --match "Arsenal vs Chelsea" --locator stake='textbox[name="Stake"]' --locator confirm='button[name="Place bet"]'

sportsbet execution run --venue venue.py:VENUE -d loader.pkl -b model.pkl \
  --event "Arsenal vs Chelsea" --stake 10 --url "$URL"
```

Expected: the terminal logs the setup match, the status as it advances, the price where available, and at the preplay
moment the exact bet it would place. No stake is placed. The receipts frame is empty.

## Scenario 2: Armed run places one bet at the moment (US1)

Same as Scenario 1 with `--live`. Expected: at the moment, the unit logs the bet about to be placed, drives the pinned
`stake` and `confirm` controls once, and the receipts frame holds exactly one placement for the model's selection at
the configured stake. No second bet is placed however the price moves afterward.

## Scenario 3: No value found (US1)

Run against an event the model finds no value on at its moment. Expected: nothing is placed, the terminal logs that no
bet was found, and the receipts frame is empty.

## Scenario 4: Moment already passed (edge case)

Start the unit after the fitted moment. Expected: the unit logs that the moment is unreachable, places nothing, and
returns an empty frame.

## Scenario 5: No URL matches, or login not completed (edge cases)

Give URLs that do not carry the event, or start without a logged-in profile. Expected: the unit stops during setup,
logs the reason, and stakes nothing.

## Scenario 6: Python API with a custom placer

Call `execute_event` from Python with a `placer` that drives a bespoke site. Expected: same behaviour as the CLI, with
the custom placer invoked once at the moment when armed. See [contracts/python-api.md](./contracts/python-api.md).

## Automated coverage

`tests/execution/test_event.py` reproduces Scenarios 1 to 5 with an injected clock and wait, a fake source scripted
through preplay, inplay, and postplay, a stub session, and a recording placer, with no network and no real browser.
The parity test asserts `execution run` and `execution_run` stay in step and that the removed `quote` and `place`
surfaces are gone.
