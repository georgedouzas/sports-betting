# Contract: `sportsbet` CLI

The CLI (and, equivalently, the `sportsbet-gui` app) MUST expose the same moment-aware
capabilities as the API (FR-019, Constitution Principle I parity). No capability may be
reachable from only one surface.

## Config-driven workflow (existing pattern, updated)

The CLI is driven by a Python config module declaring the dataloader, selection, and bettor.
The config contract MUST be updated so it no longer references removed symbols and can
express a target moment.

Current breakage: `tests/conftest.py`'s embedded `CONFIG` imports
`DummySoccerDataLoader` and `OddsComparisonBettor` — the former is deleted and MUST be
restored (see [datasets-api.md](./datasets-api.md)).

Config MUST support (in addition to existing `DATALOADER_CLASS`, `PARAM_GRID`,
`DROP_NA_THRES`, `ODDS_TYPE`, `BETTOR`, `CV`):
- `TARGET_EVENT_STATUS` — `preplay` | `inplay` | `postplay` (default `postplay`).
- `TARGET_EVENT_TIME` — in-play time (e.g. `'60min'`); ignored unless status is `inplay`.

## Commands (capabilities, not exact flags)

| Capability | Requirement |
|-----------|-------------|
| Extract training data | MUST accept the config's target moment and write/report `X`/`Y`/`O`. |
| Extract fixtures data | MUST reproduce the training column layout; `Y` empty. |
| Backtest | MUST run the configured bettor over moment-aware training data and report performance. |
| Identify value bets | MUST output value-bet selections for fixtures at the target moment. |

## Parity check (Success Criterion SC-005)

For the same config against the sample dataloader, the CLI, GUI, and API MUST produce
equivalent extraction, backtest, and value-bet results.

## Error behavior

- Invalid config (unknown symbol, bad target moment) → clear, non-traceback CLI error.
- No resolvable outcomes / empty selection → informative message, non-zero where appropriate.
