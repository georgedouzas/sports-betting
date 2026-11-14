# Proposal

## Why

A quantitative bettor wants to model a match outcome as it stands at a chosen moment. That moment is the result at full
time, or the leading side 60 minutes into play. The `development` branch held a partial redesign toward this goal. It
had an event-snapshot data model with schema validation, but it was incomplete. Fixtures extraction was a stub. The
concrete soccer loader did not conform to the new base. The dummy loader and its helpers had been deleted. The public
exports were missing. The bettors still parsed the legacy column convention. This change completes that redesign so a
user can extract, backtest and bet at a pre-match or in-play moment through the API, the CLI and the GUI.

The real soccer feed provides only pre-match and post-match information. The product decision was "engine now, data
later": build and validate the in-play engine against bundled synthetic sample data, and let in-play targets on the real
feed return no rows until an in-play source exists.

## What Changes

- Match data becomes time-ordered snapshots labelled with an event status (`preplay`, `inplay`, `postplay`) and an
  event time. Statistics and odds are separate datasets that share a snapshot identity.
- Every public statistics and odds frame is validated against a `pandera` schema before extraction. The schema declares
  for each feature the event statuses it applies to and whether it is fixed.
- `extract_train_data` gains `target_event_status` and `target_event_time`. It returns `(X, Y, O)` for any target
  moment and keeps information from after the target moment out of `X`.
- `extract_train_data` accepts `learning_type` of `'supervised'` or `'unsupervised'` only and always returns a three
  position tuple. `Y` is `None` when unsupervised. **BREAKING**: `'reinforcement'` is rejected as an invalid value.
- `extract_fixtures_data` is implemented and returns `(X, None, O)` with the training column layout.
- **BREAKING**: extracted columns follow a new grammar: fixed columns keep a bare name, time-varying features are
  `{col}__{status}__{time}`, odds are `{provider}__{col}__{status}__{time}`, and targets are
  `{col}__{target_status}__{target_time}`. Bettors parse markets from this grammar instead of the legacy
  `odds__market__outcome__market_type` convention.
- `SoccerDataLoader` keeps the scikit-learn style selection interface: `param_grid`, `get_all_params`,
  `get_odds_types`, `drop_na_thres` and `odds_type`.
- `DummySoccerDataLoader` is restored with in-play sample data and no network access.
- `sportsbet.datasets` exports `BaseDataLoader`, `SoccerDataLoader`, `DummySoccerDataLoader`, `load_dataloader`,
  `BaseStatsSchema`, `BaseOddsSchema`, `required_col` and `optional_col` again.
- The CLI configuration gains `TARGET_EVENT_STATUS` and `TARGET_EVENT_TIME`, and the GUI exposes target moment
  selection.
- Reinforcement learning is designed as a separate future method (`make_env()`) and is not implemented.

## Capabilities

### New Capabilities

- `dataloaders/data-extraction`: the event-snapshot data model, schema validation, moment-aware training and fixtures
  extraction, the selection interface, the sample dataloader, and save and load of dataloaders.
- `evaluation/bettors`: the bettor estimator contract on moment-aware `X`, `Y` and `O`, market discovery from the
  column grammar, value bets for fixtures, and save and load of bettors.
- `evaluation/backtesting`: `backtest` and `BettorGridSearchCV` over moment-aware training data.
- `interfaces/cli`: the `sportsbet` command line and the `sportsbet-gui` app reaching moment-aware extraction,
  backtesting and value-bet identification.

### Modified Capabilities

None. This is the first feature recorded in OpenSpec.

## Impact

- Code: `src/sportsbet/datasets/_base/_dataloader.py`, `src/sportsbet/datasets/_base/_schema.py`,
  `src/sportsbet/datasets/_soccer/_dataloader.py`, `src/sportsbet/datasets/_soccer/_schema.py`,
  `src/sportsbet/datasets/_soccer/_utils.py`, `src/sportsbet/datasets/_dummy.py`,
  `src/sportsbet/datasets/__init__.py`, `src/sportsbet/evaluation/_base.py`, `src/sportsbet/evaluation/_classifier.py`,
  `src/sportsbet/evaluation/_rules.py`, `src/sportsbet/evaluation/_model_selection.py`, `src/sportsbet/cli/_options.py`,
  `src/sportsbet/cli/_data.py`, `src/sportsbet/cli/_betting.py`, `src/sportsbet/gui/app/dataloader_creation.py` and
  `src/sportsbet/gui/app/model_creation.py`.
- Tests: `tests/conftest.py`, `tests/datasets/base/test_dataloader.py`, `tests/datasets/base/test_schema.py`,
  `tests/datasets/test_soccer.py`, `tests/datasets/test_dummy.py`, `tests/evaluation/*`, `tests/test_cli.py` and the
  sample data in `tests/samples/stats.csv` and `tests/samples/odds.csv`.
- API: the column names of `X`, `Y` and `O` change, `learning_type='reinforcement'` is rejected, and the dataloader
  exports return.
- Docs: `docs/overview/user_guide/dataloader.md`, `docs/overview/user_guide/bettor.md`,
  `docs/examples/plot_soccer_data.py`, `docs/examples/modelling/plot_classifier_bettor.py` and a `CHANGELOG.md` entry.
- Dependencies: `pandera[pandas]` and `cloudpickle`, both already declared. No new dependency.
- Quality checks: the specification quality checklist passed every item. The one open clarification, on reinforcement
  learning, was resolved before planning.
