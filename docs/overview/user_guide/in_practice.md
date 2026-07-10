[pandas Timedelta]: <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>

# Sports betting in practice

The [dataloader](dataloader.md) turns raw matches into moment-aware `X`, `Y` and `O`, and the [bettor](bettor.md) turns those into
a betting strategy. This page puts the two together into an end-to-end flow and answers the question that matters most once you
start betting for real: **which data can I actually use, and when?**

## The golden rule: train it the way you will use it

The target moment you pass to `extract_train_data` fixes two things at once: *what* you predict (the outcome at that moment) and
*which* snapshots become features (every snapshot strictly before it). A feature such as `home_goals__inplay__45min` is only known
once the match has reached the 45-minute mark. So:

> A model may only use features that will be available at the moment you place the bet.

`extract_fixtures_data` reproduces the exact training columns for upcoming matches, and any snapshot a match has not reached yet is
simply missing (`NaN`). The practical question is therefore always: *how far has the match progressed when I bet?*

- Betting **before kick-off** → only pre-match information exists, so the model must use pre-match features only.
- Betting **in-play** → the snapshots up to the current minute exist too, so the model may use in-play features up to that minute.

This is why including half-time data in `X` at training time requires half-time data at prediction time: you can only bet on a
match once it has actually reached half-time.

## Betting before kick-off

Before a match starts, the only information available is the fixed identity of the match and its pre-match features. In-play
snapshots do not exist yet. We can see this directly: build an upcoming fixture with `from_snapshots` and inspect what
`extract_fixtures_data` returns for it.

```python
import pandas as pd
from sportsbet.datasets import SoccerDataLoader, market_outcomes

# Two finished matches (with a half-time snapshot) and one upcoming fixture.
def snapshot(status, minutes, date, home, away, hg, ag, home_avg, away_avg):
    return dict(event_status=status, event_time=minutes, date=date, league='England', division=1, year=2025,
                home_team=home, away_team=away, home_goals=hg, away_goals=ag,
                home_points_avg=home_avg, away_points_avg=away_avg)

stats = pd.DataFrame([
    snapshot('preplay', 0, '2024-08-16', 'Arsenal', 'Chelsea', None, None, 2.1, 1.5),
    snapshot('inplay', 45, '2024-08-16', 'Arsenal', 'Chelsea', 1, 0, None, None),
    snapshot('postplay', 0, '2024-08-16', 'Arsenal', 'Chelsea', 2, 0, None, None),
    snapshot('preplay', 0, '2024-08-23', 'Everton', 'Spurs', None, None, 1.2, 1.9),
    snapshot('inplay', 45, '2024-08-23', 'Everton', 'Spurs', 0, 1, None, None),
    snapshot('postplay', 0, '2024-08-23', 'Everton', 'Spurs', 1, 2, None, None),
    snapshot('preplay', 0, '2025-09-01', 'Liverpool', 'Wolves', None, None, 2.4, 1.0),  # upcoming fixture
])
markets = ['home_win', 'draw', 'away_win']
played = stats['home_goals'].notna()
stats.loc[played, markets] = market_outcomes(stats.loc[played, 'home_goals'], stats.loc[played, 'away_goals'], markets).to_numpy()
odds = pd.DataFrame([
    dict(event_status='preplay', event_time=0, date='2024-08-16', league='England', division=1, year=2025,
         home_team='Arsenal', away_team='Chelsea', provider='market_average', home_win=1.7, draw=3.6, away_win=4.8),
    dict(event_status='preplay', event_time=0, date='2024-08-23', league='England', division=1, year=2025,
         home_team='Everton', away_team='Spurs', provider='market_average', home_win=2.6, draw=3.3, away_win=2.5),
    dict(event_status='preplay', event_time=0, date='2025-09-01', league='England', division=1, year=2025,
         home_team='Liverpool', away_team='Wolves', provider='market_average', home_win=1.4, draw=4.5, away_win=7.0),
])

dataloader = SoccerDataLoader.from_snapshots(stats, odds)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')
X_fix, _, O_fix = dataloader.extract_fixtures_data()

# For the upcoming fixture the in-play columns are unknown, the pre-match ones are populated.
inplay_cols = [col for col in X_fix.columns if '__inplay__' in col]
assert X_fix[inplay_cols].isna().all().all()
assert X_fix[['home_points_avg', 'away_points_avg']].notna().all().all()
```

So a pre-match model should be trained on the pre-match features only — the fixed identity columns and the `preplay` features,
i.e. the columns whose name has no moment suffix. The full end-to-end flow, using the offline
[`DummySoccerDataLoader`][sportsbet.datasets.DummySoccerDataLoader]:

```python
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation import ClassifierBettor, backtest

dataloader = DummySoccerDataLoader(param_grid={'league': ['England']})
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')
X_fix, _, O_fix = dataloader.extract_fixtures_data()

# Keep only the pre-match numeric features (no in-play snapshots).
pre_match = [col for col in X_train.columns if '__' not in col and X_train[col].dtype == float]

bettor = ClassifierBettor(classifier=make_pipeline(SimpleImputer(), KNeighborsClassifier(3)))
backtest(bettor, X_train[pre_match], Y_train, O_train)          # evaluate on history
bettor.fit(X_train[pre_match], Y_train)
value_bets = bettor.bet(X_fix[pre_match], O_fix)                # value bets for upcoming matches
```

## Betting in-play

Once a match is live, the snapshots up to the current minute are available, so the model may use in-play features up to that
minute — but not beyond it. To bet at half-time, train with the features up to 45 minutes and serve on the live match state.

The upcoming match bundled in [`DummySoccerDataLoader`][sportsbet.datasets.DummySoccerDataLoader] is a match already in progress
at 30 minutes, so `extract_fixtures_data` returns its 30-minute snapshot populated:

```python
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation import ClassifierBettor

dataloader = DummySoccerDataLoader(param_grid={'league': ['England']})
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average')
X_fix, _, O_fix = dataloader.extract_fixtures_data()

# Use features known by the 30th minute only: pre-match plus the 30-minute snapshot.
up_to_30 = [
    col
    for col in X_train.columns
    if '__inplay__60min' not in col and '__inplay__90min' not in col and X_train[col].dtype == float
]
assert X_fix['home_goals__inplay__30min'].notna().all()   # the live match has reached 30 minutes

bettor = ClassifierBettor(classifier=make_pipeline(SimpleImputer(), KNeighborsClassifier(3)))
bettor.fit(X_train[up_to_30], Y_train)
value_bets = bettor.bet(X_fix[up_to_30], O_fix)
```

To predict a live match that is not part of the bundled feed, assemble its current state — the snapshots observed so far — and load
it with [`from_snapshots`][sportsbet.datasets.BaseDataLoader.from_snapshots] (or
[`from_dataframe`][sportsbet.datasets.BaseDataLoader.from_dataframe] for a single moment), leaving the result unresolved so it is
treated as a fixture. See [Consuming your own data](dataloader.md#consuming-your-own-data).

The key point is the same in both cases: keep the training features and the serving features in step with the moment you bet.
