[pandas Timedelta]: <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>

# The moment you bet

The [theory of value betting](value_betting.md) says what to look for. This page is about the discipline that makes the search
honest: a model may only ever use what it would actually have known at the instant the bet is placed. Get that wrong and a
backtest turns into a fantasy.

The [dataloader](../overview/user_guide/dataloader.md) turns raw matches into moment-aware `X`, `Y` and `O`, and the
[bettor](../overview/user_guide/bettor.md) turns those into a betting strategy. Here they come together end to end, and the
question that matters most once you start betting for real: which data can I actually use, and when?

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
match once it has actually reached half-time. You express the betting moment with the
[input horizon](../overview/user_guide/dataloader.md#the-input-horizon) — the `input_event_status` and `input_event_time` arguments of
`extract_train_data` — and it is applied to both the training and the fixtures data, so the two never fall out of step.

## Betting before kick-off

Before a match starts, the only thing known about it is its identity and its pre-match features. The in-play snapshots do not
exist yet.

You do not have to arrange that, and this is the part worth understanding: **the odds decide it**. A bet is placed at the moment
its price is quoted, so the features are capped at that moment automatically. The free feed publishes the price offered before
kick-off, so a model trained on it is handed pre-match features and nothing else — the half-time score is not available to a bet
struck before the match, and asking for it is refused rather than quietly granted.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

dataloader = DataLoader(
    param_grid={'league': ['England']},
    stats=SampleSoccerStats(),
    odds=SampleSoccerOdds(),
)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average', download=True)

# The sample carries the half-time score, and none of it reached the features: the odds are pre-match.
assert not [col for col in X_train.columns if '__inplay__' in col]
```

The sample season is finished, so it has no fixtures of its own. `extract_fixtures_data` still returns the same columns,
which is the point: what training and fixtures share is their **shape**, never their contents.

Ask for a later moment on purpose and it says no:

```python
dataloader.extract_train_data(
    odds_type='market_average', input_event_status='inplay', input_event_time=pd.Timedelta('45min'),
)
# ValueError: ... could not have had ...
```

That is not pedantry. The library used to allow it, and the worked example in the README backtested at a 269% return, because
the model was being shown the score at half time and asked to bet at the price offered before the match.

The rest is ordinary scikit-learn:

```python
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sportsbet.evaluation import ClassifierBettor, backtest

num = X_train.columns[X_train.dtypes == float]   # numeric features for the classifier

bettor = ClassifierBettor(classifier=make_pipeline(SimpleImputer(), KNeighborsClassifier(3)))
backtest(bettor, X_train[num], Y_train, O_train)          # evaluate on history
bettor.fit(X_train[num], Y_train)
```

Point the fitted bettor at the fixtures of a **live** source and it returns the value bets of the matches that have not
been played. See [doing this on real data](#doing-this-on-real-data).

## Betting in-play

Once a match is live, the snapshots up to the current minute exist, so a model may use the in-play features up to that minute —
but not beyond it. To bet at half time, train on the features up to 45 minutes and serve on the live state of the match.

**This needs odds that were quoted in play**, and no free feed has them: nobody recorded what the price was at minute 45. The
sample above cannot do it, and neither can football-data. A source with time-stamped prices can —
[`OddsApi`][sportsbet.sources.OddsApi] is one, and it is a paid one.

To show the mechanism without buying anything, here is a dataloader carrying its own snapshots, with a price quoted at half time:

```python
import pandas as pd
from sportsbet.dataloaders import BaseDataLoader
from sportsbet.sources import market_outcomes

MATCHES = [('2024-08-16', 'Arsenal', 'Chelsea', 2, 0), ('2024-08-23', 'Everton', 'Spurs', 1, 2)]


class LiveDataLoader(BaseDataLoader):
    """Two matches, each priced before kick-off and again at half time."""

    def _snapshots(self):
        stats, odds = [], []
        for date, home, away, home_goals, away_goals in MATCHES:
            identity = dict(date=date, league='England', division=1, year=2025, home_team=home, away_team=away)
            outcomes = market_outcomes(
                pd.Series([home_goals]), pd.Series([away_goals]), ['home_win', 'draw', 'away_win'],
            ).iloc[0]
            stats += [
                dict(**identity, event_status='preplay', event_time=pd.Timedelta('0min'), home_points_avg=2.1),
                dict(**identity, event_status='inplay', event_time=pd.Timedelta('45min'), home_goals=1),
                dict(**identity, event_status='postplay', event_time=pd.Timedelta('0min'),
                     home_goals=home_goals, **outcomes),
            ]
            for status, minutes, price in (('preplay', '0min', 1.7), ('inplay', '45min', 2.4)):
                odds.append(dict(**identity, event_status=status, event_time=pd.Timedelta(minutes),
                                 provider='live', home_win=price, draw=3.6, away_win=4.8))
        return pd.DataFrame(stats), pd.DataFrame(odds)


dataloader = LiveDataLoader()
X, Y, O = dataloader.extract_train_data(
    odds_type='live', input_event_status='inplay', input_event_time=pd.Timedelta('45min'),
)

# Buy a price at half time and the half-time score is yours to use.
assert [col for col in X.columns if '__inplay__45min' in col]
```

The score has to be carried at more than one moment for it to *be* a moment-varying feature — here at half time and again at
the whistle. A column that appears at one moment only is time-invariant, and it keeps its bare name.

The rule has not changed, and it is the same rule as before: a bet may use whatever was known when its price was quoted, and
nothing later. Pre-match odds cap the features at kick-off; half-time odds cap them at half time.

## Doing this on real data

The sample above is a real season, but a frozen one, and it never grows a new fixture. A live feed does, and the flow is
identical — the only difference is that its data comes over the network, and it only does so when you say so.

```python
from sportsbet.dataloaders import DataLoader
from sportsbet.sources import FootballDataOdds, FootballDataStats

dataloader = DataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2025]},
    stats=FootballDataStats(),
    odds=FootballDataOdds(),
)
X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_average', download=True)
X_fix, _, O_fix = dataloader.extract_fixtures_data(download=True)
```

`download` is the only thing that reaches the network, and it is `False` unless you pass it. It is incremental, so running it
again before each betting session refreshes the fixtures and the current season without re-downloading the seasons that are
already finished. Everything else on this page — the input horizon, the training and serving symmetry, the backtest — is
unchanged. See [Downloading the data](../overview/user_guide/dataloader.md#downloading-the-data).

If the upstream feed corrects a season that has already finished, that will not be picked up on its own — the store has no
reason to look at data it considers done. Ask it to, with `download='refresh'`. See
[What happens when the upstream data changes](../overview/user_guide/dataloader.md#what-happens-when-the-upstream-data-changes).

One honest limitation. The free feed carries **pre-match closing odds**, so the in-play flow above trains and predicts correctly
but cannot be *backtested* against a real in-play price: nobody recorded what the odds were at minute 45. To backtest an in-play
bet you need a source with time-stamped prices, injected as the dataloader's [odds source](../overview/user_guide/dataloader.md#sources).
