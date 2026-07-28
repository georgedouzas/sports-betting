[scikit-learn]: <https://scikit-learn.org>

# User guide

Source objects define where the data comes from. A statistics source provides the match statistics and an odds source
provides the betting odds. Some are free, such as football-data.co.uk for soccer and the EuroLeague and NBA feeds for
basketball, and some need an API key, such as a paid odds feed. A statistics source covers a single sport, while an
odds source can cover one or many. See [Sources](sources.md).

Dataloader objects download the data and shape it for modelling. You build one from a statistics source and an optional
odds source. It gives you three things: the training data to fit a model on, the features on their own for exploring the
data, and the upcoming matches to bet on. See [Dataloader](dataloader.md).

Bettor objects are [scikit-learn] estimators. They backtest a betting strategy on the training data and predict the
value bets in the upcoming matches. You can use predefined bettors built from any scikit-learn classifier, or one that
finds value bets from the odds, and you can even define your own. See [Bettor](bettor.md).

Execution is how you place those bets for real. It takes one value bet a bettor found and places it at a bookmaker where
you hold an account, watching a single match and placing a single bet. This spends real money. See
[Execution](execution.md).
