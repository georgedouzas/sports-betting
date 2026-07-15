# Betting in practice

The rest of the documentation is about the library: how a dataloader is built, how a bettor is written, how the two are
driven from Python, the command line or an agent. This section is about the thing underneath it, which is betting
itself.

The guides here are read in order, but each stands on its own:

- [The theory of value betting](value_betting.md) is where to start. It is the one idea the whole library rests on:
  that you cannot beat a bookmaker by predicting matches, only by finding the odds they priced wrong.
- [The moment you bet](betting_moment.md) is the discipline that keeps a backtest honest: a model may only ever use what
  it would actually have known at the instant the bet is placed.

More guides will be added here as the library grows, including how to place the value bets it finds with real
bookmakers.
