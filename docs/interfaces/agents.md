# Agents

The library ships an MCP server, so an agent can drive it: find what data exists, download it, backtest a model, and hand
back the value bets — with no Python written by you.

```bash
pip install 'sports-betting[mcp]'
```

Then point your agent at the `sportsbet-mcp` command.

## The agent is outside the library, and stays there

Nothing in this package calls a model, holds a model's key, or chooses one. That is deliberate.

A model is a *nondeterministic* dependency, and this library exists to produce estimators that behave the same way every
time they are cross-validated. Putting one inside would make the thing untestable at its core, and "a betting library
that generates advice with a language model" is not a thing anyone should install.

So the library is made **legible** to an agent rather than given one. You bring your own; every model concern — the key,
the choice, the cost, the nondeterminism — stays on your side of the line.

## Every tool takes what the commands take

A tool is told what to do in its arguments, exactly as a command is — what to select, and where its data comes from.
**There is no configuration file**, and nothing has to be written down before anything can be run.

- **The two surfaces cannot drift.** One vocabulary, one builder. A second, differently-shaped way to describe a
  dataloader would be a second set of capabilities — which is precisely the disease that killed the GUI.
- **Your key is never an argument.** What the agent names is the *environment variable* holding it (`odds_key_env`),
  never the key. A key passed as a tool argument is a key written into a transcript.
- **Nothing is left behind to fall out of date.** A file the agent wrote three sessions ago, still pointing at last
  season, is a bug waiting to happen. A tool call describes itself.

## Nothing downloads unless it was asked to

Odds are often bought per request, so an agent trying to be helpful could spend on your behalf and you would find out
afterwards. It cannot: **`download` defaults to `False`**, and no other argument fetches anything.

Ask for data that is not held and the tool refuses, and says what getting it would take:

```text
extract_train_data(leagues=['Germany', 'Italy', 'France'], divisions=[1, 2], years=[2021, 2022, 2023, 2024],
                   stats='football-data', odds='football-data')

    -> The data has not been downloaded. Pass `download=True` to get it.
       Requests to make: football_data 25. Items held: 0.

extract_train_data(..., download=True)

    -> runs, and makes those 25 requests
```

The count is the real list of what would be fetched, and learning it costs nothing. What those requests are *worth* is
between you and the vendor you buy them from — a vendor sets its own prices and changes them, so the library reports the
fact and leaves the price to you.

## The tools

| Tool | Fetches? | Does |
| --- | --- | --- |
| `available_params` | never | What leagues, divisions and seasons exist. |
| `extract_train_data` | only if told | The training data. |
| `extract_fixtures_data` | only if told | The games not yet played. |
| `backtest` | only if told | Backtests a betting model. |
| `bet` | only if told | The value bets for the fixtures. |

Every one of them takes `stats` and, optionally, `odds` — a dataloader does not choose where its data comes from, you
do — and every one takes `download`, which is the only thing that reaches the network.

A model is named: `odds-comparison` or `logistic`, or one of your own as `models.py:BETTOR`, since no set of arguments
can describe a scikit-learn estimator. That is the thing an agent can do which no flag can: **write the estimator**, run
it, and tell you whether it was any good.

Everything the command line can do, an agent can do, because they are told the same things in the same way — which,
since the command line reaches every sport and every source, means all of it.
