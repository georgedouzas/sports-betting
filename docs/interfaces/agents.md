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

## What it costs

Extracting the data downloads it, and for a paid odds feed that spends. There is no gate in front of that — a metered
key is a metered key — so the thing worth telling an agent is which sources are free and which are not:

- The statistics feeds — football-data, the EuroLeague, the NBA — are **free**.
- The odds are free only from football-data. `odds-api` buys prices per request, so what an NBA or in-play selection
  costs is between whoever holds the key and the vendor.

An agent that knows this asks before pointing a paid key at a large selection, rather than after.

## The tools

| Tool | Does |
| --- | --- |
| `available_params` | What leagues, divisions and seasons exist. Downloads nothing. |
| `extract_train_data` | Downloads and returns the training data. |
| `extract_fixtures_data` | Downloads and returns the games not yet played. |
| `backtest` | Backtests a betting model. |
| `bet` | The value bets for the fixtures. |

Every one of them takes `stats` and, optionally, `odds` — a dataloader does not choose where its data comes from, you
do.

A model is named: `odds-comparison` or `logistic`, or one of your own as `models.py:BETTOR`, since no set of arguments
can describe a scikit-learn estimator. That is the thing an agent can do which no flag can: **write the estimator**, run
it, and tell you whether it was any good.

Everything the command line can do, an agent can do, because they are told the same things in the same way — which,
since the command line reaches every sport and every source, means all of it.
