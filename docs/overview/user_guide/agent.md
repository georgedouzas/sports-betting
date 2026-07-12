# AI assistant

The library ships an MCP server, so an assistant can drive it: find what data exists, price a download before it spends
anything, prepare it, backtest a model, and hand back the value bets — with no Python written by you.

```bash
pip install 'sports-betting[mcp]'
```

Then point your assistant at the `sportsbet-mcp` command.

## The assistant is outside the library, and stays there

Nothing in this package calls a model, holds a model's key, or chooses one. That is deliberate.

A model is a *nondeterministic* dependency, and this library exists to produce estimators that behave the same way every
time they are cross-validated. Putting one inside would make the thing untestable at its core, and "a betting library
that generates advice with a language model" is not a thing anyone should install.

So the library is made **legible** to an assistant rather than given one. You bring your own; every model concern — the
key, the choice, the cost, the nondeterminism — stays on your side of the line.

## Every tool takes what the commands take

A tool is told what to do in its arguments, exactly as a command is — a sport, what to select from it, and where its
data comes from. **There is no configuration file**, and nothing has to be written down before anything can be run.

- **The two surfaces cannot drift.** One vocabulary, one builder. A second, differently-shaped way to describe a
  dataloader would be a second set of capabilities — which is precisely the disease that killed the GUI.
- **Your key is never an argument.** What the assistant names is the *environment variable* holding it
  (`odds_key_env`), never the key. A key passed as a tool argument is a key written into a transcript.
- **Nothing is left behind to fall out of date.** A file the assistant wrote three sessions ago, still pointing at last
  season, is a bug waiting to happen. A tool call describes itself.

## An assistant cannot spend your money by accident

**This is the part that matters.**

Odds are bought per request. A single NBA season costs **17,722 credits**. An assistant trying to be helpful could buy
them in one tool call, and you would find out afterwards.

So the price is free to ask for, and the refusal is in the **code** — not in a sentence in a tool description asking the
model to be careful:

```text
estimate_preparation(sport='basketball', leagues=['NBA'], years=[2026], stats='nba', odds='odds-api')
    -> {to_fetch: 898, cost: {'odds_api': 17722}}

prepare(...)                     -> REFUSED. The cost was never confirmed.
prepare(..., confirm_cost=100)   -> REFUSED. It costs 17722, not 100.
prepare(..., confirm_cost=17722) -> runs
```

The estimate is **exact** — it is the real list of what would be bought — and it costs **nothing**, because the free
statistics are fetched, the schedule derived from them, and the odds priced against it. So there is no reason to skip it,
and no way to.

A preparation that costs **nothing** needs no confirmation. The rule exists to stop a surprise on a bill, not to add
ceremony to a free download.

## The tools

| Tool | Spends? | Does |
| --- | --- | --- |
| `available_params` | no | What leagues and seasons exist. Downloads nothing. |
| `estimate_preparation` | no | What a preparation would fetch, and exactly what it would cost. |
| `prepare` | **yes** | Downloads. Requires the cost to be confirmed. |
| `extract_train_data` | no | The training data. |
| `extract_fixtures_data` | no | The games not yet played. |
| `backtest` | no | Backtests a betting model. |
| `bet` | no | The value bets for the fixtures. |

A model is named: `odds-comparison` or `logistic`, or one of your own as `models.py:BETTOR`, since no set of arguments
can describe a scikit-learn estimator.

Everything the command line can do, an assistant can do, because they are told the same things in the same way —
which, since the command line reaches every sport and every source, means all of it.
