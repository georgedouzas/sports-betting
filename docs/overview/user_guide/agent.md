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

## Every tool reads the same configuration the CLI reads

A tool is not handed a dataloader in its arguments. It is handed the **path** of a
[configuration file](dataloader.md), the same Python module `sportsbet` reads.

That is worth explaining, because it is the design decision the whole surface rests on.

- **The two surfaces cannot drift.** One configuration, one reader. A second, JSON-shaped way to describe a dataloader
  would be a second set of capabilities — which is precisely the disease that killed the GUI.
- **Your key never enters a tool call.** It stays in `os.environ`, read by the configuration. A key passed as a tool
  argument is a key written into a transcript.
- **The assistant can write the configuration**, which is a thing assistants are good at. You are left with a file you
  can read, edit, keep and re-run — not a conversation you have to have again.

## An assistant cannot spend your money by accident

**This is the part that matters.**

Odds are bought per request. A single NBA season costs **17,722 credits**. An assistant trying to be helpful could buy
them in one tool call, and you would find out afterwards.

So the price is free to ask for, and the refusal is in the **code** — not in a sentence in a tool description asking the
model to be careful:

```text
estimate_preparation(config)          -> {to_fetch: 898, cost: {'odds_api': 17722}}

prepare(config)                       -> REFUSED. The cost was never confirmed.
prepare(config, confirm_cost=100)     -> REFUSED. It costs 17722, not 100.
prepare(config, confirm_cost=17722)   -> runs
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
| `backtest` | no | Backtests the configuration's bettor. |
| `bet` | no | The value bets for the fixtures. |

Everything the [command line](dataloader.md) can do, an assistant can do — which, since the CLI reaches every sport and
every source, means all of it.
