# Contract: the MCP server

`sportsbet-mcp` exposes the library to an assistant. The assistant lives **outside** the package — no model, no model
key and no model call is added to it.

It is an **optional extra**:

```bash
pip install 'sportsbet[mcp]'
```

## Every tool takes a config path

The tools do not take JSON describing a dataloader. They take the path to the **same Python config module the CLI
reads** (see [cli.md](cli.md)).

That choice does three things at once (research D3): the CLI and the server cannot drift, because there is one contract
and one `get_dataloader`; **the credential never enters a tool argument**, so it is never logged or shown in a
transcript; and the assistant can *write* the config, which is a thing it is good at — leaving the user with a file they
can read, edit and keep.

## The tools

| Tool | Spends? | Does |
| --- | --- | --- |
| `available_params` | **no** | What leagues and seasons exist, asked of the sources. Downloads nothing. |
| `estimate_preparation` | **no** | What a preparation would fetch, and **exactly what it would cost**. |
| `prepare` | **yes** | Downloads. **Requires the cost to have been confirmed** — see below. |
| `extract_train_data` | no | The training data of prepared data. |
| `extract_fixtures_data` | no | The upcoming games. |
| `backtest` | no | Backtests the config's bettor. |
| `bet` | no | The value bets for the fixtures. |

## The spending rule

**This is the load-bearing part of the contract.**

An odds vendor charges per request, and a single NBA season quotes at **17,722 credits** — measured, not hypothetical.
An assistant trying to be helpful can spend real money in one tool call, and the user finds out afterwards.

So `prepare` cannot spend unless the caller has been told the price and says it back:

```text
estimate_preparation(config) -> {to_fetch: 898, held: 0, cost: {'odds_api': 17722}}
prepare(config, confirm_cost=17722)        # runs
prepare(config)                            # refused: the cost was never confirmed
prepare(config, confirm_cost=100)          # refused: the real cost is 17722
```

The estimate is **exact and free**. `prepare(dry_run=True)` fetches the free statistics, derives the schedule from them,
and prices the odds items precisely — spending nothing. So there is no cost to making this mandatory, and no excuse for
skipping it.

**A tool description that merely asks the model to check first is not a guardrail.** If the only thing between a user and
17,722 credits is a sentence in a docstring, the money gets spent eventually. The refusal is in the code.

A preparation that costs **nothing** — every source free, or everything already held — needs no confirmation. The rule
exists to prevent surprise spending, not to add ceremony to a free download.
