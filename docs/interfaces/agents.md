# Agents

The library ships an MCP server, so an agent can drive it. It can find what data exists, download it, backtest a model
and hand back the value bets, with no Python written by you.

```bash
pip install 'sports-betting[mcp]'
```

Then point your agent at the `sportsbet-mcp` command.

## The agent is outside the library, and stays there

Nothing in this package calls a model, holds a model's key, or chooses one. That is deliberate.

A model is a nondeterministic dependency, and this library exists to produce estimators that behave the same way every
time they are cross validated. Putting one inside would make the core untestable.

The library is built for an agent to use, not to contain one. You bring your own. Every model concern, the key, the
choice, the cost, the nondeterminism, stays on your side of the line.

## Every tool takes what the commands take

A tool is told what to do in its arguments, exactly as a command is. What to select, and where its data comes from.
There is no configuration file, and nothing has to be written down before anything runs.

* The two surfaces cannot drift. One vocabulary, one builder. A second, differently shaped way to describe a dataloader
  would be a second set of capabilities to keep in step.
* Your key is never an argument. What the agent names is the environment variable holding it, `odds_key_env`, never the
  key. A key passed as a tool argument is a key written into a transcript.
* Nothing is left behind to fall out of date. A file the agent wrote three sessions ago, still pointing at last season,
  is a bug waiting to happen. A tool call describes itself.

## What it costs

Extracting the data downloads it, and for a paid odds feed that spends. There is no gate in front of that, a metered key
is a metered key, so the thing worth telling an agent is which sources are free and which are not.

* The statistics feeds, football-data, the EuroLeague and the NBA, are free.
* The odds are free only from football-data. `odds-api` buys prices per request, so what an NBA or in play selection
  costs is between whoever holds the key and the vendor.

An agent that knows this asks before pointing a paid key at a large selection, rather than after.

## The tools

| Tool | Does |
| --- | --- |
| `available_params` | What leagues, divisions and seasons exist. Reads the source's index and no season. |
| `odds_types` | The odds types a selection carries. Downloads the selected seasons to read them. |
| `extract_exploration_data` | Downloads and returns the features on their own, with no targets or odds. |
| `extract_train_data` | Downloads and returns the training data, and saves the dataloader to `output`. |
| `extract_fixtures_data` | The games not yet played, from a saved dataloader. |
| `backtest` | Backtests a betting model on a saved dataloader. |
| `fit` | Fits a model on a saved dataloader and saves it. |
| `bet` | The value bets for the fixtures, from a saved dataloader and a fitted model. |

The first four take `stats` and, optionally, `odds`, because a dataloader does not choose where its data comes from,
you do. `extract_train_data` writes the dataloader it built to `output`, and the last four name that file as
`dataloader`, so the selection is described once.

The server also carries the execution tools, which place bets at a venue. The [execution guide][execution] explains
venues and what driving a bookmaker risks.

| Tool | Does |
| --- | --- |
| `execution_venue_info` | What a venue is, whether it cancels, and its owner's notes. |
| `execution_authenticate` | Logs in at a venue, reading each secret from the variable the venue names. |
| `execution_read_balance` | The balance and what is at stake. |
| `execution_list_markets` | The markets a venue offers on the given matches, with their prices. |
| `execution_read_status` | What the venue holds for one bet. |
| `execution_cancel` | Cancels a bet, where the venue cancels. |
| `execution_run` | Watches one event and places the model's bet at its moment. Stakes nothing without `live`. |
| `browser_navigate` | Opens a page of a bookmaker's website and returns it as an accessibility snapshot. |
| `browser_snapshot` | Returns the page, or a part of it. |
| `browser_click` | Clicks an element and returns the page it produced. |
| `browser_type` | Fills an element and returns the page it produced. |
| `browser_select` | Chooses an option and returns the page it produced. |
| `browser_fix` | Pins the locators that exploring found. |

A model is named as a Python expression, such as `OddsComparisonBettor(alpha=0.05)` or
`ClassifierBettor(LogisticRegression())`, or one of your own as `models.py:BETTOR`, since no set of arguments can
describe a scikit-learn estimator. That is the thing an agent can do which no flag can. It writes the estimator, runs
it, and tells you whether it was any good.

Everything the command line can do, an agent can do, because they are told the same things in the same way. Since the
command line reaches every sport and every source, that means all of it.

[execution]: ../overview/user_guide/execution.md
