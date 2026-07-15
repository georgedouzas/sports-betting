# CLI

The command line reaches everything the Python API reaches. A command is told what to do in its own arguments, so there
is no configuration file to write first. `--stats` and `--odds` say where the data comes from, and extracting it downloads it. A model of your own is Python, so you build it and name it: `--model models.py:BETTOR`.

```bash
# Extract training data
sportsbet data training --stats football-data --odds football-data \
  --league Italy --division 1 --year 2024 --odds-type market_maximum

# Backtest a model
sportsbet model backtest --stats football-data --odds football-data \
  --league Italy --division 1 --year 2024 --odds-type market_maximum --model logistic

# Value bets for the upcoming matches
sportsbet model bet --stats football-data --odds football-data \
  --league Italy --division 1 --odds-type market_maximum --model logistic
```

The full command reference below is generated from the tool itself, so it never falls out of date.

::: mkdocs-click
    :module: sportsbet.cli
    :command: main
    :prog_name: sportsbet
    :depth: 1
