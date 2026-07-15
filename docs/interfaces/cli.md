# CLI

The command line reaches everything the Python API reaches, and the commands mirror it: `dataloader` selects, downloads
and extracts the data, and `evaluation` backtests, fits and bets with a model on it. A command is told what to do in its
own arguments, so there is no configuration file to write first. `--stats` and `--odds` say where the data comes from,
and a source that needs a key reads it from the environment, so the key never enters a command. A model of your own is
Python, so you build it and name it: `--model models.py:BETTOR`.

`dataloader train extract` downloads the seasons once and saves the dataloader to a file; the evaluation commands read
that file with `--dataloader`, so the data is downloaded once and the model is trained once.

```bash
# Download once and save the dataloader
sportsbet dataloader train extract --stats football-data --odds football-data \
  --league Italy --division 1 --year 2024 --odds-type market_maximum -o italy.pkl

# Backtest a model on the saved data
sportsbet evaluation backtest --dataloader italy.pkl --model logistic --betting-market draw --cv 4

# Fit the model once and save it
sportsbet evaluation fit --dataloader italy.pkl --model logistic --betting-market draw -o model.pkl

# Value bets for the upcoming matches, through the fitted model
sportsbet evaluation bet --dataloader italy.pkl --bettor model.pkl
```

Before a param grid exists, `dataloader params` shows what a source publishes and `dataloader odds-types` shows the odds
a selection carries. The full command reference below is generated from the tool itself, so it never falls out of date.

::: mkdocs-click
    :module: sportsbet.cli
    :command: main
    :prog_name: sportsbet
    :depth: 1
