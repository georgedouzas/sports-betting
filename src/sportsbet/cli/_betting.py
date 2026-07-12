"""Module that contains the evaluation functionality of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from sklearn.model_selection import TimeSeriesSplit

from ..evaluation import backtest as run_backtest
from ._options import EXTRACTION, MODEL, OUTPUT, SELECTION, options
from ._utils import extraction, modelled, print_console, selected


@click.group()
def model() -> None:
    """Backtest a betting model and predict the value bets."""
    return


@model.command()
@options(SELECTION, EXTRACTION, MODEL, OUTPUT)
def backtest(cv: int, n_jobs: int, verbose: int, data_path: str | None, **selection: object) -> None:
    """Backtest a betting model."""
    with selected(selection) as dataloader, modelled(selection) as bettor:
        if dataloader is None or bettor is None:
            return
        dataloader.prepare()
        X_train, Y_train, O_train = dataloader.extract_train_data(**extraction(selection))
        if O_train is None or O_train.empty:
            Console().print(Panel.fit('[bold red]There are no odds, so there is nothing to backtest against.'))
            return
        results = run_backtest(
            bettor,
            X_train,
            Y_train,
            O_train,
            cv=TimeSeriesSplit(cv),
            n_jobs=n_jobs,
            verbose=verbose,
        )
        print_console([results], ['Backtesting results'])
        if data_path is not None:
            written = Path(data_path) / 'sports-betting-data'
            written.mkdir(parents=True, exist_ok=True)
            results.to_csv(written / 'backtesting_results.csv')


@model.command()
@options(SELECTION, EXTRACTION, MODEL, OUTPUT)
def bet(cv: int, n_jobs: int, verbose: int, data_path: str | None, **selection: object) -> None:
    """Predict the value bets of the games that have not been played yet."""
    with selected(selection) as dataloader, modelled(selection) as bettor:
        if dataloader is None or bettor is None:
            return
        dataloader.prepare()
        X_train, Y_train, O_train = dataloader.extract_train_data(**extraction(selection))
        X_fix, _, O_fix = dataloader.extract_fixtures_data()
        if X_fix.empty or O_fix is None or O_fix.empty:
            Console().print(Panel.fit('[bold red]Fixtures data were empty.'))
            return
        bettor.fit(X_train, Y_train, O_train)
        value_bets = pd.DataFrame(
            bettor.bet(X_fix, O_fix),
            columns=list(bettor.betting_markets_),
            index=X_fix.index,
        )
        print_console([value_bets], ['Value bets'])
        if data_path is not None:
            written = Path(data_path) / 'sports-betting-data'
            written.mkdir(parents=True, exist_ok=True)
            value_bets.to_csv(written / 'value_bets.csv')
