"""Module that contains the evaluation commands of the CLI."""

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
from ..evaluation import load_bettor, save_bettor
from ._options import BACKTEST, DATALOADER, MODEL, OUTPUT, options
from ._utils import load_dataloader, modelled, print_console, reported


def _needs_model(selection: dict[str, object]) -> None:
    """Say a model is needed, since backtesting and fitting cannot happen without one."""
    if not selection.get('model'):
        msg = 'A model is needed. Name a ready-made one or one of your own with `--model`.'
        raise click.UsageError(msg)


@click.group()
def evaluation() -> None:
    """Backtest, fit and bet with a model on a saved dataloader.

    Each command reads the file `dataloader train extract` writes, so the data is downloaded once and, for `fit` and
    `bet`, the model is trained once.
    """
    return


@evaluation.command()
@options(DATALOADER, MODEL, BACKTEST, OUTPUT)
def backtest(
    dataloader_path: str,
    cv: int,
    n_jobs: int,
    verbose: int,
    data_path: str | None,
    **selection: object,
) -> None:
    """Backtest a model on a saved dataloader's training data."""
    _needs_model(selection)
    with reported(), modelled(selection) as bettor:
        if bettor is None:
            return
        _, (X_train, Y_train, O_train) = load_dataloader(dataloader_path)
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
            results.to_csv(written / 'backtesting_results.csv', index=False)


@evaluation.command()
@options(DATALOADER, MODEL)
@click.option('--output', '-o', 'output', required=True, type=click.Path(), help='Where to save the fitted model.')
def fit(dataloader_path: str, output: str, **selection: object) -> None:
    """Fit a model on a saved dataloader's training data and save it."""
    _needs_model(selection)
    with reported(), modelled(selection) as bettor:
        if bettor is None:
            return
        _, (X_train, Y_train, O_train) = load_dataloader(dataloader_path)
        bettor.fit(X_train, Y_train, O_train)
        save_bettor(bettor, output)
        Console().print(f'Saved the fitted model to [bold]{output}[/bold].')


@evaluation.command()
@options(DATALOADER)
@click.option(
    '--bettor',
    '-b',
    'bettor_path',
    required=True,
    type=click.Path(exists=True),
    help='A model saved by `fit`.',
)
@click.option('--output', '-o', 'data_path', type=click.Path(), help='A directory to write the bets to, as CSV.')
def bet(dataloader_path: str, bettor_path: str, data_path: str | None) -> None:
    """Predict the value bets of the upcoming matches with a model saved by `fit`."""
    with reported():
        loader, _ = load_dataloader(dataloader_path)
        bettor = load_bettor(bettor_path)
        X_fix, _, O_fix = loader.extract_fixtures_data()
        if X_fix.empty or O_fix is None or O_fix.empty:
            Console().print(Panel.fit('[bold red]There are no upcoming matches to bet on.'))
            return
        value_bets = pd.DataFrame(bettor.bet(X_fix, O_fix), columns=list(bettor.betting_markets_), index=X_fix.index)
        print_console([value_bets], ['Value bets'])
        if data_path is not None:
            written = Path(data_path) / 'sports-betting-data'
            written.mkdir(parents=True, exist_ok=True)
            value_bets.to_csv(written / 'value_bets.csv')
