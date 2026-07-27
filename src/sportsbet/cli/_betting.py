"""Run the evaluation commands from the command line."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from sklearn.model_selection import TimeSeriesSplit

from ..dataloaders import load_dataloader
from ..evaluation import backtest as run_backtest
from ..evaluation import load_bettor, save_bettor
from ._building import _build_modelled, _report_errors
from ._options import BACKTEST, DATALOADER, MODEL, OUTPUT, options
from ._utils import _print_console


def _require_model(selection: dict[str, object]) -> None:
    """Require a model, raising a usage error when none was named."""
    if not selection.get('model'):
        msg = 'A model is needed. Name a ready-made one or one of your own with `--model`.'
        raise click.UsageError(msg)


@click.group()
def evaluation() -> None:
    """Backtest, fit and bet with a model on a saved dataloader."""
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
    _require_model(selection)
    with _report_errors(), _build_modelled(selection) as bettor:
        if bettor is None:
            return
        X_train, Y_train, O_train = load_dataloader(dataloader_path).extract_train_data()
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
        _print_console([results], ['Backtesting results'])
        if data_path is not None:
            written = Path(data_path) / 'sports-betting-data'
            written.mkdir(parents=True, exist_ok=True)
            results.to_csv(written / 'backtesting_results.csv', index=False)


@evaluation.command()
@options(DATALOADER, MODEL)
@click.option('--output', '-o', 'output', required=True, type=click.Path(), help='Where to save the fitted model.')
def fit(dataloader_path: str, output: str, **selection: object) -> None:
    """Fit a model on a saved dataloader's training data and save it."""
    _require_model(selection)
    with _report_errors(), _build_modelled(selection) as bettor:
        if bettor is None:
            return
        X_train, Y_train, O_train = load_dataloader(dataloader_path).extract_train_data()
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
    with _report_errors():
        loader = load_dataloader(dataloader_path)
        bettor = load_bettor(bettor_path)
        X_fix, _, O_fix = loader.extract_fixtures_data()
        if X_fix.empty or O_fix is None or O_fix.empty:
            Console().print(Panel.fit('[bold red]There are no upcoming matches to bet on.'))
            return
        value_bets = pd.DataFrame(bettor.bet(X_fix, O_fix), columns=list(bettor.betting_markets_), index=X_fix.index)
        _print_console([value_bets], ['Value bets'])
        if data_path is not None:
            written = Path(data_path) / 'sports-betting-data'
            written.mkdir(parents=True, exist_ok=True)
            value_bets.to_csv(written / 'value_bets.csv')
