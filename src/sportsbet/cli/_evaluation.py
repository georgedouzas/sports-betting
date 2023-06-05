"""Module that contains the evaluation functionality of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from sklearn.utils.validation import NotFittedError, check_is_fitted

from ._options import get_bettor_config_path_option, get_dataloader_config_path_option
from ._utils import get_backtesting_params, get_bettor, get_dataloader, get_module, print_console


@click.group()
def bettor() -> None:
    """Backtest a bettor and predict the value bets."""
    return


@bettor.command()
@get_bettor_config_path_option()
@get_dataloader_config_path_option()
def backtest(bettor_config_path: str, dataloader_config_path: str) -> None:
    """Apply backtesting to the bettor."""
    dataloader_mod = get_module(dataloader_config_path)
    dataloader = get_dataloader(dataloader_mod)
    if dataloader is None:
        return
    bettor_mod = get_module(bettor_config_path)
    bettor = get_bettor(bettor_mod)
    if bettor is None:
        return
    if not hasattr(dataloader, 'train_data_'):
        console = Console()
        warning = Panel.fit(
            '[bold red]Dataloader has not been used to extract training data. Backtesting of bettor is not possible.',
        )
        console.print(warning)
        return
    X, Y, O = dataloader.train_data_
    if O is None:
        console = Console()
        warning = Panel.fit(
            '[bold red]Dataloader does not support odds data. Backtesting of bettor is not possible.',
        )
        console.print(warning)
        return
    backtesting_params = get_backtesting_params(bettor_mod)
    bettor.backtest(X, Y, O, **backtesting_params, refit=True)
    if bettor_mod is not None:
        path = Path(bettor_config_path).parent / bettor_mod.MAIN.get('path')
        path.parent.mkdir(parents=True, exist_ok=True)
        bettor.save(path)
        print_console([bettor.backtest_results_], ['Backtesting results'])


@bettor.command()
@get_bettor_config_path_option()
@get_dataloader_config_path_option()
def bet(bettor_config_path: str, dataloader_config_path: str) -> None:
    """Get value bets."""
    dataloader_mod = get_module(dataloader_config_path)
    dataloader = get_dataloader(dataloader_mod)
    if dataloader is None:
        return
    bettor_mod = get_module(bettor_config_path)
    bettor = get_bettor(bettor_mod)
    if bettor is None:
        return
    if not hasattr(dataloader, 'fixtures_data_'):
        console = Console()
        warning = Panel.fit(
            '[bold red]Dataloader has not been used to extract fixtures data. It is '
            'not possible to estimate the value bets.',
        )
        console.print(warning)
        return
    X, _, O = dataloader.fixtures_data_
    if X.empty and O.empty:
        console = Console()
        warning = Panel.fit(
            '[bold red]Fixtures data were empty.',
        )
        console.print(warning)
        return
    try:
        check_is_fitted(bettor)
    except NotFittedError:
        bettor.fit(*dataloader.train_data_[:-1])
    value_bets = bettor.bet(X, O)
    if bettor_mod is not None:
        path = Path(bettor_config_path).parent / bettor_mod.MAIN.get('path')
        path.parent.mkdir(parents=True, exist_ok=True)
        bettor.save(path)
        print_console([value_bets], ['Value bets'])
