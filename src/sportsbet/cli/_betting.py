"""Module that contains the evaluation functionality of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from ..evaluation import backtest as bt
from ._options import get_config_path_option, get_data_path_option
from ._utils import (
    get_bettor,
    get_cv,
    get_dataloader_cls,
    get_drop_na_thres,
    get_module,
    get_n_jobs,
    get_odds_type,
    get_param_grid,
    get_target_event_status,
    get_target_event_time,
    get_verbose,
    print_console,
)


@click.group()
def bettor() -> None:
    """Backtest a bettor and predict the value bets."""
    return


@bettor.command()
@get_config_path_option()
@get_data_path_option()
def backtest(config_path: str, data_path: str) -> None:
    """Apply backtesting to the bettor."""
    mod = get_module(config_path)
    dataloader_cls = get_dataloader_cls(mod)
    if dataloader_cls is None:
        return
    param_grid = get_param_grid(mod)
    drop_na_thres = get_drop_na_thres(mod)
    odds_type = get_odds_type(mod)
    dataloader = dataloader_cls(param_grid)
    X_train, Y_train, O_train = dataloader.extract_train_data(
        drop_na_thres=drop_na_thres,
        odds_type=odds_type,
        target_event_status=get_target_event_status(mod),
        target_event_time=get_target_event_time(mod),
    )
    bettor = get_bettor(mod)
    if bettor is None:
        return
    if O_train is None or O_train.empty:
        console = Console()
        warning = Panel.fit(
            '[bold red]Dataloader does not support odds data. Backtesting of bettor is not possible.',
        )
        console.print(warning)
        return
    backtesting_results = bt(
        bettor,
        X_train,
        Y_train,
        O_train,
        cv=get_cv(mod),
        n_jobs=get_n_jobs(mod),
        verbose=get_verbose(mod),
    )
    if mod is not None:
        print_console([backtesting_results], ['Backtesting results'])
        if data_path is not None:
            (Path(data_path) / 'sports-betting-data').mkdir(parents=True, exist_ok=True)
            backtesting_results.to_csv(Path(data_path) / 'sports-betting-data' / 'backtesting_results.csv')


@bettor.command()
@get_config_path_option()
@get_data_path_option()
def bet(config_path: str, data_path: str) -> None:
    """Get value bets."""
    mod = get_module(config_path)
    dataloader_cls = get_dataloader_cls(mod)
    if dataloader_cls is None:
        return
    param_grid = get_param_grid(mod)
    drop_na_thres = get_drop_na_thres(mod)
    odds_type = get_odds_type(mod)
    dataloader = dataloader_cls(param_grid)
    X_train, Y_train, O_train = dataloader.extract_train_data(
        drop_na_thres=drop_na_thres,
        odds_type=odds_type,
        target_event_status=get_target_event_status(mod),
        target_event_time=get_target_event_time(mod),
    )
    bettor = get_bettor(mod)
    if bettor is None:
        return
    X_fix, _, O_fix = dataloader.extract_fixtures_data()
    if X_fix.empty or O_fix is None or O_fix.empty:
        console = Console()
        warning = Panel.fit(
            '[bold red]Fixtures data were empty.',
        )
        console.print(warning)
        return
    bettor.fit(X_train, Y_train, O_train)
    value_bets = pd.DataFrame(
        bettor.bet(X_fix, O_fix),
        columns=list(bettor.betting_markets_),
        index=X_fix.index,
    )
    if mod is not None:
        print_console([value_bets], ['Value bets'])
        if data_path is not None:
            (Path(data_path) / 'sports-betting-data').mkdir(parents=True, exist_ok=True)
            value_bets.to_csv(Path(data_path) / 'sports-betting-data' / 'value_bets.csv')
