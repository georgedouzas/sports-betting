"""Module that contains the datasets functionality of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from pathlib import Path
from typing import cast

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from ._options import get_config_path_option, get_data_path_option
from ._utils import (
    get_dataloader_cls,
    get_drop_na_thres,
    get_module,
    get_odds_type,
    get_param_grid,
    get_target_event_status,
    get_target_event_time,
    print_console,
)


@click.group()
def dataloader() -> None:
    """Use or create a dataloader."""
    return


@dataloader.command()
@get_config_path_option()
def params(config_path: str) -> None:
    """Show the available parameters to select data for a dataloader.

    It asks the data source, since a parameter grid cannot be written before it is known what exists.
    """
    mod = get_module(config_path)
    dataloader_cls = get_dataloader_cls(mod)
    if dataloader_cls is None:
        return
    stats_source, *_ = dataloader_cls().sources
    all_params = stats_source.available_params()
    cols = list({param for params in all_params for param in params})
    available_params = pd.DataFrame({col: [params.get(col, '-') for params in all_params] for col in cols})
    print_console([available_params], ['Available parameters'])


@dataloader.command()
@get_config_path_option()
@click.option('--dry-run', is_flag=True, help='Report what a preparation would fetch and cost, without doing it.')
def prepare(config_path: str, dry_run: bool) -> None:
    """Download the data a dataloader needs, so an extraction never has to."""
    mod = get_module(config_path)
    dataloader_cls = get_dataloader_cls(mod)
    if dataloader_cls is None:
        return
    param_grid = get_param_grid(mod)
    report = dataloader_cls(param_grid).prepare(dry_run=dry_run)
    summary = pd.DataFrame(
        {
            'To fetch': [len(report.to_fetch)],
            'Held': [len(report.held)],
            'Estimated cost': [', '.join(f'{name}: {cost}' for name, cost in report.estimated_cost.items()) or '-'],
        },
    )
    print_console([summary], ['Preparation' if not dry_run else 'Preparation (dry run)'])
    if report.unavailable:
        print_console([pd.DataFrame(report.unavailable)], ['Unavailable parameters'])


@dataloader.command()
@get_config_path_option()
def odds_types(config_path: str) -> None:
    """Show the odds types that can be selected to extract odds data."""
    mod = get_module(config_path)
    dataloader_cls = get_dataloader_cls(mod)
    if dataloader_cls is None:
        return
    param_grid = get_param_grid(mod)
    dataloader = dataloader_cls(param_grid)
    dataloader.prepare()
    odds_types = pd.DataFrame(dataloader.get_odds_types(), columns=['Type'])
    print_console([odds_types], ['Available odds types'])


@dataloader.command()
@get_config_path_option()
@get_data_path_option()
def training(config_path: str, data_path: str) -> None:
    """Use a dataloader to extract the training data."""
    mod = get_module(config_path)
    dataloader_cls = get_dataloader_cls(mod)
    if dataloader_cls is None:
        return
    param_grid = get_param_grid(mod)
    drop_na_thres = get_drop_na_thres(mod)
    odds_type = get_odds_type(mod)
    dataloader = dataloader_cls(param_grid)
    dataloader.prepare()
    X_train, Y_train, O_train = dataloader.extract_train_data(
        drop_na_thres=drop_na_thres,
        odds_type=odds_type,
        target_event_status=get_target_event_status(mod),
        target_event_time=get_target_event_time(mod),
    )
    Y_train = cast('pd.DataFrame', Y_train)  # supervised extraction always returns targets
    has_odds = O_train is not None and not O_train.empty
    print_console(
        [X_train, Y_train] + ([O_train] if has_odds else []),
        ['Training input data', 'Training output data'] + (['Training odds data'] if has_odds else []),
    )
    if data_path is not None:
        (Path(data_path) / 'sports-betting-data').mkdir(parents=True, exist_ok=True)
        X_train.to_csv(Path(data_path) / 'sports-betting-data' / 'X_train.csv')
        Y_train.to_csv(Path(data_path) / 'sports-betting-data' / 'Y_train.csv')
        if has_odds:
            O_train.to_csv(Path(data_path) / 'sports-betting-data' / 'O_train.csv')


@dataloader.command()
@get_config_path_option()
@get_data_path_option()
def fixtures(config_path: str, data_path: str) -> None:
    """Use a dataloader to extract the fixtures data."""
    console = Console()
    mod = get_module(config_path)
    dataloader_cls = get_dataloader_cls(mod)
    if dataloader_cls is None:
        return
    param_grid = get_param_grid(mod)
    drop_na_thres = get_drop_na_thres(mod)
    odds_type = get_odds_type(mod)
    dataloader = dataloader_cls(param_grid)
    dataloader.prepare()
    dataloader.extract_train_data(
        drop_na_thres=drop_na_thres,
        odds_type=odds_type,
        target_event_status=get_target_event_status(mod),
        target_event_time=get_target_event_time(mod),
    )
    X_fix, _, O_fix = dataloader.extract_fixtures_data()
    if not X_fix.empty:
        print_console([X_fix], ['Fixtures input data'])
        if O_fix is not None and not O_fix.empty:
            print_console([O_fix], ['Fixtures odds data'])
        if data_path is not None:
            (Path(data_path) / 'sports-betting-data').mkdir(parents=True, exist_ok=True)
            X_fix.to_csv(Path(data_path) / 'sports-betting-data' / 'X_fix.csv')
            if O_fix is not None and not O_fix.empty:
                O_fix.to_csv(Path(data_path) / 'sports-betting-data' / 'O_fix.csv')
    else:
        warning = Panel.fit(
            '[bold red]Fixtures data were empty',
        )
        console.print(warning)
