"""Module that contains the datasets functionality of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from ._options import get_dataloader_config_path_option
from ._utils import get_dataloader, get_module, get_train_params, print_console


@click.group()
def dataloader() -> None:
    """Use or create a dataloader."""
    return


@dataloader.command()
@get_dataloader_config_path_option()
def params(dataloader_config_path: str) -> None:
    """Show the available parameters to select data for a dataloader."""
    mod = get_module(dataloader_config_path)
    dataloader = get_dataloader(mod)
    if dataloader is None:
        return
    all_params = dataloader.get_all_params()
    cols = list({param for params in all_params for param in params})
    available_params = pd.DataFrame({col: [params.get(col, '-') for params in all_params] for col in cols})
    print_console([available_params], ['Available parameters'])


@dataloader.command()
@get_dataloader_config_path_option()
def odds_types(dataloader_config_path: str) -> None:
    """List the odds types that can be selected to extract odds data."""
    mod = get_module(dataloader_config_path)
    dataloader = get_dataloader(mod)
    if dataloader is None:
        return
    odds_types = pd.DataFrame(dataloader.get_odds_types(), columns=['Type'])
    print_console([odds_types], ['Available odds types'])


@dataloader.command()
@get_dataloader_config_path_option()
def training(dataloader_config_path: str) -> None:
    """Use a dataloader to extract the training data."""
    mod = get_module(dataloader_config_path)
    dataloader = get_dataloader(mod)
    if dataloader is None:
        return
    train_params = get_train_params(mod)
    X_train, Y_train, O_train = dataloader.extract_train_data(**train_params)
    if mod is not None:
        path = Path(dataloader_config_path).parent / mod.MAIN.get('path')
        path.parent.mkdir(parents=True, exist_ok=True)
        dataloader.save(path)
        print_console(
            [X_train, Y_train] + ([O_train] if O_train is not None else []),
            ['Training input data', 'Training output data'] + (['Training odds data'] if O_train is not None else []),
        )


@dataloader.command()
@get_dataloader_config_path_option()
def fixtures(dataloader_config_path: str) -> None:
    """Use a dataloader to extract the fixtures data."""
    console = Console()
    mod = get_module(dataloader_config_path)
    dataloader = get_dataloader(mod)
    if dataloader is None:
        return
    if not hasattr(dataloader, 'train_data_'):
        warning = Panel.fit(
            '[bold red]No information was extracted. Dataloader has not be used yet to extract training data.',
        )
        console.print(warning)
        return
    if mod is not None:
        param_grid = mod.OPTIONAL.get('param_grid')
        train_params = get_train_params(mod)
        if param_grid != dataloader.param_grid or train_params != {
            'drop_na_thres': dataloader.drop_na_thres_,
            'odds_type': dataloader.odds_type_,
        }:
            warning = Panel.fit(
                '[bold red]Configuration file has been modified. Extract again the training data.',
            )
            console.print(warning)
            return
    X_fix, _, O_fix = dataloader.extract_fixtures_data()
    if mod is not None:
        path = Path(dataloader_config_path).parent / mod.MAIN.get('path')
        path.parent.mkdir(parents=True, exist_ok=True)
        dataloader.save(path)
    if not X_fix.empty:
        print_console([X_fix], ['Fixtures input data'])
        if O_fix is not None and not O_fix.empty:
            print_console([O_fix], ['Fixtures odds data'])
    else:
        warning = Panel.fit(
            '[bold red]Fixtures data were empty',
        )
        console.print(warning)
