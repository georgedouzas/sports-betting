"""Module that contains the utilities functions of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, Union

import pandas as pd
from rich.console import Console
from rich.panel import Panel

from ..datasets import DummySoccerDataLoader, SoccerDataLoader
from ..evaluation import ClassifierBettor, OddsComparisonBettor

DataLoader = Union[SoccerDataLoader, DummySoccerDataLoader]
Bettor = Union[ClassifierBettor, OddsComparisonBettor]


def get_module(config_path: str) -> ModuleType | None:
    """Get the configuration file as module."""
    console = Console()
    if not Path(config_path).exists():
        warning = Panel.fit(
            '[bold red]Path of configuration file does not exist.',
        )
        console.print(warning)
        return None
    spec = spec_from_file_location('mod', config_path)
    if spec is not None:
        mod = module_from_spec(spec)
        if spec.loader is not None:
            spec.loader.exec_module(mod)
            return mod
    else:
        warning = Panel.fit(
            '[bold red]Configuration file does not have the expected content.',
        )
        console.print(warning)
    return None


def get_dataloader(mod: ModuleType | None) -> DataLoader | None:
    """Get the dataloader."""
    console = Console()
    if mod is None:
        return None
    if not hasattr(mod, 'CONFIG'):
        warning = Panel.fit(
            '[bold red]Configuration file does not have a `CONFIG` dictionary.',
        )
        console.print(warning)
        return None
    elif mod.CONFIG.get('data') is None:
        warning = Panel.fit(
            '[bold red]`CONFIG` dictionary should have a `\'data\'` key.',
        )
        console.print(warning)
        return None
    elif mod.CONFIG['data'].get('dataloader') not in (SoccerDataLoader, DummySoccerDataLoader):
        warning = Panel.fit(
            '[bold red]`CONFIG` dictionary should have a `\'dataloader\'` key and a '
            'dataloader class value in `\'data\'` key.',
        )
        console.print(warning)
        return None
    dataloader = mod.CONFIG['data']['dataloader'](mod.CONFIG['data'].get('param_grid'))
    return dataloader


def get_bettor(mod: ModuleType | None) -> Bettor | None:
    """Get the bettor."""
    console = Console()
    if mod is None:
        return None
    if not hasattr(mod, 'CONFIG'):
        warning = Panel.fit(
            '[bold red]Configuration file does not have a `CONFIG` dictionary.',
        )
        console.print(warning)
        return None
    elif mod.CONFIG.get('data') is None:
        warning = Panel.fit(
            '[bold red]`CONFIG` dictionary should have a `\'betting\'` key.',
        )
        console.print(warning)
        return None
    elif mod.CONFIG['betting'].get('bettor') not in (ClassifierBettor, OddsComparisonBettor):
        warning = Panel.fit(
            '[bold red]`CONFIG` dictionary should have a `\'bettor\'` key '
            'and a bettor class value in `\'betting\'` key.',
        )
        console.print(warning)
        return None
    bettor = mod.CONFIG['betting']['bettor'](
        **{k: v for k, v in mod.CONFIG['betting'].items() if k not in ('bettor', 'tscv', 'init_cash')},
    )
    return bettor


def get_train_params(mod: ModuleType | None) -> dict[str, Any]:
    train_params = {'drop_na_thres': 0.0, 'odds_type': None}
    if mod is not None and hasattr(mod, 'CONFIG'):
        train_params['drop_na_thres'] = mod.CONFIG['data'].get('drop_na_thres', 0.0)
        train_params['odds_type'] = mod.CONFIG['data'].get('odds_type')
    return train_params


def get_backtesting_params(mod: ModuleType | None) -> dict[str, Any]:
    backtesting_params = {'tscv': None, 'init_cash': None}
    if mod is not None and hasattr(mod, 'CONFIG'):
        backtesting_params['tscv'] = mod.CONFIG['betting'].get('tscv')
        backtesting_params['init_cash'] = mod.CONFIG['betting'].get('init_cash')
    return backtesting_params


def print_console(dfs: list[pd.DataFrame], titles: list[str]) -> None:
    """Print to the console."""
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    console = Console()
    with console.pager(styles=True):
        for df, title in zip(dfs, titles):
            formatted_title = Panel.fit(f'[bold green]{title}')
            console.print(formatted_title)
            console.print(df)
