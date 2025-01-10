"""Module that contains the utilities functions of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from sklearn.model_selection import TimeSeriesSplit

from .. import ParamGrid
from ..datasets._base import BaseDataLoader
from ..evaluation._base import BaseBettor


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


def get_dataloader_cls(mod: ModuleType | None) -> type[BaseDataLoader] | None:
    """Get the dataloader class."""
    console = Console()
    if mod is None:
        return None
    if not hasattr(mod, 'DATALOADER_CLASS'):
        warning = Panel.fit(
            '[bold red]Configuration file does not have a `DATALOADER_CLASS` variable.',
        )
        console.print(warning)
        return None
    elif not issubclass(mod.DATALOADER_CLASS, BaseDataLoader):
        warning = Panel.fit(
            '[bold red]`DATALOADER_CLASS` variable should be a `\'dataloader\'` class.',
        )
        console.print(warning)
        return None
    return mod.DATALOADER_CLASS


def get_param_grid(mod: ModuleType | None) -> ParamGrid | None:
    """Get the parameters grid class."""
    if mod is not None and hasattr(mod, 'PARAM_GRID'):
        return mod.PARAM_GRID
    return None


def get_drop_na_thres(mod: ModuleType | None) -> float:
    if mod is not None and hasattr(mod, 'DROP_NA_THRES'):
        return mod.DROP_NA_THRES
    return 0.0


def get_odds_type(mod: ModuleType | None) -> str | None:
    if mod is not None and hasattr(mod, 'ODDS_TYPE'):
        return mod.ODDS_TYPE
    return None


def get_bettor(mod: ModuleType | None) -> BaseBettor | None:
    """Get the bettor."""
    console = Console()
    if mod is None:
        return None
    if not hasattr(mod, 'BETTOR'):
        warning = Panel.fit(
            '[bold red]Configuration file does not have a `BETTOR` variable.',
        )
        console.print(warning)
        return None
    elif not hasattr(mod, 'DATALOADER_CLASS'):
        warning = Panel.fit(
            '[bold red]Configuration file does not have a `DATALOADER_CLASS` variable.',
        )
        console.print(warning)
        return None
    elif not isinstance(mod.BETTOR, BaseBettor):
        warning = Panel.fit(
            '[bold red]`BETTOR` variable should be a `\'bettor\'` object.',
        )
        console.print(warning)
        return None
    return mod.BETTOR


def get_cv(mod: ModuleType | None) -> TimeSeriesSplit | None:
    if mod is not None and hasattr(mod, 'CV'):
        return mod.CV
    return None


def get_n_jobs(mod: ModuleType | None) -> int:
    if mod is not None and hasattr(mod, 'N_JOBS'):
        return mod.N_JOBS
    return -1


def get_verbose(mod: ModuleType | None) -> int:
    if mod is not None and hasattr(mod, 'VERBOSE'):
        return mod.VERBOSE
    return 0


def print_console(dfs: list[pd.DataFrame], titles: list[str]) -> None:
    """Print to the console."""
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    console = Console()
    with console.pager(styles=True):
        for df, title in zip(dfs, titles, strict=True):
            formatted_title = Panel.fit(f'[bold green]{title}')
            console.print(formatted_title)
            console.print(df)
