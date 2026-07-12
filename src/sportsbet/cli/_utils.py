"""Module that contains the utilities functions of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from types import ModuleType

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from sklearn.model_selection import TimeSeriesSplit

from .._config import ConfigError, read_bettor, read_dataloader, read_module
from ..datasets import BaseDataLoader
from ..evaluation._base import BaseBettor


def get_module(config_path: str) -> ModuleType | None:
    """Get the configuration file as module."""
    try:
        return read_module(config_path)
    except ConfigError as error:
        Console().print(Panel.fit(f'[bold red]{error}'))
        return None


def get_dataloader(mod: ModuleType | None) -> BaseDataLoader | None:
    """Get the dataloader a configuration hands over.

    It is a dataloader, not a class, because only a built one carries its sources, and only a source carries a key. That
    is what lets every sport and every source reach the command line.
    """
    if mod is None:
        return None
    try:
        return read_dataloader(mod)
    except ConfigError as error:
        Console().print(Panel.fit(f'[bold red]{error}'))
        return None


def get_drop_na_thres(mod: ModuleType | None) -> float:
    if mod is not None and hasattr(mod, 'DROP_NA_THRES'):
        return mod.DROP_NA_THRES
    return 0.0


def get_odds_type(mod: ModuleType | None) -> str | None:
    if mod is not None and hasattr(mod, 'ODDS_TYPE'):
        return mod.ODDS_TYPE
    return None


def get_target_event_status(mod: ModuleType | None) -> str | None:
    """Get the target event status (`preplay`/`inplay`/`postplay`)."""
    if mod is not None and hasattr(mod, 'TARGET_EVENT_STATUS'):
        return mod.TARGET_EVENT_STATUS
    return None


def get_target_event_time(mod: ModuleType | None) -> pd.Timedelta | None:
    """Get the target in-play event time; ignored unless the status is `inplay`."""
    if mod is not None and hasattr(mod, 'TARGET_EVENT_TIME') and mod.TARGET_EVENT_TIME is not None:
        return pd.Timedelta(mod.TARGET_EVENT_TIME)
    return None


def get_bettor(mod: ModuleType | None) -> BaseBettor | None:
    """Get the bettor."""
    if mod is None:
        return None
    try:
        return read_bettor(mod)
    except ConfigError as error:
        Console().print(Panel.fit(f'[bold red]{error}'))
        return None


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
