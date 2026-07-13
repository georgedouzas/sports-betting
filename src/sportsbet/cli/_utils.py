"""Module that contains the utilities functions of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import pandas as pd
from rich.console import Console
from rich.panel import Panel

from .._selection import SelectionError, build_bettor, build_dataloader
from ..datasets import BaseDataLoader, NotPreparedError
from ..evaluation import BaseBettor

SELECTED = (
    'sport',
    'leagues',
    'divisions',
    'years',
    'stats',
    'odds',
    'odds_key_env',
    'odds_markets',
    'odds_regions',
    'odds_moments',
    'store',
    'aliases',
    'max_unmatched_rate',
)
EXTRACTED = (
    'drop_na_thres',
    'odds_type',
    'learning_type',
    'target_event_status',
    'target_event_time',
    'input_event_status',
    'input_event_time',
)
MODELLED = ('model', 'alpha', 'init_cash', 'stake', 'betting_markets', 'model_odds_types')
TIMED = ('target_event_time', 'input_event_time')


def _listed(value: object) -> object:
    """Return a repeated option as a list, since it arrives as a tuple."""
    return list(value) if isinstance(value, tuple) else value


def _told(selection: dict[str, object], names: tuple[str, ...]) -> dict[str, Any]:
    """Return what a command was told, of the things a builder asks for."""
    return cast('dict[str, Any]', {name: _listed(selection[name]) for name in names if name in selection})


@contextmanager
def selected(selection: dict[str, object]) -> Iterator[BaseDataLoader | None]:
    """Build the dataloader a command was told to use, or say what is wrong with what it was told."""
    try:
        yield build_dataloader(**_told(selection, SELECTED))
    except SelectionError as error:
        Console().print(Panel.fit(f'[bold red]{error}'))
        yield None


@contextmanager
def modelled(selection: dict[str, object]) -> Iterator[BaseBettor | None]:
    """Build the bettor a command was told to use, or say what is wrong with what it was told."""
    try:
        yield build_bettor(**_told(selection, MODELLED))
    except SelectionError as error:
        Console().print(Panel.fit(f'[bold red]{error}'))
        yield None


def extraction(selection: dict[str, object]) -> dict[str, Any]:
    """Return how a command was told to extract."""
    settings = {name: value for name, value in _told(selection, EXTRACTED).items() if value is not None}
    for name in TIMED:
        if settings.get(name) is not None:
            settings[name] = pd.Timedelta(settings[name])
    return settings


@contextmanager
def reported() -> Iterator[None]:
    """Say what went wrong, rather than showing where.

    A stack trace is what the library tells a programmer. A command is not talking to one.
    """
    try:
        yield
    except NotPreparedError:
        Console().print(
            Panel.fit(
                '[bold red]The data has not been downloaded.\n\n[/bold red]Run `sportsbet data prepare` first. It is '
                'a separate step because it is the only one that costs anything, and nothing should spend your money '
                'without being asked to.',
            ),
        )
        raise SystemExit(1) from None
    except (SelectionError, ValueError) as error:
        Console().print(Panel.fit(f'[bold red]{error}'))
        raise SystemExit(1) from None


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
