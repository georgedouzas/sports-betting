"""Module that contains the utilities functions of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .._selection import SelectionError, build_bettor, build_dataloader
from ..dataloaders import BaseDataLoader
from ..evaluation import BaseBettor

SELECTED = (
    'leagues',
    'divisions',
    'years',
    'stats',
    'odds',
    'odds_key_env',
    'odds_markets',
    'odds_regions',
    'odds_moments',
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
    except (SelectionError, ValueError) as error:
        Console().print(Panel.fit(f'[bold red]{error}'))
        raise SystemExit(1) from None


def _cell(value: object) -> Text:
    """Return a value as a cell, so a table reads as a table rather than as a dump of a dataframe."""
    if value is None or value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return Text('-', style='dim')
    if isinstance(value, bool | np.bool_):
        return Text('yes', style='green') if value else Text('no', style='dim')
    if isinstance(value, float | np.floating):
        return Text(f'{value + 0.0:,.2f}')
    if isinstance(value, pd.Timestamp):
        return Text(value.strftime('%Y-%m-%d'))
    return Text(str(value))


def _columns(frame: pd.DataFrame, *, index: bool) -> tuple[list[str], list[list[Text]]]:
    """Return the headings and the cells of a frame, as they will be shown."""
    multi = isinstance(frame.index, pd.MultiIndex)
    levels = [str(name or '') for name in frame.index.names] if index else []
    headings = [*levels, *[str(column) for column in frame.columns]]
    rows = []
    for key, row in zip(frame.index, frame.to_dict('records'), strict=True):
        keys = [_cell(part) for part in (key if multi else [key])] if index else []
        rows.append([*keys, *[_cell(value) for value in row.values()]])
    return headings, rows


def _wanted(headings: list[str], rows: list[list[Text]]) -> int:
    """Return the width a table needs to be read.

    A heading may wrap onto another line, so it needs room for its longest word rather than for the whole of itself. A
    cell may not, so it needs all of itself. What ruins a table is a heading broken through the middle of a word, a
    letter to a line.
    """
    widths = [
        max([*[len(word) for word in heading.split()], *[row[position].cell_len for row in rows]])
        for position, heading in enumerate(headings)
    ]
    return sum(widths) + 2 * len(widths)


def _table(headings: list[str], rows: list[list[Text]], levels: int) -> Table:
    """Return the rows as a table, laid out across the terminal."""
    table = Table(header_style='bold', box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    for position, heading in enumerate(headings):
        below = position < levels
        table.add_column(heading, style='cyan' if below else '', justify='left' if below else 'right', no_wrap=below)
    for row in rows:
        table.add_row(*row)
    return table


def _expanded(headings: list[str], rows: list[list[Text]]) -> Table:
    """Return the rows a record at a time, for when they are too wide to be a table.

    Thirteen columns in eighty characters is not a table, it is a column of single letters. What cannot be laid out
    across the terminal is laid out down it instead, which is what a database client does with the same problem.
    """
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, show_header=False)
    table.add_column(style='bold cyan', no_wrap=True)
    table.add_column(overflow='fold')
    for position, row in enumerate(rows):
        if position:
            table.add_section()
        for heading, cell in zip(headings, row, strict=True):
            table.add_row(heading, cell)
    return table


def print_console(dfs: list[pd.DataFrame], titles: list[str], *, index: bool = True) -> None:
    """Print the results, a record at a time when they are too wide, and paged when they are too tall.

    The title is printed beside the table rather than inside it, so a narrow table does not wrap its own heading.
    """
    console = Console()
    rendered = []
    height = 0
    for frame, title in zip(dfs, titles, strict=True):
        headings, rows = _columns(frame, index=index)
        levels = len(frame.index.names) if index else 0
        if _wanted(headings, rows) <= console.size.width:
            rendered.append((title, _table(headings, rows, levels)))
            height += len(rows) + 4
        else:
            rendered.append((title, _expanded(headings, rows)))
            height += len(rows) * (len(headings) + 1)

    def show() -> None:
        for title, table in rendered:
            console.print(f'[bold green]{title}[/bold green]')
            console.print(table)

    if height > console.size.height:
        with console.pager(styles=True):
            show()
        return
    show()
