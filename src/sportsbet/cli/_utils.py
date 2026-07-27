"""Print the command results as tables."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import math

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


def _build_cell(value: object) -> Text:
    """Return a value as a table cell."""
    if value is None or value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return Text('-', style='dim')
    if isinstance(value, bool | np.bool_):
        return Text('yes', style='green') if value else Text('no', style='dim')
    if isinstance(value, float | np.floating):
        return Text(f'{value + 0.0:,.2f}')
    if isinstance(value, pd.Timestamp):
        return Text(value.strftime('%Y-%m-%d'))
    return Text(str(value))


def _build_columns(frame: pd.DataFrame, *, index: bool) -> tuple[list[str], list[list[Text]]]:
    """Return the headings and the cells of a frame, as they will be shown."""
    multi = isinstance(frame.index, pd.MultiIndex)
    levels = [str(name or '') for name in frame.index.names] if index else []
    headings = [*levels, *[str(column) for column in frame.columns]]
    rows = []
    for key, row in zip(frame.index, frame.to_dict('records'), strict=True):
        keys = [_build_cell(part) for part in (key if multi else [key])] if index else []
        rows.append([*keys, *[_build_cell(value) for value in row.values()]])
    return headings, rows


def _measure_width(headings: list[str], rows: list[list[Text]]) -> int:
    """Return the width a table needs to be read."""
    widths = [
        max([*[len(word) for word in heading.split()], *[row[position].cell_len for row in rows]])
        for position, heading in enumerate(headings)
    ]
    return sum(widths) + 2 * len(widths)


def _build_table(headings: list[str], rows: list[list[Text]], levels: int) -> Table:
    """Return the rows as a table, laid out across the terminal."""
    table = Table(header_style='bold', box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    for position, heading in enumerate(headings):
        below = position < levels
        table.add_column(heading, style='cyan' if below else '', justify='left' if below else 'right', no_wrap=below)
    for row in rows:
        table.add_row(*row)
    return table


def _build_record_table(headings: list[str], rows: list[list[Text]]) -> Table:
    """Return the rows a record at a time, for when they are too wide to be a table."""
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, show_header=False)
    table.add_column(style='bold cyan', no_wrap=True)
    table.add_column(overflow='fold')
    for position, row in enumerate(rows):
        if position:
            table.add_section()
        for heading, cell in zip(headings, row, strict=True):
            table.add_row(heading, cell)
    return table


def _print_console(dfs: list[pd.DataFrame], titles: list[str], *, index: bool = True) -> None:
    """Print the results, a record at a time when they are too wide, and paged when they are too tall."""
    console = Console()
    rendered = []
    height = 0
    for frame, title in zip(dfs, titles, strict=True):
        headings, rows = _build_columns(frame, index=index)
        levels = len(frame.index.names) if index else 0
        if _measure_width(headings, rows) <= console.size.width:
            rendered.append((title, _build_table(headings, rows, levels)))
            height += len(rows) + 4
        else:
            rendered.append((title, _build_record_table(headings, rows)))
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
