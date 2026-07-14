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

from ._options import DOWNLOAD, EXTRACTION, OUTPUT, SELECTION, options
from ._utils import downloaded, extraction, print_console, reported, selected


@click.group()
def data() -> None:
    """Select, download and extract data."""
    return


@data.command()
@options(SELECTION)
def params(**selection: object) -> None:
    """Show what can be selected.

    It asks the source, since nothing can be selected before it is known what exists.
    """
    with reported(), selected(selection) as dataloader:
        if dataloader is None:
            return
        stats_source, *_ = dataloader.sources
        available = stats_source.available_params()
        cols = list({param for params in available for param in params})
        frame = pd.DataFrame({col: [params.get(col, '-') for params in available] for col in cols})
        print_console([frame], ['Available parameters'], index=False)


@data.command(name='odds-types')
@options(SELECTION, DOWNLOAD)
def odds_types(**selection: object) -> None:
    """Show the odds that can be extracted."""
    with reported(), selected(selection) as dataloader:
        if dataloader is None:
            return
        dataloader._fetched(downloaded(selection)['download'])
        frame = pd.DataFrame(dataloader.get_odds_types(), columns=['Type'])
        print_console([frame], ['Available odds types'], index=False)


@data.command()
@options(SELECTION, EXTRACTION, DOWNLOAD, OUTPUT)
def training(data_path: str | None, **selection: object) -> None:
    """Extract the training data."""
    with reported(), selected(selection) as dataloader:
        if dataloader is None:
            return
        X_train, Y_train, O_train = dataloader.extract_train_data(**extraction(selection), **downloaded(selection))
        Y_train = cast('pd.DataFrame', Y_train)
        has_odds = O_train is not None and not O_train.empty
        print_console(
            [X_train, Y_train] + ([O_train] if has_odds else []),
            ['Training input data', 'Training output data'] + (['Training odds data'] if has_odds else []),
        )
        if data_path is not None:
            written = Path(data_path) / 'sports-betting-data'
            written.mkdir(parents=True, exist_ok=True)
            X_train.to_csv(written / 'X_train.csv')
            Y_train.to_csv(written / 'Y_train.csv')
            if has_odds:
                O_train.to_csv(written / 'O_train.csv')


@data.command()
@options(SELECTION, EXTRACTION, DOWNLOAD, OUTPUT)
def fixtures(data_path: str | None, **selection: object) -> None:
    """Extract the games that have not been played yet."""
    with reported(), selected(selection) as dataloader:
        if dataloader is None:
            return
        dataloader.extract_train_data(**extraction(selection), **downloaded(selection))
        X_fix, _, O_fix = dataloader.extract_fixtures_data()
        if X_fix.empty:
            Console().print(Panel.fit('[bold red]Fixtures data were empty'))
            return
        has_odds = O_fix is not None and not O_fix.empty
        print_console(
            [X_fix] + ([O_fix] if has_odds else []),
            ['Fixtures input data'] + (['Fixtures odds data'] if has_odds else []),
        )
        if data_path is not None:
            written = Path(data_path) / 'sports-betting-data'
            written.mkdir(parents=True, exist_ok=True)
            X_fix.to_csv(written / 'X_fix.csv')
            if has_odds:
                O_fix.to_csv(written / 'O_fix.csv')
