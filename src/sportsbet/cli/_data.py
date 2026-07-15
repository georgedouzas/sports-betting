"""Module that contains the dataloader commands of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from ._options import DATALOADER, EXTRACTION, OUTPUT, SELECTION, options
from ._utils import extraction, load_dataloader, print_console, reported, save_dataloader, selected


@click.group()
def dataloader() -> None:
    """Select, download and extract data.

    `params` and `odds-types` show what a source publishes. `train extract` downloads the seasons and saves the
    dataloader, and `fixtures extract` downloads the upcoming matches of a saved dataloader.
    """
    return


@dataloader.command()
@options(SELECTION)
def params(**selection: object) -> None:
    """Show the leagues, divisions and seasons a source publishes.

    It asks the source, which is where a `param_grid` starts.
    """
    with reported(), selected(selection) as loader:
        if loader is None:
            return
        stats_source, *_ = loader.sources
        available = stats_source.available_params()
        cols = list({param for params in available for param in params})
        frame = pd.DataFrame({col: [params.get(col, '-') for params in available] for col in cols})
        print_console([frame], ['Available parameters'], index=False)


@dataloader.command(name='odds-types')
@options(SELECTION)
def odds_types(**selection: object) -> None:
    """Show the odds types a selection carries.

    Downloads the data to read them.
    """
    with reported(), selected(selection) as loader:
        if loader is None:
            return
        frame = pd.DataFrame(loader.get_odds_types(), columns=['Type'])
        print_console([frame], ['Available odds types'], index=False)


@dataloader.group()
def train() -> None:
    """Work with the training data a model learns from."""
    return


@train.command(name='extract')
@options(SELECTION, EXTRACTION)
@click.option('--output', '-o', 'output', required=True, type=click.Path(), help='Where to save the dataloader.')
def extract_train(output: str, **selection: object) -> None:
    """Download the training data and save the dataloader to a file.

    This is the only command that downloads the seasons. The evaluation commands read the file it writes, so the data is
    downloaded once and reused.
    """
    with reported(), selected(selection) as loader:
        if loader is None:
            return
        X_train, Y_train, O_train = loader.extract_train_data(**extraction(selection))
        save_dataloader(output, loader, (X_train, Y_train, O_train))
        has_odds = O_train is not None and not O_train.empty
        frames = [X_train, *([Y_train] if Y_train is not None else []), *([O_train] if has_odds else [])]
        titles = [
            'Training input data',
            *(['Training output data'] if Y_train is not None else []),
            *(['Training odds data'] if has_odds else []),
        ]
        print_console(frames, titles)
        Console().print(f'Saved the dataloader to [bold]{output}[/bold].')


@dataloader.group()
def fixtures() -> None:
    """Work with the upcoming matches a model bets on."""
    return


@fixtures.command(name='extract')
@options(DATALOADER, OUTPUT)
def extract_fixtures(dataloader_path: str, data_path: str | None) -> None:
    """Download the upcoming matches of a saved dataloader."""
    with reported():
        loader, _ = load_dataloader(dataloader_path)
        X_fix, _, O_fix = loader.extract_fixtures_data()
        if X_fix.empty:
            Console().print(Panel.fit('[bold red]There are no upcoming matches.'))
            return
        has_odds = O_fix is not None and not O_fix.empty
        print_console(
            [X_fix, *([O_fix] if has_odds else [])],
            ['Fixtures input data', *(['Fixtures odds data'] if has_odds else [])],
        )
        if data_path is not None:
            written = Path(data_path) / 'sports-betting-data'
            written.mkdir(parents=True, exist_ok=True)
            X_fix.to_csv(written / 'X_fix.csv')
            if has_odds:
                O_fix.to_csv(written / 'O_fix.csv')
