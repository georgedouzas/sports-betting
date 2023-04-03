"""Module that contains the CLI."""

from __future__ import annotations

from collections.abc import Callable
from inspect import getmembers, isclass
from pathlib import Path

import click
import pandas as pd
from click.decorators import FC
from rich.console import Console
from rich.panel import Panel

from sportsbet import ParamGrid, datasets
from sportsbet.datasets import SoccerDataLoader


def get_name_option() -> Callable[[FC], FC]:
    """Get the name option."""
    return click.option(
        '--name',
        nargs=1,
        required=True,
        type=click.Choice(['soccer']),
        help='The name of the selected sport.',
    )


def get_param_grid_option() -> Callable[[FC], FC]:
    """Get the param grid option."""
    return click.option(
        '--param-grid',
        multiple=True,
        required=False,
        default=None,
        show_default='None',
        type=str,
        help='The parameters grid to select the training data. Indirectly, it also affects the fixtures data.',
    )


def get_dataloader_path_option() -> Callable[[FC], FC]:
    """Get the dataloader path option."""
    return click.option(
        '--dataloader-path',
        nargs=1,
        required=True,
        type=str,
        help='The path of the dataloader.',
    )


def get_data_path_option() -> Callable[[FC], FC]:
    """Get the data path option."""
    return click.option(
        '--data-path',
        nargs=1,
        required=False,
        default=None,
        show_default='None',
        type=str,
        help='The path of the directory to export the training in CSV format.',
    )


def get_parameters_grids(param_grid: tuple) -> ParamGrid | None:
    """Get the parsed param grid as a parameters grid."""
    if not param_grid:
        parameters_grids = None
    else:
        parameters_grids = []
        for grid in param_grid:
            partial_grids = [partial_grid.split(':') for partial_grid in grid.split('|')]
            parameters_grids.append(
                {
                    partial_grid[0].strip(): [val.strip() for val in partial_grid[1].split(',')]
                    for partial_grid in partial_grids
                },
            )
    return parameters_grids


def get_dataloader_class(name: str) -> type[SoccerDataLoader]:
    """Get the dataloader class."""
    return getattr(datasets, f'{name.capitalize()}DataLoader')


def print_console(dfs: list[pd.DataFrame], titles: list[str]) -> None:
    """Print to the console."""
    pd.set_option('display.max_rows', None)
    console = Console()
    with console.pager(styles=True):
        for df, title in zip(dfs, titles, strict=True):
            formatted_title = Panel.fit(f'[bold green]{title}')
            console.print(formatted_title)
            console.print(df)


@click.group()
def main() -> None:
    """CLI for sports-betting.

    This command is executed when you type `sportsbet` or `python -m
    sportsbet`.
    """


@main.group()
def dataloader() -> None:
    """Use or create a dataloader."""


@dataloader.command()
def names() -> None:
    """List the sports names that can be selected."""
    sports = [
        (name.lower().replace('dataloader', ''), 'Yes')
        for name, _ in getmembers(datasets, isclass)
        if not name.startswith('Dummy')
    ]
    sports += [('basketball', 'No'), ('tennis', 'No')]
    sports = pd.DataFrame(sports, columns=['Name', 'Availability'])
    print_console([sports], ['Available names'])


@dataloader.command()
@get_name_option()
def params(name: str) -> None:
    """Show the available parameters to select data for a dataloader."""
    dataloader = get_dataloader_class(name)
    all_params = dataloader.get_all_params()
    all_params = pd.DataFrame([params.values() for params in all_params], columns=list(all_params[0].keys()))
    print_console([all_params], ['Available parameters'])


@dataloader.command()
@get_dataloader_path_option()
def odds_types(dataloader_path: str) -> None:
    """List the odds types that can be selected to extract odds data."""
    dataloader = datasets.load(dataloader_path)
    odds_types = pd.DataFrame(dataloader.get_odds_types(), columns=['Type'])
    print_console([odds_types], ['Available odds types'])


@dataloader.command()
@get_name_option()
@get_param_grid_option()
@get_dataloader_path_option()
def create(name: str, param_grid: tuple, dataloader_path: str) -> None:
    """Create and save a dataloader."""
    parameters_grids = get_parameters_grids(param_grid)
    dataloader = get_dataloader_class(name)(param_grid=parameters_grids)
    dataloader.save(dataloader_path)


@dataloader.command()
@get_dataloader_path_option()
def info(dataloader_path: str) -> None:
    """Get information about the dataloader."""
    dataloader = datasets.load(dataloader_path)
    if not (
        hasattr(dataloader, 'param_grid_')
        and hasattr(dataloader, 'odds_type_')
        and hasattr(dataloader, 'drop_na_thres_')
        and hasattr(dataloader, 'input_cols_')
        and hasattr(dataloader, 'output_cols_')
        and hasattr(dataloader, 'odds_cols_')
    ):
        console = Console()
        warning = Panel.fit(
            '[bold red]No information was extracted. Dataloader has not be used yet to extract training data.',
        )
        console.print(warning)
        return
    param_grid = pd.DataFrame(dataloader.param_grid_)
    train_params = pd.DataFrame(
        {'Odds type': [dataloader.odds_type_], 'Missing values threshold': [dataloader.drop_na_thres_]},
    )
    input_cols = pd.DataFrame({'Columns': dataloader.input_cols_})
    output_cols = pd.DataFrame({'Columns': dataloader.output_cols_})
    odds_cols = pd.DataFrame({'Columns': dataloader.odds_cols_})
    print_console(
        [param_grid, train_params, input_cols, output_cols, odds_cols],
        [
            'Available parameters',
            'Training data extraction parameters',
            'Input columns',
            'Output columns',
            'Odds columns',
        ],
    )


@dataloader.command()
@click.option(
    '--odds-type',
    nargs=1,
    required=False,
    default=None,
    show_default='None',
    type=str,
    help='The type of selected odds of the training data. Indirectly, it also affects the fixtures data.',
)
@click.option(
    '--drop-na-thres',
    nargs=1,
    required=False,
    default=0.0,
    show_default=True,
    type=float,
    help='The threshold to decide whether columns with missing are dropped for the training data. It should be a '
    'value in [0.0, 1.0], while higher values drop more columns. Indirectly, it also affects the fixtures data.',
)
@get_dataloader_path_option()
@get_data_path_option()
def training(
    odds_type: str,
    drop_na_thres: float,
    dataloader_path: str,
    data_path: str | None,
) -> None:
    """Use a dataloader to display and export the training data."""
    dataloader = datasets.load(dataloader_path)
    X_train, Y_train, O_train = dataloader.extract_train_data(odds_type=odds_type, drop_na_thres=drop_na_thres)
    dataloader.save(dataloader_path)
    print_console(
        [X_train, Y_train] + ([O_train] if O_train is not None else []),
        ['Training input data', 'Training output data'] + (['Training odds data'] if O_train is not None else []),
    )
    if data_path is not None:
        if Path(data_path).exists():
            X_train.to_csv((Path(data_path) / 'X_train.csv').absolute())
            Y_train.to_csv((Path(data_path) / 'Y_train.csv').absolute(), index=False)
            if O_train is not None:
                O_train.to_csv((Path(data_path) / 'O_train.csv').absolute(), index=False)
        else:
            console = Console()
            warning = Panel.fit(
                f'[bold red]Data were not exported. Directory {Path(data_path).absolute()} does not exist.',
            )
            console.print(warning)


@dataloader.command()
@get_dataloader_path_option()
@get_data_path_option()
def fixtures(
    dataloader_path: str,
    data_path: str | None,
) -> None:
    """Use a dataloader to display and export the fixtures data."""
    dataloader = datasets.load(dataloader_path)
    if not (
        hasattr(dataloader, 'param_grid_')
        and hasattr(dataloader, 'odds_type_')
        and hasattr(dataloader, 'drop_na_thres_')
        and hasattr(dataloader, 'input_cols_')
        and hasattr(dataloader, 'output_cols_')
        and hasattr(dataloader, 'odds_cols_')
    ):
        console = Console()
        warning = Panel.fit(
            '[bold red]No information was extracted. Dataloader has not be used yet to extract training data.',
        )
        console.print(warning)
        return
    X_fix, _, O_fix = dataloader.extract_fixtures_data()
    if not X_fix.empty:
        print_console([X_fix], ['Fixtures input data'])
        if O_fix is not None and not O_fix.empty:
            print_console([O_fix], ['Fixtures odds data'])
    else:
        console = Console()
        warning = Panel.fit(
            '[bold red]Fixtures data were empty',
        )
        console.print(warning)
    if data_path is not None:
        if Path(data_path).exists():
            if not X_fix.empty:
                X_fix.to_csv((Path(data_path) / 'X_fix.csv').absolute())
                if O_fix is not None and not O_fix.empty:
                    O_fix.to_csv((Path(data_path) / 'O_fix.csv').absolute(), index=False)
        else:
            console = Console()
            warning = Panel.fit(
                f'[bold red]Data were not exported. Directory {Path(data_path).absolute()} does not exist.',
            )
            console.print(warning)


@main.group()
def bettor() -> None:
    """Backtest a bettor and predict the value bets."""
