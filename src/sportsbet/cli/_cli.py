"""Run the command line."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import click

from ._betting import evaluation
from ._data import dataloader
from ._execution import execution


@click.group()
def main() -> None:
    """Create, test and use sports betting models.

    The commands mirror the Python API. `dataloader` selects, downloads and extracts the data, `evaluation` backtests,
    fits and bets with a model on it, and `execution` places those bets at a venue.
    """
    return


for group in (dataloader, evaluation, execution):
    main.add_command(group)
