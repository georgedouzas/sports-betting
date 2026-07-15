"""Module that contains the main function of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import click

from ._betting import evaluation
from ._data import dataloader


@click.group()
def main() -> None:
    """Create, test and use sports betting models.

    The commands mirror the Python API: `dataloader` selects, downloads and extracts the data, and `evaluation`
    backtests, fits and bets with a model on it. `dataloader train extract` saves a dataloader that the evaluation
    commands read, so the data is downloaded once and reused.
    """
    return


for group in (dataloader, evaluation):
    main.add_command(group)
