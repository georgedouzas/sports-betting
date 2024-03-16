"""Module that contains the main function of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import click

from ._betting import bettor
from ._data import dataloader


@click.group()
def main() -> None:
    """CLI for sports-betting.

    This command is executed when you type `sportsbet` or `python -m
    sportsbet`.
    """
    return


main.add_command(dataloader)
main.add_command(bettor)
