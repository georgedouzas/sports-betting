"""Module that contains the main function of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import click

from ._betting import model
from ._data import data


@click.group()
def main() -> None:
    """CLI for sports-betting.

    This command is executed when you type `sportsbet` or `python -m sportsbet`.
    """
    return


main.add_command(data)
main.add_command(model)
