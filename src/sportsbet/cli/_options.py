"""Module that contains the options of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable

import click
from click.decorators import FC


def get_dataloader_config_path_option() -> Callable[[FC], FC]:
    """Get the dataloader configuration file path option."""
    return click.option(
        '--dataloader-config-path',
        '-d',
        nargs=1,
        required=True,
        type=str,
        help='The path of the dataloader configuration file.',
    )


def get_bettor_config_path_option() -> Callable[[FC], FC]:
    """Get the bettor configuration file path option."""
    return click.option(
        '--bettor-config-path',
        '-b',
        nargs=1,
        required=True,
        type=str,
        help='The path of the bettor configuration file.',
    )
